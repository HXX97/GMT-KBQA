import argparse
import os
import glob
import timeit
from typing import OrderedDict
import torch
from functools import partial
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import trange, tqdm
from config import load_untrained_model, register_args, set_seed
from components.utils import load_json, mkdir_p, dump_json
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    WEIGHTS_NAME
)
import sys
import numpy as np
from config import get_model_class
import logging
from executor.sparql_executor import get_label_with_odbc
logger = logging.getLogger(__name__)

# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
#     from tensorboardX import SummaryWriter

from inputDataset.disamb_dataset import load_and_cache_disamb_examples


def train(args, train_dataset, model, tokenizer):
    if args.local_rank in [-1, 0]:
        # tb_writer = SummaryWriter()
        mkdir_p(args.output_dir)
    
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_collate_fn = partial(disamb_collate_fn, tokenizer = tokenizer)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_collate_fn)

    # set train steps by args
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and scheduler (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max(args.warmup_steps, t_total * args.warmup_ratio)),
        num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
    
    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Warmup steps = %d", int(max(args.warmup_steps, t_total * args.warmup_ratio)))
    logger.info("  Total optimization steps = %d", t_total)


    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()    
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )

    # Added here for reproductibility
    set_seed(args)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "token_type_ids": batch[1],
                "attention_mask": batch[2],
                "sample_mask": batch[3],
                "labels": batch[4],
            }

            # token_type_ids not needed for roberta
            if args.model_type in ["roberta", "distilbert", "camembert", "bart"]:
                del inputs["token_type_ids"]

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps >1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # gradient clip and normalization
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step() # update learning rate schedule
                model.zero_grad()
                global_step +=1

                # Log information
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    logs['epoch'] = _+(step+1)/len(epoch_iterator)
                    logs['learning_rate'] = scheduler.get_last_lr()[0]
                    logs['loss'] = (tr_loss - logging_loss) / args.logging_steps
                    logs['step'] = global_step
                    # tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    # tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info("Training logs: {}".format(logs))
                    logging_loss = tr_loss


                # Log metrics
                if args.local_rank in [-1, 0] and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        # for key, value in results.items():
                        #     tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        logger.info("Eval results: {}".format(dict(results)))
                
                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    # if args.local_rank in [-1,0]:
    #     tb_writer.close()
    
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, output_prediction=False):
    # load examples
    dataset, examples = load_and_cache_disamb_examples(args, tokenizer, evaluate=True, output_examples=True)
    if not os.path.exists(args.output_dir) and args.local_rank in [-1,0]:
        os.makedirs(args.output_dir)
    
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=partial(disamb_collate_fn, tokenizer=tokenizer))

    # multi-gpu evaluate evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    
    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    start_time = timeit.default_timer()

    all_pred_indexes = []
    all_pred_logits = []
    all_labels = []
    

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "token_type_ids": batch[1],
                "attention_mask": batch[2],
                "sample_mask": batch[3],
                "labels": batch[4],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart"]:
                del inputs["token_type_ids"]
            
            logits = model(**inputs)[1] # len of candidate entities
            pred_indexes = torch.argmax(logits, 1).detach().cpu()
            

        all_pred_indexes.append(pred_indexes)
        all_labels.append(batch[4].cpu())
        all_pred_logits.extend(logits.cpu().numpy().tolist())

    all_pred_indexes = torch.cat(all_pred_indexes).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = np.sum(all_pred_indexes == all_labels) / len(all_pred_indexes)
    evalTime = timeit.default_timer()-start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))
    coverage = coverage_evaluation(examples, dataset, all_pred_indexes)
    results = {'num problem': len(all_pred_indexes), 'acc': acc, 'cov': coverage}
    # results = {}

    saving_predictions = OrderedDict([(feat.pid, pred) for feat, pred in zip(dataset, all_pred_indexes.tolist())])
    saving_logits = OrderedDict([(feat.pid, pred_logits) for feat, pred_logits in zip(dataset,all_pred_logits)])

    # print(saving)
    if output_prediction:
        dump_json(saving_predictions, os.path.join(args.output_dir, 'predictions.json'))
        dump_json(saving_logits, os.path.join(args.output_dir, 'predict_logits.json'))

        split_file = args.predict_file if evaluate else args.train_file
        dataset_id = os.path.basename(split_file).split('_')[0]
        split_id = os.path.basename(split_file).split('_')[1]
        
        # rank the candidate entities with predicted logits
        get_candidate_entity_linking_with_logits(dataset_id, split_id)

    return results


def get_candidate_entity_linking_with_logits(dataset, split):
    print(f'Preparing candidate entity linking results with logits for {dataset}_{split}')
    logits_bank = load_json(f'data/{dataset}/entity_retrieval/candidate_entities/disamb_results/{dataset}_{split}/predict_logits.json')
    candidate_bank = load_json(f'data/{dataset}/entity_retrieval/candidate_entities/{dataset}_{split}_entities_facc1_unranked.json')
    res = OrderedDict()
    
    for qid,data in tqdm(candidate_bank.items(),total=len(candidate_bank),desc=f'Processing {split}'):
        problem_num = len(data)
        if problem_num ==0:
            res[qid]=[]
        entity_list = []

        problem_id = -1
        for problem in data:
            problem_id+=1
            logits = logits_bank.get(qid+'#'+str(problem_id),[1.0]*len(problem))
            for idx,cand_ent in enumerate(problem):
                logit = logits[idx]
                # cand_ent['label'] = get_label_with_odbc(cand_ent['id'])
                cand_ent['logit'] = logit
                entity_list.append(cand_ent)

        entity_list.sort(key=lambda x:x['logit'],reverse=True)

        res[qid] = entity_list
    
    dump_json(res,f'data/{dataset}/entity_retrieval/candidate_entities/{dataset}_{split}_cand_entities_facc1.json',indent=4,ensure_ascii=False)

 
def coverage_evaluation(instances, valid_features, predicted_indexes):
    # build result index
    indexed_pred = dict([(feat.pid, pred) for feat, pred in zip(valid_features,predicted_indexes)])

    covered = 0
    for inst in instances:
        gt_entities = inst.target_entities
        pred_entities = []
        for problem in inst.disamb_problems:
            if len(problem.candidates)==0:
                continue
            if len(problem.candidates) ==1 or problem.target_id is None:
                pred_entities.append(problem.candidates[0].id)
                continue
                
            pred_idx = indexed_pred[problem.pid]
            pred_entities.append(problem.candidates[pred_idx].id)

        if set(gt_entities).issubset(set(pred_entities)):
            covered +=1
    
    coverage = covered / len(instances)
    return coverage
    

def _collect_constrastive_inputs(feat, num_sample, dummy_inputs):
    """dynamically pad inputs"""
    input_ids = []
    token_type_ids = []
    sample_mask = []

    input_ids.extend(feat.candidate_input_ids)
    token_type_ids.extend(feat.candidate_token_type_ids)
    filled_num = len(input_ids)

    # force padding
    for _ in range(filled_num, num_sample):
        input_ids.append(dummy_inputs['input_ids'])
        token_type_ids.append(dummy_inputs['token_type_ids'])
    
    sample_mask = [1]* filled_num + [0]*(num_sample-filled_num)
    
    return input_ids, token_type_ids, sample_mask


def disamb_collate_fn(data, tokenizer):
    dummpy_inputs = tokenizer('','', return_token_type_ids=True)
    # batch size
    # input_id: B * N_Sample * L
    # token_type: B * N_Sample * L
    # attention_mask: B * N_Sample * N
    # sample_mask: B * N_Sample
    # labels: B, all zero
    
    batch_size = len(data)
    num_sample = max([len(x.candidate_input_ids) for x in data])
    
    all_input_ids = []
    all_token_type_ids = []
    all_sample_masks = []

    for feat in data:
        input_ids, token_type_ids, sample_mask = _collect_constrastive_inputs(feat, num_sample, dummpy_inputs)

        all_input_ids.extend(input_ids)
        all_token_type_ids.extend(token_type_ids)
        all_sample_masks.append(sample_mask)

    encoded = tokenizer.pad({'input_ids': all_input_ids, 'token_type_ids': all_token_type_ids},return_tensors='pt')
    all_sample_masks = torch.BoolTensor(all_sample_masks)
    labels = torch.LongTensor([x.target_idx for x in data])

    all_input_ids = encoded['input_ids'].view((batch_size, num_sample, -1))
    all_token_type_ids = encoded['token_type_ids'].view((batch_size, num_sample, -1))
    all_attention_masks = encoded['attention_mask'].view((batch_size, num_sample, -1))

    return all_input_ids, all_token_type_ids, all_attention_masks, all_sample_masks, labels



def main():
    # parse args
    parser = argparse.ArgumentParser()
    register_args(parser)
    args = parser.parse_args()

    # check output dir
    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # args.server_ip = '0.0.0.0'
    # args.server_port = '12346'

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach", flush=True)
        ptvsd.enable_attach(address=(args.server_ip, args.server_port))
        ptvsd.wait_for_attach()


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    args.logger = logger

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()

    # load model for training
    config, tokenizer, model = load_untrained_model(args)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()
    
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_disamb_examples(args, tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = get_model_class(args).from_pretrained(args.output_dir)  # , force_download=True)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)


    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1,0]:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            checkpoints = [args.model_name_or_path]
            
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        
        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = get_model_class(args).from_pretrained(checkpoint)  # , force_download=True)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, output_prediction=True)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))        

    return results

    
if __name__ == "__main__":
    main()