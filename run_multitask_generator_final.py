from collections import defaultdict
import os
import copy
import random
import argparse
import numpy as np
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup
from generation.models.T5_models_final import (
    T5_generation, 
    T5_generation_concat,
    T5_Multitask_Relation_Concat, 
    T5_MultiTask_Relation_Entity_Concat,
    T5_Multitask_Entity_Concat
)
from inputDataset.gen_mtl_dataset import MTLGenDataset, MTLGenerationExample
from components.utils import dump_json, load_json
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch
from tqdm import tqdm
from functools import partial

"""
Important: Edit this when change source data
""" 
def load_data(split, args):
    if args.dataset_type == "CWQ":
        data_file_name = 'data/CWQ/generation/merged_0724_ep1/CWQ_{}.json'.format(split)
        # data_file_name = 'data/CWQ/generation/merged_0715_retrain/CWQ_{}.json'.format(split)
        # data_file_name = 'data/CWQ/generation/merged/CWQ_{}.json'.format(split)
        # data_file_name = 'data/CWQ/generation/merged_0714/CWQ_{}.json'.format(split)
        # data_file_name = 'data/CWQ/generation/merged_old/CWQ_{}.json'.format(split)
        # data_file_name = 'data/CWQ/generation/xwu_merged_new/CWQ_{}.json'.format(split)
    elif args.dataset_type == "WebQSP":
        # data_file_name = 'data/WebQSP/generation/0722/merged_question_relation_ep3_2hop/WebQSP_{}.json'.format(split)
        data_file_name = 'data/WebQSP/generation/merged_relation_final/WebQSP_{}.json'.format(split)
        # data_file_name = 'data/WebQSP/generation/merged_yhshu/WebQSP_{}.json'.format(split)
        # data_file_name = 'data/WebQSP/generation/merged_0715_retrain_biencoder_ep5_reserve_150/WebQSP_{}.json'.format(split)
        # data_file_name = 'data/WebQSP/generation/merged_0715_retrain_biencoder_ep5/WebQSP_{}.json'.format(split)
        # data_file_name = 'data/WebQSP/generation/merged_0715_retrain/WebQSP_{}.json'.format(split)
        # if split == 'train':
        #     data_file_name = 'data/WebQSP/generation/merged_old/WebQSP_{}.json'.format(split)
        # elif split == 'test':
        #     data_file_name = 'data/WebQSP/generation/merged_0714/WebQSP_test.json'
        # data_file_name = 'data/WebQSP/generation/merged_old/WebQSP_{}.json'.format(split)
        # data_file_name = 'data/WebQSP/generation/xwu_merged_new/WebQSP_{}.json'.format(split)
    print('Loading data from:',data_file_name)
    data_dict = load_json(data_file_name)
    return data_dict


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_debug',default='False',help='whether to do training')
    parser.add_argument('--do_train',default=False,action='store_true',help='whether to do training')
    parser.add_argument('--do_eval',default=False,action='store_true',help='whether to do eval when training')
    parser.add_argument('--do_predict',default=False,action='store_true',help='whether to do prediction')
    parser.add_argument('--predict_split',default='test',help='which dataset to perform prediction')
    parser.add_argument('--pretrained_model_path', default="t5-base", help='model name like "t5-base" or a local directory with t5 model in it')
    parser.add_argument('--model_save_dir', default="exps/gen_multitask/model_saved", help='model path for saving and loading model')
    parser.add_argument('--max_src_len',default=256, type=int, help='maximum source length')
    parser.add_argument('--max_tgt_len',default=196, type=int, help='maximum target length')
    parser.add_argument('--train_batch_size', default=8, type=int, help='batch_size for training')
    parser.add_argument('--eval_batch_size', default=8, type=int, help='batch_size for evaluation')
    parser.add_argument('--test_batch_size',default=4, type=int, help='batch_size for testing')
    parser.add_argument('--lr',default=2e-5,type=float,help='learning_rate')
    parser.add_argument('--weight_decay',default=1e-3,type=float,help='weight_decay')
    parser.add_argument('--epochs',default=15,type=int,help='epochs')
    parser.add_argument('--iters_to_accumulate',default=1,type=int,help='the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size')
    parser.add_argument('--print_every',default=100,type=int,help='every steps to print training information')
    parser.add_argument('--save_every_epochs',default=10,type=int,help='save the model every n eopchs')
    parser.add_argument('--warmup_ratio',default=0.1,type=float,help='the ratio of warm up steps')
    parser.add_argument('--output_dir',default='exps/gen_multitask',help='where to save model')
    parser.add_argument('--overwrite_output_dir',default=False,action='store_true',help='whether to overwrite the output dir')
    parser.add_argument('--eval_beams',default=50,type=int, help="beam size for generating")
    parser.add_argument('--do_lower',default=False,action='store_true',help='whether to do lower for both inputs and outputs')
    parser.add_argument('--normalize_relations', default=False, action='store_true', help="normalize relations when concatenating it to generation model input")
    parser.add_argument('--sample_size', default=10, type=int, help="number of candidate relations/entities")
    parser.add_argument('--cross_entropy_loss', default=False, action='store_true', help="False to use BCEWithLogitsLoss; True to use CrossEntropyLoss")
    parser.add_argument('--add_prefix', default=False, action='store_true', help="add prefix for classification task")
    parser.add_argument('--model', default='T5_generation', type=str, help="T5_generation | T5_generation_concat | T5_Multitask_Relation_Concat | T5_MultiTask_Relation_Entity | T5_Multitask_Relation_Entity_Concat | T5_SExpr_Generation_Structure_Generation | T5_SExpr_Generation_Structure_Generation_Concat | T5_Structure_Classification | T5_Multitask_Entity_Concat")
    parser.add_argument('--dataset_type', default="CWQ", type=str, help="CWQ | WebQSP")
    parser.add_argument('--warmup_epochs', default=0, type=int, help="for concat models, starts concat after warmup_epochs")
    parser.add_argument('--concat_golden', default=False, action='store_true', help="concat golden relations/entities to input")
    """
    deprecated
    """
    parser.add_argument('--structure_gen_beam_size', default=1, type=int)
    parser.add_argument('--max_structure_tgt_len', default=70, type=int)
    parser.add_argument('--structure_sample_size', default=5, type=int, help="number of candidate structures")
    parser.add_argument('--use_rich_relation', default=False, action='store_true', help="use rich relation for classification")
    parser.add_argument('--structure_syntax_check', default=False, action='store_true', help="conduct syntax check on structures")
    parser.add_argument('--train_concat_true', default=False, action='store_true', help="only concat true classification results during training")
    args = parser.parse_args()
    return args
  

def generate_candidate_entity_map_classification_res(predictions, dirname, dataset, args):
    """
    generate candidate_entity_map according to output of entity disambiguation task
    for entities with same label(linked by same question), sort by prediction logits.
    """
    predicted_entities = defaultdict(dict)

    assert len(predictions) == len(dataset), print(len(predictions), len(dataset))

    for (pred, data) in zip(predictions, dataset):
        qid = data["ID"]
        pred_clf_logits = pred["pred_entity_clf_labels"]
        pred_clf_indexes = [idx for (idx, value) in enumerate(pred_clf_logits) if float(value) > 0.5]
        for idx in pred_clf_indexes:
            cand_entity = data["cand_entity_list"][idx]
            logits = float(pred_clf_logits[idx])
            if cand_entity['label'].lower() in predicted_entities[qid]:
                # entity with same logist, sort by prediction logits
                prev_logit = predicted_entities[qid][cand_entity['label'].lower()]['pred_logits']
                if logits > prev_logit:
                    predicted_entities[qid][cand_entity['label'].lower()] = {
                        'id': cand_entity['id'],
                        'pred_logits': logits
                    }
            else:
                predicted_entities[qid][cand_entity['label'].lower()] = {
                    'id': cand_entity['id'],
                    'pred_logits': logits
                }
    
    if args.dataset_type == "CWQ":
        dump_json(predicted_entities, os.path.join(dirname, f'CWQ_{args.predict_split}_{args.test_batch_size}_candidate_entity_map.json'))
    elif args.dataset_type == "WebQSP":
        dump_json(predicted_entities, os.path.join(dirname, f'WebQSP_{args.predict_split}_{args.test_batch_size}_candidate_entity_map.json'))


def _collate_fn(data,tokenizer):
    """For mini-batch dynamic padding"""
    all_src_input_ids = []
    all_tgt_input_ids = []
    all_relation_clf_pair_input_ids = []
    all_relation_clf_pair_labels = []
    candidate_relations = []
    input_src = []
    all_entity_clf_pair_input_ids = []
    all_entity_clf_pair_labels = []
    rich_candidate_entities_list = []
    # all_structure_tgt_input_ids = []
    # candidate_structures = []
    # all_structure_clf_pair_labels = []
    all_rich_relation_clf_pair_input_ids = []
    candidate_rich_relations = []
    # all_structure_clf_pair_input_ids = []
    all_src_concatenated_input_ids = []
    all_src_golden_concatenated_input_ids = []
    # print(len(data))
    for data_tuple in data:
        # print(data_tuple)
        all_src_input_ids.append(data_tuple[0])
        all_tgt_input_ids.append(data_tuple[1])
        all_relation_clf_pair_input_ids.extend(data_tuple[2])
        all_relation_clf_pair_labels.extend(data_tuple[3])
        input_src.extend(data_tuple[4])
        candidate_relations.extend(data_tuple[5])
        all_entity_clf_pair_input_ids.extend(data_tuple[6])
        all_entity_clf_pair_labels.extend(data_tuple[7])
        rich_candidate_entities_list.extend(data_tuple[8])
        #all_structure_tgt_input_ids.append(data_tuple[9])
        #candidate_structures.extend(data_tuple[10])
        #all_structure_clf_pair_labels.extend(data_tuple[11])
        all_rich_relation_clf_pair_input_ids.extend(data_tuple[9])
        candidate_rich_relations.extend(data_tuple[10])
        #all_structure_clf_pair_input_ids.extend(data_tuple[14])
        all_src_concatenated_input_ids.append(data_tuple[11])
        all_src_golden_concatenated_input_ids.append(data_tuple[12])

    
    src_encoded = tokenizer.pad({'input_ids': all_src_input_ids},return_tensors='pt')
    tgt_encoded = tokenizer.pad({'input_ids': all_tgt_input_ids},return_tensors='pt')
    relation_clf_pair_encoded = tokenizer.pad({'input_ids': all_relation_clf_pair_input_ids},return_tensors='pt')
    relation_clf_pair_labels = torch.tensor(all_relation_clf_pair_labels)
    rich_relation_clf_pair_encoded = tokenizer.pad({'input_ids': all_rich_relation_clf_pair_input_ids},return_tensors='pt')
    entity_clf_pair_encoded = tokenizer.pad({'input_ids': all_entity_clf_pair_input_ids},return_tensors='pt')
    entity_clf_pair_labels = torch.tensor(all_entity_clf_pair_labels)
    # structure_tgt_encoded = tokenizer.pad({'input_ids': all_structure_tgt_input_ids},return_tensors='pt')
    # structure_clf_pair_labels = torch.tensor(all_structure_clf_pair_labels)
    # structure_clf_pair_encoded = tokenizer.pad({'input_ids': all_structure_clf_pair_input_ids},return_tensors='pt')
    src_concatenated_encoded = tokenizer.pad({'input_ids': all_src_concatenated_input_ids},return_tensors='pt')
    src_golden_concatenated_encoded = tokenizer.pad({'input_ids': all_src_golden_concatenated_input_ids},return_tensors='pt')

    return (
        src_encoded,
        tgt_encoded,
        relation_clf_pair_encoded,
        relation_clf_pair_labels,
        input_src,
        candidate_relations,
        entity_clf_pair_encoded,
        entity_clf_pair_labels,
        rich_candidate_entities_list,
        # structure_tgt_encoded,
        # candidate_structures,
        # structure_clf_pair_labels,
        rich_relation_clf_pair_encoded,
        candidate_rich_relations,
        # structure_clf_pair_encoded,
        src_concatenated_encoded,
        src_golden_concatenated_encoded
    )


def prepare_dataloader(args,split,tokenizer,batch_size):
    assert split in ['train','test','dev','train_sample','dev_sample','test_sample']

    data = load_data(split, args)
    print(f'Origin {split} dataset len: {len(data)}')
    assert type(data)==list
    if 'train' in split or 'dev' in split:
        # for train and dev, filter the examples without sexpr
        examples = []
        for x in data:
            if x['sexpr'].lower()!="null":
                examples.append(MTLGenerationExample(x))                
    else:
        examples = [MTLGenerationExample(x) for x in data]
    print(f'Real {split} dataset len: {len(examples)}')

    # examples = examples[:100]
    dataset = MTLGenDataset(examples, 
                            tokenizer=tokenizer,
                            do_lower=args.do_lower,
                            normalize_relations=args.normalize_relations,
                            max_src_len=args.max_src_len,
                            max_tgt_len=args.max_tgt_len,
                            max_structure_tgt_len=args.max_structure_tgt_len,
                            add_prefix=args.add_prefix
                            )

    # print(train_dataset.__getitem__(0))
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            collate_fn=partial(_collate_fn,tokenizer=tokenizer),
                            shuffle=False
                            )
    return dataloader


def save_model(model_save_dir,model_to_save,epoch,is_final_epoch=False):
    if is_final_epoch:
        output_model_file = os.path.join(model_save_dir,'pytorch_model.bin')    
    else:
        output_model_file = os.path.join(model_save_dir,f'pytorch_model_epoch_{epoch}.bin')
    
    output_config_file = os.path.join(model_save_dir,'config_file.json')
    # output_vocab_file = os.path.join(model_save_dir,'vocab_file.bin')
    output_tokenizer_dir = os.path.join(model_save_dir,'custom_tokenizer')
    
    torch.save(model_to_save.state_dict(),output_model_file)
    model_to_save.t5.config.to_json_file(output_config_file)
    tokenizer.save_pretrained(output_tokenizer_dir)
    # tokenizer.save_vocabulary(output_vocab_file)
    if is_final_epoch:
        print("The final model has been saved at {}".format(output_model_file))
    else:
        print("The model of eopch {} has been saved at {}".format(epoch,output_model_file))



def train_model(args,model,tokenizer,device,train_dataloader,dev_dataloader=None,model_save_dir=None):
    # train
    print('Start training...')
    # set parameters
    lr = args.lr # learning rate
    iters_to_accumulate = args.iters_to_accumulate  # the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size
    print_every = args.print_every
    # set weight_decay for different parameters
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }
    ]
    opti = AdamW(
                optimizer_grouped_parameters, 
                lr=lr, 
                )
    # opti = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    warmup_ratio = args.warmup_ratio # The number of steps for the warmup phase.
    # num_training_steps = epochs * len(train_dataloader)  # The total number of training steps
    t_total = (len(train_dataloader) // iters_to_accumulate) * epochs  # Necessary to take into account Gradient accumulation
    lr_scheduler = get_linear_schedule_with_warmup(
                                        optimizer=opti,
                                        num_warmup_steps=t_total * warmup_ratio,
                                        num_training_steps=t_total
                                        )
    # scaler = GradScaler()
    best_loss = np.Inf
    best_epoch = 1
    num_iterations = len(train_dataloader)

    # dir to save model
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # train step
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for it,data in enumerate(tqdm(train_dataloader,desc=f'Epoch {epoch+1}')):
            src_encoded = data[0]
            tgt_encoded = data[1]
            relation_clf_pair_encoded = data[2]
            relation_clf_pair_labels = data[3]
            input_src = data[4]
            candidate_relations = data[5]
            entity_clf_pair_encoded = data[6]
            entity_clf_pair_labels = data[7]
            rich_textual_candidate_entities_list = data[8]
            # structure_tgt_encoded = data[9]
            # candidate_structures = data[10]
            # structure_clf_pair_labels = data[11]
            rich_relation_clf_pair_encoded = data[9]
            candidate_rich_relations = data[10]
            # structure_clf_pair_encoded = data[14]
            src_concatenated_encoded = data[11]
            src_golden_concatenated_encoded = data[12]
            # print(data)

            if isinstance(model, T5_generation):
                loss = model(
                    input_ids_gen=src_encoded['input_ids'].to(device),
                    gen_labels=tgt_encoded['input_ids'].to(device),
                    gen_attention_mask=src_encoded['attention_mask'].to(device),
                )
            elif isinstance(model, T5_generation_concat):
                if args.concat_golden:
                    loss = model(
                        input_ids_gen=src_golden_concatenated_encoded['input_ids'].to(device),
                        gen_labels=tgt_encoded['input_ids'].to(device),
                        gen_attention_mask=src_golden_concatenated_encoded['attention_mask'].to(device),
                    )
                else:
                    loss = model(
                        input_ids_gen=src_concatenated_encoded['input_ids'].to(device),
                        gen_labels=tgt_encoded['input_ids'].to(device),
                        gen_attention_mask=src_concatenated_encoded['attention_mask'].to(device),
                    )
            elif isinstance(model, T5_Multitask_Relation_Concat):
                loss = model(
                    input_ids_gen=src_encoded['input_ids'].to(device),
                    input_ids_clf=relation_clf_pair_encoded['input_ids'].to(device),
                    gen_labels=tgt_encoded['input_ids'].to(device),
                    clf_labels=relation_clf_pair_labels.to(device),
                    clf_attention_mask=relation_clf_pair_encoded['attention_mask'].to(device),
                    textual_candidate_relations=candidate_relations,
                    textual_input_src_gen=input_src,
                    normalize_relations=args.normalize_relations,
                    rich_relation_clf_input_ids=rich_relation_clf_pair_encoded['input_ids'].to(device),
                    rich_relation_clf_attention_mask=rich_relation_clf_pair_encoded['attention_mask'].to(device),
                    textual_candidate_rich_relations=candidate_rich_relations,
                    do_concat=epoch >= args.warmup_epochs
                )
            elif isinstance(model, T5_MultiTask_Relation_Entity_Concat):
                loss = model(
                    input_ids_gen=src_encoded['input_ids'].to(device),
                    input_ids_relation_clf=relation_clf_pair_encoded['input_ids'].to(device),
                    gen_labels=tgt_encoded['input_ids'].to(device),
                    relation_clf_labels=relation_clf_pair_labels.to(device),
                    entity_clf_labels=entity_clf_pair_labels.to(device),
                    relation_clf_attention_mask=relation_clf_pair_encoded['attention_mask'].to(device),
                    textual_candidate_relations=candidate_relations,
                    textual_input_src_gen=input_src,
                    normalize_relations=args.normalize_relations,
                    rich_textual_candidate_entities_list=rich_textual_candidate_entities_list,
                    do_concat=epoch >= args.warmup_epochs
                )
            elif isinstance(model, T5_Multitask_Entity_Concat):
                loss = model(
                    input_ids_gen=src_encoded['input_ids'].to(device),
                    input_ids_clf=entity_clf_pair_encoded['input_ids'].to(device),
                    gen_labels=tgt_encoded['input_ids'].to(device),
                    clf_labels=entity_clf_pair_labels.to(device),
                    clf_attention_mask=entity_clf_pair_encoded['attention_mask'].to(device),
                    textual_candidate_entities=rich_textual_candidate_entities_list,
                    textual_input_src_gen=input_src,
                    do_concat=epoch >= args.warmup_epochs
                )
                
            loss = loss / iters_to_accumulate

            if (it+1)%iters_to_accumulate == 0:
                loss.backward()
                opti.step()
                lr_scheduler.step()
                opti.zero_grad()
            
            running_loss += loss.item()

            if (it + 1) % print_every == 0: # Print training loss inforamtion
                # tqdm.write("Iteration {}/{} of epoch {} complete. Loss : {} "
                #     .format(it+1, num_iterations, epoch+1, running_loss / print_every)
                #     )
                print(flush=True)
                print("Iteration {}/{} of epoch {} (Total:{}) complete. Loss : {} "
                    .format(it+1, num_iterations, epoch+1, epochs, running_loss / print_every)
                    ,flush=True)

            running_loss = 0.0

        if args.do_eval:
            # after training on one epoch, check dev_loss
            dev_loss = evaluate_loss(args, model,device,dev_dataloader)
            print()
            print("Epoch {} complete! Validation Loss : {}".format(epoch+1, dev_loss))
                    
            if dev_loss < best_loss:
                print('Best validation loss improved from {} to {}'.format(best_loss, dev_loss))
                print()
                model_copy = copy.deepcopy(model) # save a copy of the model
                best_loss = dev_loss
                best_epoch = epoch+1
            # save the best model
            model_to_save = model_copy
        else:
            print()
            print("Epoch {} complete!".format(epoch+1))
            model_to_save = model
        
        # save intermediate models after every n epochs
        if (epoch+1)%args.save_every_epochs==0:
            save_model(model_save_dir,model_to_save,(epoch+1),is_final_epoch=False)

        
    # empty cache
    torch.cuda.empty_cache()
    # save final model
    save_model(model_save_dir,model_to_save,epochs,is_final_epoch=True)
    print('Best epoch is: {}'.format(best_epoch))
    
    return model_to_save
    
    
def evaluate_loss(args, model,device,dataloader):
    model.eval()
    mean_loss = 0
    count = 0
    with torch.no_grad():
        for it, data in enumerate(tqdm(dataloader,desc='Evaluating')):
            src_encoded = data[0]
            tgt_encoded = data[1]
            relation_clf_pair_encoded = data[2]
            relation_clf_pair_labels = data[3]
            input_src = data[4]
            candidate_relations = data[5]
            entity_clf_pair_encoded = data[6]
            entity_clf_pair_labels = data[7]
            rich_textual_candidate_entities_list = data[8]
            # structure_tgt_encoded = data[9]
            # candidate_structures = data[10]
            # structure_clf_pair_labels = data[11]
            rich_relation_clf_pair_encoded = data[9]
            candidate_rich_relations = data[10]
            # structure_clf_pair_encoded = data[14]
            src_concatenated_encoded = data[11]
            src_golden_concatenated_encoded = data[12]
            # print(data)

            if isinstance(model, T5_generation):
                loss = model(
                    input_ids_gen=src_encoded['input_ids'].to(device),
                    gen_labels=tgt_encoded['input_ids'].to(device),
                    gen_attention_mask=src_encoded['attention_mask'].to(device),
                )
            elif isinstance(model, T5_generation_concat):
                if args.concat_golden:
                    loss = model(
                        input_ids_gen=src_golden_concatenated_encoded['input_ids'].to(device),
                        gen_labels=tgt_encoded['input_ids'].to(device),
                        gen_attention_mask=src_golden_concatenated_encoded['attention_mask'].to(device),
                    )
                else:
                    loss = model(
                        input_ids_gen=src_concatenated_encoded['input_ids'].to(device),
                        gen_labels=tgt_encoded['input_ids'].to(device),
                        gen_attention_mask=src_concatenated_encoded['attention_mask'].to(device),
                    )
            elif isinstance(model, T5_Multitask_Relation_Concat):
                loss = model(
                    input_ids_gen=src_encoded['input_ids'].to(device),
                    input_ids_clf=relation_clf_pair_encoded['input_ids'].to(device),
                    gen_labels=tgt_encoded['input_ids'].to(device),
                    clf_labels=relation_clf_pair_labels.to(device),
                    clf_attention_mask=relation_clf_pair_encoded['attention_mask'].to(device),
                    textual_candidate_relations=candidate_relations,
                    textual_input_src_gen=input_src,
                    normalize_relations=args.normalize_relations,
                    rich_relation_clf_input_ids=rich_relation_clf_pair_encoded['input_ids'].to(device),
                    rich_relation_clf_attention_mask=rich_relation_clf_pair_encoded['attention_mask'].to(device),
                    textual_candidate_rich_relations=candidate_rich_relations
                )
            elif isinstance(model, T5_MultiTask_Relation_Entity_Concat):
                loss = model(
                    input_ids_gen=src_encoded['input_ids'].to(device),
                    input_ids_relation_clf=relation_clf_pair_encoded['input_ids'].to(device),
                    gen_labels=tgt_encoded['input_ids'].to(device),
                    relation_clf_labels=relation_clf_pair_labels.to(device),
                    entity_clf_labels=entity_clf_pair_labels.to(device),
                    relation_clf_attention_mask=relation_clf_pair_encoded['attention_mask'].to(device),
                    textual_candidate_relations=candidate_relations,
                    textual_input_src_gen=input_src,
                    normalize_relations=args.normalize_relations,
                    rich_textual_candidate_entities_list=rich_textual_candidate_entities_list
                )
            elif isinstance(model, T5_Multitask_Entity_Concat):
                loss = model(
                    input_ids_gen=src_encoded['input_ids'].to(device),
                    input_ids_clf=entity_clf_pair_encoded['input_ids'].to(device),
                    gen_labels=tgt_encoded['input_ids'].to(device),
                    clf_labels=entity_clf_pair_labels.to(device),
                    clf_attention_mask=entity_clf_pair_encoded['attention_mask'].to(device),
                    textual_candidate_entities=rich_textual_candidate_entities_list,
                    textual_input_src_gen=input_src
                )
            
            mean_loss += loss.item()
            count+=1
    
    # torch.cuda.empty_cache()
    
    return mean_loss/count        


def run_prediction(args,model,device,dataloader,tokenizer,output_dir,output_predictions=True):
    print()
    print(f'Start predicting {args.predict_split}, beam_size:{args.eval_beams}, batch_size:{args.test_batch_size}')
    
    model.eval()
    all_gen_predictions = []
    all_gen_labels = []
    all_relation_clf_predictions = []
    all_relation_clf_labels = []
    all_entity_clf_predictions = []
    all_entity_clf_labels = []
    # all_structure_gen_predictions = []
    # all_structure_gen_labels = []
    # all_structure_clf_predictions = []
    # all_structure_clf_labels = []
    for it,data in enumerate(tqdm(dataloader,desc='Predicting')):
            src_encoded = data[0]
            tgt_encoded = data[1]
            relation_clf_pair_encoded = data[2]
            relation_clf_pair_labels = data[3]
            input_src = data[4]
            candidate_relations = data[5]
            entity_clf_pair_encoded = data[6]
            entity_clf_pair_labels = data[7]
            rich_textual_candidate_entities_list = data[8]
            # structure_tgt_encoded = data[9]
            # candidate_structures = data[10]
            # structure_clf_pair_labels = data[11]
            rich_relation_clf_pair_encoded = data[9]
            candidate_rich_relations = data[10]
            # structure_clf_pair_encoded = data[14]
            src_concatenated_encoded = data[11]
            src_golden_concatenated_encoded = data[12]

            entity_clf_outputs = None
            relation_clf_outputs = None
            # structure_gen_outputs = None
            # structure_clf_outputs = None

            if isinstance(model, T5_generation):
                gen_outputs = model.inference(
                    input_ids_gen=src_encoded['input_ids'].to(device),
                    gen_attention_mask=src_encoded['attention_mask'].to(device),
                    num_beams=args.eval_beams,
                )
            elif isinstance(model, T5_generation_concat):
                if args.concat_golden:
                    gen_outputs = model.inference(
                        input_ids_gen=src_golden_concatenated_encoded['input_ids'].to(device),
                        gen_attention_mask=src_golden_concatenated_encoded['attention_mask'].to(device),
                        num_beams=args.eval_beams,
                    )
                else:
                    gen_outputs = model.inference(
                        input_ids_gen=src_concatenated_encoded['input_ids'].to(device),
                        gen_attention_mask=src_concatenated_encoded['attention_mask'].to(device),
                        num_beams=args.eval_beams,
                    )
            elif isinstance(model, T5_Multitask_Relation_Concat):
                gen_outputs, relation_clf_outputs = model.inference(
                    input_ids_gen=src_encoded['input_ids'].to(device),
                    input_ids_clf=relation_clf_pair_encoded['input_ids'].to(device),
                    clf_attention_mask=relation_clf_pair_encoded['attention_mask'].to(device),
                    num_beams=args.eval_beams,
                    clf_sample_size=args.sample_size,
                    textual_candidate_relations=candidate_relations,
                    textual_input_src_gen=input_src,
                    normalize_relations=args.normalize_relations,
                    rich_relation_clf_input_ids=rich_relation_clf_pair_encoded['input_ids'].to(device),
                    rich_relation_clf_attention_mask=rich_relation_clf_pair_encoded['attention_mask'].to(device),
                    textual_candidate_rich_relations=candidate_rich_relations
                )
            elif isinstance(model, T5_MultiTask_Relation_Entity_Concat):
                gen_outputs, relation_clf_outputs, entity_clf_outputs = model.inference(
                    input_ids_gen=src_encoded['input_ids'].to(device),
                    input_ids_relation_clf=relation_clf_pair_encoded['input_ids'].to(device),
                    relation_clf_attention_mask=relation_clf_pair_encoded['attention_mask'].to(device),
                    num_beams=args.eval_beams,
                    textual_candidate_relations=candidate_relations,
                    textual_input_src_gen=input_src,
                    normalize_relations=args.normalize_relations,
                    rich_textual_candidate_entities_list=rich_textual_candidate_entities_list
                )
            elif isinstance(model, T5_Multitask_Entity_Concat):
                gen_outputs, entity_clf_outputs = model.inference(
                    input_ids_gen=src_encoded['input_ids'].to(device),
                    input_ids_clf=entity_clf_pair_encoded['input_ids'].to(device),
                    clf_attention_mask=entity_clf_pair_encoded['attention_mask'].to(device),
                    num_beams=args.eval_beams,
                    textual_candidate_entities=rich_textual_candidate_entities_list,
                    textual_input_src_gen=input_src
                )

            gen_outputs = [p.cpu().numpy() for p in gen_outputs]
            gen_labels = tgt_encoded['input_ids'].numpy()
            all_gen_predictions.extend(gen_outputs)
            all_gen_labels.extend(gen_labels)
            
            if relation_clf_outputs is not None:
                relation_clf_outputs = torch.sigmoid(relation_clf_outputs).detach().cpu().reshape(-1,args.sample_size)
                relation_clf_pair_labels = relation_clf_pair_labels.cpu().reshape(-1,args.sample_size)
                all_relation_clf_predictions.extend([p.numpy() for p in relation_clf_outputs])
                all_relation_clf_labels.extend([l.numpy() for l in relation_clf_pair_labels])

            if entity_clf_outputs is not None:
                entity_clf_outputs = torch.sigmoid(entity_clf_outputs).detach().cpu().reshape(-1,args.sample_size)
                entity_clf_pair_labels = entity_clf_pair_labels.cpu().reshape(-1,args.sample_size)
                all_entity_clf_predictions.extend([p.numpy() for p in entity_clf_outputs])
                all_entity_clf_labels.extend([l.numpy() for l in entity_clf_pair_labels])
            
            # if structure_gen_outputs is not None:
            #     structure_gen_outputs = [p.cpu().numpy() for p in structure_gen_outputs]
            #     structure_gen_labels = structure_tgt_encoded['input_ids'].numpy()
            #     all_structure_gen_predictions.extend(structure_gen_outputs)
            #     all_structure_gen_labels.extend(structure_gen_labels)
            
            # if structure_clf_outputs is not None:
            #     structure_clf_outputs = torch.sigmoid(structure_clf_outputs).detach().cpu().reshape(-1,args.structure_sample_size)
            #     structure_clf_pair_labels = structure_clf_pair_labels.cpu().reshape(-1,args.structure_sample_size)
            #     all_structure_clf_predictions.extend([p.numpy() for p in structure_clf_outputs])
            #     all_structure_clf_labels.extend([l.numpy() for l in structure_clf_pair_labels])

    ex_cnt = 0
    contains_ex_cnt = 0
    output_list = []
    real_total = 0
    for i,pred in enumerate(all_gen_predictions):
        predictions = tokenizer.batch_decode(pred, skip_special_tokens=True)
        gen_label = tokenizer.decode(all_gen_labels[i], skip_special_tokens=True)
        # if len(all_structure_gen_predictions) > 0 and len(all_entity_clf_predictions) > 0 and len(all_relation_clf_predictions) > 0:
        #     structure_predictions = tokenizer.batch_decode(all_structure_gen_predictions[i],skip_special_tokens=True)
        #     structure_label = tokenizer.decode(all_structure_gen_labels[i], skip_special_tokens=True)
        #     output_list.append({
        #         'predictions':predictions,
        #         'gen_label':gen_label,
        #         'structure_predictions': structure_predictions,
        #         'structure_label': structure_label,
        #         'pred_relation_clf_labels':[float(p) for p in list(all_relation_clf_predictions[i])],
        #         'gold_relation_clf_labels':[float(l) for l in list(all_relation_clf_labels[i])],
        #         'pred_entity_clf_labels':[float(p) for p in list(all_entity_clf_predictions[i])],
        #         'gold_entity_clf_labels':[float(p) for p in list(all_entity_clf_labels[i])],
        #     })
        # elif len(all_structure_clf_predictions) > 0 and len(all_entity_clf_predictions) > 0 and len(all_relation_clf_predictions) > 0:
        #     output_list.append({
        #         'predictions':predictions,
        #         'gen_label':gen_label,
        #         'pred_relation_clf_labels':[float(p) for p in list(all_relation_clf_predictions[i])],
        #         'gold_relation_clf_labels':[float(l) for l in list(all_relation_clf_labels[i])],
        #         'pred_entity_clf_labels':[float(p) for p in list(all_entity_clf_predictions[i])],
        #         'gold_entity_clf_labels':[float(p) for p in list(all_entity_clf_labels[i])],
        #         'pred_structure_clf_labels': [float(p) for p in list(all_structure_clf_predictions[i])],
        #         'gold_structure_clf_labels': [float(p) for p in list(all_structure_clf_labels[i])],
        #     })
        if len(all_entity_clf_predictions) > 0 and len(all_relation_clf_predictions) > 0:
            output_list.append({
                'predictions':predictions,
                'gen_label':gen_label,
                'pred_relation_clf_labels':[float(p) for p in list(all_relation_clf_predictions[i])],
                'gold_relation_clf_labels':[float(l) for l in list(all_relation_clf_labels[i])],
                'pred_entity_clf_labels':[float(p) for p in list(all_entity_clf_predictions[i])],
                'gold_entity_clf_labels':[float(p) for p in list(all_entity_clf_labels[i])],
            })
        elif len(all_relation_clf_predictions) > 0:
            output_list.append({
                'predictions':predictions,
                'gen_label':gen_label,
                'pred_relation_clf_labels':[float(p) for p in list(all_relation_clf_predictions[i])],
                'gold_relation_clf_labels':[float(l) for l in list(all_relation_clf_labels[i])],  
            })
        elif len(all_entity_clf_predictions) > 0:
            output_list.append({
                'predictions':predictions,
                'gen_label':gen_label,
                'pred_entity_clf_labels':[float(p) for p in list(all_entity_clf_predictions[i])],
                'gold_entity_clf_labels':[float(p) for p in list(all_entity_clf_labels[i])],
            })
        else:
            output_list.append({
                'predictions':predictions,
                'gen_label':gen_label,
            })

        if predictions[0].lower()==gen_label.lower():
            ex_cnt+=1

        if any([x.lower()==gen_label.lower() for x in predictions]):
            contains_ex_cnt+=1
        
        if gen_label.lower()!='null':
            real_total+=1

    
    print(f"""total:{len(output_list)}, 
                    ex_cnt:{ex_cnt}, 
                    ex_rate:{ex_cnt/len(output_list)}, 
                    real_ex_rate:{ex_cnt/real_total}, 
                    contains_ex_cnt:{contains_ex_cnt}, 
                    contains_ex_rate:{contains_ex_cnt/len(output_list)}
                    real_contains_ex_rate:{contains_ex_cnt/real_total}
                    """)

        
    if output_predictions:
        file_path = os.path.join(output_dir,f'beam_{args.eval_beams}_{args.predict_split}_{args.test_batch_size}_top_k_predictions.json')
        
        gen_statistics_file_path = os.path.join(output_dir,f'beam_{args.eval_beams}_{args.predict_split}_{args.test_batch_size}_gen_statistics.json')
        gen_statistics = {
            'total':len(output_list),
            'exmatch_num': ex_cnt,
            'exmatch_rate': ex_cnt/len(output_list),
            'real_exmatch_rate':ex_cnt/real_total, 
            'contains_ex_num':contains_ex_cnt,
            'contains_ex_rate':contains_ex_cnt/len(output_list),
            'real_contains_ex_rate':contains_ex_cnt/real_total
        }

        dump_json(output_list, file_path, indent=4)
        dump_json(gen_statistics, gen_statistics_file_path,indent=4)
        if args.dataset_type == 'CWQ':
            dataset = load_json(f'data/CWQ/generation/merged_0724_ep1/CWQ_{args.predict_split}.json')
            # dataset = load_json(f'data/CWQ/generation/merged_0715_retrain/CWQ_{args.predict_split}.json')
            # dataset = load_json(f'data/CWQ/generation/merged_0714/CWQ_{args.predict_split}.json')
            # dataset = load_json(f'data/CWQ/generation/merged/CWQ_{args.predict_split}.json')
            # dataset = load_json(f'data/CWQ/generation/merged_old/CWQ_{args.predict_split}.json')
            # dataset = load_json(f'data/CWQ/generation/xwu_merged_new/CWQ_{args.predict_split}.json')
        elif args.dataset_type == 'WebQSP':
            dataset = load_json(f'data/WebQSP/generation/merged_relation_final/WebQSP_{args.predict_split}.json')
            # dataset = load_json(f'data/WebQSP/generation/0722/merged_question_relation_ep3_2hop/WebQSP_{args.predict_split}.json')
            # dataset = load_json(f'data/WebQSP/generation/merged_yhshu/WebQSP_{args.predict_split}.json')
            # dataset = load_json(f'data/WebQSP/generation/merged_0715_retrain_biencoder_ep5/WebQSP_{args.predict_split}.json')
            # dataset = load_json(f'data/WebQSP/generation/merged_0715_retrain/WebQSP_{args.predict_split}.json')
            # if args.predict_split == 'train':
            #     dataset = load_json('data/WebQSP/generation/merged_old/WebQSP_{}.json'.format(args.predict_split))
            # elif args.predict_split == 'test':
            #     dataset = load_json('data/WebQSP/generation/merged_0714/WebQSP_test.json')
            # dataset = load_json(f'data/WebQSP/generation/merged_corrected/WebQSP_{args.predict_split}.json')
            # dataset = load_json(f'data/WebQSP/generation/merged_old/WebQSP_{args.predict_split}.json')
            #  dataset = load_json(f'data/WebQSP/generation/xwu_merged_new/WebQSP_{args.predict_split}.json')
        if len(all_entity_clf_predictions) > 0:
            generate_candidate_entity_map_classification_res(output_list, output_dir, dataset, args)


def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__=='__main__':
    
    args = _parse_args()
    print(args)

    # do_debug = False
    do_debug = args.do_debug
    
    if do_debug=='True':
        import ptvsd
        server_ip = "0.0.0.0"
        server_port = 12345
        print('Waiting for debugger attach...',flush=True)
        ptvsd.enable_attach(address=(server_ip,server_port))
        ptvsd.wait_for_attach()

    # set seed, for reproduce
    set_seed(42) # default seed
    # set parameters
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    test_batch_size = args.test_batch_size
    epochs = args.epochs
    lr = args.lr
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    tokenizer.add_special_tokens(
            {"additional_special_tokens":["[DES]","[INQ]", "[des]","[inq]"]}
    )
    tokenizer.add_tokens(["[ENT]", "[REL]", "[LIT]", "[ent]", "[rel]", "[lit]"])
    
    if args.do_train:
        # load data
        train_dataloader = prepare_dataloader(args,'train',tokenizer, batch_size=train_batch_size)
        if args.do_eval:
            dev_dataloader = prepare_dataloader(args,'dev',tokenizer, batch_size=eval_batch_size)
        else: 
            dev_dataloader = None

        
        # load model
        if args.model == 'T5_generation':
            print('T5_generation')
            model = T5_generation(
                args.pretrained_model_path,
                is_test=False,
                max_tgt_len=args.max_tgt_len
            )
        elif args.model == 'T5_generation_concat':
            print('T5_generation_concat')
            model = T5_generation_concat(
                args.pretrained_model_path,
                is_test=False,
                max_tgt_len=args.max_tgt_len
            )
        elif args.model == 'T5_Multitask_Relation_Concat':
            print('T5_Multitask_Relation_Concat')
            model = T5_Multitask_Relation_Concat(
                args.pretrained_model_path,
                device=device, 
                max_src_len=args.max_src_len,
                max_tgt_len=args.max_tgt_len,
                tokenizer=tokenizer,
                is_test=False,
                sample_size=args.sample_size,
                do_lower=args.do_lower,
                cross_entropy_loss=args.cross_entropy_loss,
                add_prefix=args.add_prefix,
                use_rich_relation=args.use_rich_relation
            )
        elif args.model == 'T5_MultiTask_Relation_Entity_Concat':
            print('T5_MultiTask_Relation_Entity_Concat')
            model = T5_MultiTask_Relation_Entity_Concat(
                args.pretrained_model_path,
                device=device, 
                max_src_len=args.max_src_len,
                max_tgt_len=args.max_tgt_len,
                tokenizer=tokenizer,
                is_test=False,
                sample_size=args.sample_size,
                do_lower=args.do_lower,
                cross_entropy_loss=args.cross_entropy_loss,
                add_prefix=args.add_prefix,
                train_concat_true=args.train_concat_true
            )
        elif args.model == 'T5_Multitask_Entity_Concat':
            print('T5_Multitask_Entity_Concat')
            model = T5_Multitask_Entity_Concat(
                args.pretrained_model_path,
                device=device,
                max_src_len=args.max_src_len,
                max_tgt_len=args.max_tgt_len,
                tokenizer=tokenizer,
                is_test=False,
                sample_size=args.sample_size,
                do_lower=args.do_lower,
                add_prefix=args.add_prefix,
            )

        model.t5.resize_token_embeddings(len(tokenizer))
        model = model.to(device)

        # define model path to
        output_dir = args.output_dir
        model_save_dir = args.model_save_dir
        # path_to_model = output_dir+'t5_mtl_lr_{}_ep_{}_batch_{}.pt'.format(lr,epochs,batch_size)

        # train model
        model = train_model(args,model,tokenizer,device,train_dataloader,dev_dataloader,model_save_dir=model_save_dir)

        
    if args.do_predict:
        # test load model
        if args.do_train:
            print()
            print('Use trained model to do prediction')
            model = model.to(device)
        else:
            print()
            print("Loading the weights of the model...")
            # load model
            if args.model == 'T5_generation':
                print('T5_generation')
                model = T5_generation(
                    args.pretrained_model_path,
                    is_test=False,
                    max_tgt_len=args.max_tgt_len
                )
            elif args.model == 'T5_generation_concat':
                print('T5_generation_concat')
                model = T5_generation_concat(
                    args.pretrained_model_path,
                    is_test=False,
                    max_tgt_len=args.max_tgt_len
                )
            elif args.model == 'T5_Multitask_Relation_Concat':
                print('T5_Multitask_Relation_Concat')
                model = T5_Multitask_Relation_Concat(
                    args.pretrained_model_path,
                    device=device, 
                    max_src_len=args.max_src_len,
                    max_tgt_len=args.max_tgt_len,
                    tokenizer=tokenizer,
                    is_test=False,
                    sample_size=args.sample_size,
                    do_lower=args.do_lower,
                    cross_entropy_loss=args.cross_entropy_loss,
                    add_prefix=args.add_prefix,
                    use_rich_relation=args.use_rich_relation
                )
            elif args.model == 'T5_MultiTask_Relation_Entity_Concat':
                print('T5_MultiTask_Relation_Entity_Concat')
                model = T5_MultiTask_Relation_Entity_Concat(
                    args.pretrained_model_path,
                    device=device, 
                    max_src_len=args.max_src_len,
                    max_tgt_len=args.max_tgt_len,
                    tokenizer=tokenizer,
                    is_test=False,
                    sample_size=args.sample_size,
                    do_lower=args.do_lower,
                    cross_entropy_loss=args.cross_entropy_loss,
                    add_prefix=args.add_prefix,
                    train_concat_true=args.train_concat_true
                )
            elif args.model == 'T5_Multitask_Entity_Concat':
                print('T5_Multitask_Entity_Concat')
                model = T5_Multitask_Entity_Concat(
                    args.pretrained_model_path,
                    device=device,
                    max_src_len=args.max_src_len,
                    max_tgt_len=args.max_tgt_len,
                    tokenizer=tokenizer,
                    is_test=False,
                    sample_size=args.sample_size,
                    do_lower=args.do_lower,
                    add_prefix=args.add_prefix,
                )
            model.t5.resize_token_embeddings(len(tokenizer))
            state_dict = torch.load(os.path.join(args.model_save_dir,'pytorch_model.bin'))
            model.load_state_dict(state_dict)
            model.to(device)
            print('Model loaded')

        test_dataloader = prepare_dataloader(args, args.predict_split,tokenizer=tokenizer,batch_size=test_batch_size)
        # print('Predicting Num:', len(test_dataloader)*test_batch_size)
        run_prediction(args,model,device,test_dataloader,tokenizer,output_dir=args.output_dir,output_predictions=True)

        print('Prediction Finished')

