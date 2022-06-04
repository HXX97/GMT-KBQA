from email.policy import default
import torch
import random
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import copy
import argparse

from biencoder import BiEncoderModule
BLANK_TOKEN = '[BLANK]'


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--add_special_tokens', default=False, action='store_true',help='True when mask entity mention')
    parser.add_argument('--dataset_type', default="CWQ", type=str, help="CWQ | WEBQSP")
    parser.add_argument('--model_save_path', default='data/', type=str)
    parser.add_argument('--max_len', default=32, type=int, help="32 for CWQ, 80 for WebQSP with richRelation, 28 for LC")
    parser.add_argument('--batch_size', default=4, type=int, help="4 for CWQ")
    parser.add_argument('--epochs', default=1, type=int, help="1 for CWQ, 3 for WebQSP")
    parser.add_argument('--log_dir', default='log/', type=str)
    args = parser.parse_args()
    return args


def data_process(dataset_type):
    if dataset_type == "CWQ":
        train_df = pd.read_csv('../../Data/CWQ/relation_retrieval/bi-encoder/CWQ.train.maskEntity.sampled.tsv', sep='\t', error_bad_lines=False).dropna()
        dev_df = pd.read_csv('../../Data/CWQ/relation_retrieval/bi-encoder/CWQ.dev.maskEntity.sampled.tsv', sep='\t', error_bad_lines=False).dropna()
        test_df = pd.read_csv('../../Data/CWQ/relation_retrieval/bi-encoder/CWQ.test.maskEntity.sampled.tsv', sep='\t', error_bad_lines=False).dropna()
    else:
        train_df = pd.read_csv('../../Data/WEBQSP/relation_retrieval/bi-encoder/train.sampled.richRelation.1parse.tsv', sep='\t', error_bad_lines=False).dropna()
        dev_df = None
        test_df = pd.read_csv('../../Data/WEBQSP/relation_retrieval/bi-encoder/test.sampled.richRelation.1parse.tsv', sep='\t', error_bad_lines=False).dropna()
    
    return train_df, dev_df, test_df

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def evaluate(model, device, dataloader):
    model.eval()
    
    mean_loss = 0
    count = 0
    golden_truth = []
    preds = []
    
    with torch.no_grad():
        for question_token_ids, question_attn_masks, question_token_type_ids, relations_token_ids, relations_attn_masks, relations_token_type_ids, golden_id in tqdm(dataloader):
            scores, loss = model(
                question_token_ids.to(device),
                question_attn_masks.to(device),
                question_token_type_ids.to(device),
                relations_token_ids.to(device),
                relations_attn_masks.to(device),
                relations_token_type_ids.to(device),
                golden_id.to(device)
            )
            mean_loss += loss
            count += 1
            pred_id = torch.argmax(scores, dim=1) 
            # print('pred_id: {}'.format(pred_id.shape))
            # print('golden_id: {}'.format(golden_id.shape))
            preds += pred_id.tolist()
            golden_truth += golden_id.tolist()
    
    accuracy = accuracy_score(golden_truth, preds)
    
    return mean_loss / count, accuracy
    

class CustomDataset(Dataset):
    def __init__(self, data, maxlen, tokenizer=None, bert_model='/home3/xwu/bertModels/bert-base-uncased',  sample_size=100):
        self.data = data
        self.sample_size = sample_size
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(bert_model)
        self.maxlen = maxlen
    
    def __len__(self):
        return int(len(self.data) / self.sample_size)
    
    def __getitem__(self, index):
        start = self.sample_size * index
        end = min(self.sample_size*(index+1), len(self.data))
        question = str(self.data.loc[start, 'question'])
        relations = [str(self.data.loc[i, 'relation']) for i in range(start, end)]
        golden_id = [i-start for i in range(start, end) if self.data.loc[i, 'label'] == 1]
        assert len(golden_id) == 1, print(start, end)
        
        encoded_question = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=self.maxlen,
            return_tensors='pt'
        )
        encoded_relations = [self.tokenizer(
            relation,
            padding='max_length',
            truncation=True,
            max_length=self.maxlen,
            return_tensors='pt'
        ) for relation in relations]
        
        question_token_ids = encoded_question['input_ids'].squeeze(0)  # tensor of token ids
        question_attn_masks = encoded_question['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        question_token_type_ids = encoded_question['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens
        
        relations_token_ids = torch.cat([encoded_relation['input_ids'] for encoded_relation in encoded_relations], 0)
        relations_attn_masks = torch.cat([encoded_relation['attention_mask'] for encoded_relation in encoded_relations], 0)
        relations_token_type_ids = torch.cat([encoded_relation['token_type_ids'] for encoded_relation in encoded_relations], 0)
        
        return question_token_ids, question_attn_masks, question_token_type_ids, relations_token_ids, relations_attn_masks, relations_token_type_ids, golden_id[0]


def train_bert(model, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate, device, log_path, model_save_path):
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 5
    if log_path:
        log_w = open(log_path, 'w')
    scaler = GradScaler()
    
    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        
        for it, (question_token_ids, question_attn_masks, question_token_type_ids, relations_token_ids, relations_attn_masks, relations_token_type_ids, golden_id) in enumerate(tqdm(train_loader)):
            scores, loss = model(
                question_token_ids.to(device),
                question_attn_masks.to(device),
                question_token_type_ids.to(device),
                relations_token_ids.to(device),
                relations_attn_masks.to(device),
                relations_token_type_ids.to(device),
                golden_id.to(device)
            )
            loss = loss / iters_to_accumulate
            scaler.scale(loss).backward()
        
            if (it + 1) % iters_to_accumulate == 0:
                scaler.step(opti)
                # Updates the scale for next iteration.
                scaler.update()
                # Adjust the learning rate based on the number of iterations.
                lr_scheduler.step()
                # Clear gradients
                opti.zero_grad()
        
            running_loss += loss.item()
            if (it + 1) % print_every == 0:  # Print training loss information
                print()
                print("Iteration {}/{} of epoch {} complete. Loss : {} "
                        .format(it+1, nb_iterations, ep+1, running_loss / print_every))

                running_loss = 0.0
        
        if val_loader:
            val_loss, accuracy = evaluate(model, device, val_loader)
            print("Epoch {} complete! Validation Loss : {}".format(ep+1, val_loss))
            print("Accuracy on dev data: {}\n".format(accuracy))
            if log_w:
                log_w.write("Epoch {} complete! Validation Loss : {}\n".format(ep+1, val_loss))
                log_w.write("Accuracy on dev data: {}\n".format(accuracy))
        
        model_copy = copy.deepcopy(model)
        # if val_loss < best_loss:
        #     print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
        #     print()
        #     best_loss = val_loss
        
        model_path = os.path.join(model_save_path, '{}_lr_{}_ep_{}.pt'.format("bert-base-uncased", lr, ep+1))
        torch.save(model_copy.state_dict(), model_path)
        print("The model has been saved in {}".format(model_path))

    if log_w:
        log_w.close()
    del loss
    torch.cuda.empty_cache()
 

def main(args):
    bert_model = '/home3/xwu/bertModels/bert-base-uncased'
    freeze_bert = False
    maxlen = args.max_len
    bs = args.batch_size
    iters_to_accumulate = 2  # the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size
    lr = 2e-5  # learning rate
    epochs = args.epochs
    log_path = os.path.join(args.log_dir, 'log.txt') 
    
    if args.add_special_tokens:
        print('add special tokens')
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        special_tokens_dict = {'additional_special_tokens': [BLANK_TOKEN]}
        tokenizer.add_special_tokens(special_tokens_dict)
    else: 
        tokenizer = AutoTokenizer.from_pretrained(bert_model)

    set_seed(1)
    print("Reading training data...")
    # train_df, dev_df, test_df = data_process()
    train_df, dev_df, _ = data_process(args.dataset_type)
    print(train_df.shape)
    train_set = CustomDataset(train_df, maxlen, tokenizer=tokenizer, bert_model=bert_model)
    train_loader = DataLoader(train_set, batch_size=bs, num_workers=2)
    if dev_df is not None:
        print("Reading validation data...")
        print(dev_df.shape)
        val_set = CustomDataset(dev_df, maxlen, tokenizer=tokenizer, bert_model=bert_model)
        val_loader = DataLoader(val_set, batch_size=bs, num_workers=2)
    else:
        val_loader = None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiEncoderModule(device, bert_model=bert_model, tokenizer=tokenizer, freeze_bert=freeze_bert)
    model.to(device)
    
    opti = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    num_warmup_steps = 0 # The number of steps for the warmup phase.
    num_training_steps = epochs * len(train_loader)  # The total number of training steps
    t_total = (len(train_loader) // iters_to_accumulate) * epochs  # Necessary to take into account Gradient accumulation
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
    
    train_bert(model, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate, device, log_path, args.model_save_path)
         

if __name__=='__main__':
    args = _parse_args()
    print(args)
    main(args)