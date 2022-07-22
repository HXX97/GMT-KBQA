import torch
import torch.nn as nn
import os
import json
from collections import defaultdict
import argparse
# import matplotlib.pyplot as plt
import copy
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, precision_score, recall_score

BLANK_TOKEN = '[BLANK]'

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train',default=False,action='store_true',help='whether to do training')
    parser.add_argument('--do_eval',default=False,action='store_true',help='whether to do eval when training')
    parser.add_argument('--do_predict',default=False,action='store_true',help='whether to do prediction')
    parser.add_argument('--predict_split',default='test',help='which dataset to perform prediction')
    parser.add_argument('--max_len', default=128, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--iters_to_accumulate', default=2, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--epochs', default=6, type=int)
    parser.add_argument('--log_dir', default='logs/', type=str)
    parser.add_argument('--dataset_type', default="CWQ", type=str, help="CWQ | WebQSP")
    parser.add_argument('--mask_entity_mention', default=False, action='store_true', help="mask entity mentions in a question")
    parser.add_argument('--model_save_path', type=str, help="load model checkpoint from this path")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--loss_type', default="CE", type=str, help="loss function. 'BCE'|'CE' ")
    parser.add_argument('--cache_dir', default='bert-base-uncased')
    args = parser.parse_args()
    return args


def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def data_process(args):
    if args.do_train or args.do_eval:
        if args.dataset_type == 'WebQSP':
            train_df = pd.read_csv('data/WebQSP/relation_retrieval_final/cross-encoder/rich_relation_3epochs_question_relation/WebQSP.train.tsv', delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
            # train_df = pd.read_csv('data/WebQSP/relation_retrieval/cross-encoder/0715_retrain/WebQSP.train.tsv', sep='\t', error_bad_lines=False).dropna()
            # train_df = pd.read_csv('data/WebQSP/relation_retrieval_0717/cross-encoder/rich_relation_3epochs_question_relation_top100/WebQSP.ptrain.tsv', sep='\t', error_bad_lines=False).dropna()
            # train_df = pd.read_csv('data/WebQSP/relation_retrieval_0717/cross-encoder/rich_relation_3epochs_mask_mention_rich_relation/WebQSP.train.tsv', sep='\t', error_bad_lines=False).dropna()
            # train_df = pd.read_csv('data/WebQSP/relation_retrieval/cross-encoder/0715_retrain_biencoder_ep5/WebQSP.train.tsv', sep='\t', error_bad_lines=False).dropna()
            dev_df = None
            # dev_df = pd.read_csv('data/WebQSP/relation_retrieval_0717/cross-encoder/rich_relation_3epochs_question_relation_top100/WebQSP.pdev.tsv', sep='\t', error_bad_lines=False).dropna()
            # test_df = pd.read_csv('data/WebQSP/relation_retrieval_0717/cross-encoder/rich_relation_3epochs_question_relation_top100/WebQSP.test.tsv', sep='\t', error_bad_lines=False).dropna()
            # test_df = pd.read_csv('data/WebQSP/relation_retrieval_0717/cross-encoder/rich_relation_3epochs_mask_mention_rich_relation/WebQSP.test.tsv', sep='\t', error_bad_lines=False).dropna()
            test_df = None
            # test_df = pd.read_csv('data/WebQSP/relation_retrieval/cross-encoder/0715_retrain/WebQSP.test.tsv', sep='\t', error_bad_lines=False).dropna()
            # test_df = pd.read_csv('data/WebQSP/relation_retrieval/cross-encoder/WebQSP.test.biEncoder.train_all.richRelation.crossEncoder.train_all.richRelation.2hopValidation.richEntity.top100.1parse.tsv', sep='\t', error_bad_lines=False).dropna()
        else:
            train_df = pd.read_csv('data/CWQ/relation_retrieval/cross-encoder/0715_retrain/CWQ.train.tsv', sep='\t', error_bad_lines=False).dropna()
            dev_df = pd.read_csv('data/CWQ/relation_retrieval/cross-encoder/0715_retrain/CWQ.dev.tsv', sep='\t', error_bad_lines=False).dropna()
            # test_df = pd.read_csv('data/CWQ/relation_retrieval/cross-encoder/xwu_test/CWQ.test.tsv', sep='\t', error_bad_lines=False).dropna()
            test_df = pd.read_csv('data/CWQ/relation_retrieval/cross-encoder/0715_retrain/CWQ.test.tsv', sep='\t', error_bad_lines=False).dropna()
        return train_df, dev_df, test_df
    elif args.do_predict:
        print('do inference')
        if args.dataset_type.lower() == 'webqsp':
            train_2hop_df = pd.read_csv('data/WebQSP/relation_retrieval_final/cross-encoder/rich_relation_3epochs_question_relation/WebQSP.train_2hop.tsv', delimiter='\t', dtype={"id":int, "question":str, "relation":str, 'label':int})
            train_df = pd.read_csv('data/WebQSP/relation_retrieval_final/cross-encoder/rich_relation_3epochs_question_relation/WebQSP.train.tsv', delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
            test_2hop_df = pd.read_csv('data/WebQSP/relation_retrieval_final/cross-encoder/rich_relation_3epochs_question_relation/WebQSP.test_2hop.tsv', delimiter='\t', dtype={"id":int, "question":str, "relation":str, 'label':int})
            # TODO: 如何区分这两个 test
            test_df = pd.read_csv('data/WebQSP/relation_retrieval_final/cross-encoder/rich_relation_3epochs_question_relation/WebQSP.test.tsv', delimiter='\t', dtype={"id":int, "question":str, "relation":str, 'label':int})
            return train_2hop_df, train_df, test_2hop_df, test_df
        else:
            # TODO:
            train_df = pd.read_csv('data/CWQ/relation_retrieval/cross-encoder/0715_retrain/CWQ.train.tsv', sep='\t', error_bad_lines=False).dropna()
            dev_df = pd.read_csv('data/CWQ/relation_retrieval/cross-encoder/0715_retrain/CWQ.dev.tsv', sep='\t', error_bad_lines=False).dropna()
            # test_df = pd.read_csv('data/CWQ/relation_retrieval/cross-encoder/xwu_test/CWQ.test.tsv', sep='\t', error_bad_lines=False).dropna()
            test_df = pd.read_csv('data/CWQ/relation_retrieval/cross-encoder/0715_retrain/CWQ.test.tsv', sep='\t', error_bad_lines=False).dropna()
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
    

def evaluate(net, device, criterion, dataloader, loss_type):
    net.eval()

    mean_loss = 0
    count = 0
    golden_truth = []
    preds = []

    with torch.no_grad():
        for _, (seq, attn_masks, token_type_ids, labels, indexes) in enumerate(tqdm(dataloader)):
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            logits = net(seq, attn_masks, token_type_ids)
            # mean_loss += criterion(logits.squeeze(-1), labels.float()).item() # BCELoss
            mean_loss += criterion(logits, labels).item()
            count += 1
            if loss_type == "BCE":
                probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
                pred = np.where(probs > 0.5, 1, 0)
            elif loss_type == "CE":
                pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
            preds += pred.tolist()
            golden_truth += labels.tolist()
    accuracy = accuracy_score(golden_truth, preds)
    kappa = cohen_kappa_score(golden_truth, preds)
    f1 = f1_score(golden_truth, preds, pos_label=1, average='binary')

    return mean_loss / count, accuracy, kappa, f1


class CustomDataset(Dataset):
    def __init__(self, data, maxlen, tokenizer=None, with_labels=True, bert_model='/home3/xwu/bertModels/bert-base-uncased'):

        self.data = data  # pandas dataframe
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(bert_model)

        self.maxlen = maxlen
        self.with_labels = with_labels
    
    def __len__(self):
        return len(self.data)
    
    def get_original_item(self, index):
        id = self.data.loc[index, 'id']
        question = self.data.loc[index, 'question']
        relation = self.data.loc[index, 'relation']
        label = self.data.loc[index, 'label']
        return {
            'id': id,
            'question': question,
            'relation': relation,
            'label': label
        }

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent1 = str(self.data.loc[index, 'question'])
        sent2 = str(self.data.loc[index, 'relation'])

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent1, sent2, 
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = self.data.loc[index, 'label']
            return token_ids, attn_masks, token_type_ids, label, index  
        else:
            return token_ids, attn_masks, token_type_ids, index


class SentencePairClassifier(nn.Module):
    def __init__(self, bert_model="/home3/xwu/bertModels/bert-base-uncased", tokenizer=None, freeze_bert=False):
        super(SentencePairClassifier, self).__init__()
        #  Instantiating BERT-based model object
        self.bert_layer = AutoModel.from_pretrained(bert_model)
        if tokenizer:
            self.bert_layer.resize_token_embeddings(len(tokenizer))

        # TODO: support more models
        if "bert-base-uncased" in bert_model:
            hidden_size = 768
        else:
            hidden_size = 768

        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        # self.cls_layer = nn.Linear(hidden_size, 1) # BCELoss
        self.cls_layer = nn.Linear(hidden_size, 2) # CrossEntropyLoss

        self.dropout = nn.Dropout(p=0.1)
    
    @autocast()
    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''

        pooler_output = self.bert_layer(input_ids, attn_masks, token_type_ids).pooler_output
        logits = self.cls_layer(self.dropout(pooler_output))

        return logits


def train_bert(args, net, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate, device, log_path, output_dir):

    best_loss = np.Inf
    # best_f1 = -1.0
    best_epoch = 1
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 5  # print the training loss 5 times per epoch
    iters = []
    train_losses = []
    val_losses = []
    
    log_w = open(log_path, 'w')

    scaler = GradScaler()

    for ep in range(epochs):

        net.train()
        running_loss = 0.0
        for it, (seq, attn_masks, token_type_ids, labels, indexes) in enumerate(tqdm(train_loader)):

            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
    
            with autocast():
                logits = net(seq, attn_masks, token_type_ids)

                # loss = criterion(logits.squeeze(-1), labels.float())
                loss = criterion(logits, labels) # CrossEntropyLoss
                loss = loss / iters_to_accumulate  # Normalize the loss because it is averaged

            scaler.scale(loss).backward()

            if (it + 1) % iters_to_accumulate == 0:
                scaler.step(opti)
                scaler.update()
                lr_scheduler.step()
                # Clear gradients
                opti.zero_grad()

            running_loss += loss.item()

            if (it + 1) % print_every == 0:  # Print training loss information
                print()
                print("Iteration {}/{} of epoch {} complete. Loss : {} "
                      .format(it+1, nb_iterations, ep+1, running_loss / print_every))

                running_loss = 0.0

        if val_loader is not None:
            val_loss, accuracy, kappa, val_f1 = evaluate(net, device, criterion, val_loader, args.loss_type)  # Compute validation loss
            print()
            print("Epoch {} complete! Validation Loss : {}".format(ep+1, val_loss))
            log_w.write("Epoch {} complete! Validation Loss : {}\n".format(ep+1, val_loss))
            log_w.write("Accuracy on dev data: {}\n".format(accuracy))
            log_w.write("kappa on dev data: {}\n".format(kappa))
            log_w.write("f1 on dev data: {}\n".format(val_f1))
            print("Epoch {} complete! Validation Loss : {}\n".format(ep+1, val_loss))
            print("Accuracy on dev data: {}\n".format(accuracy))
            print("kappa on dev data: {}\n".format(kappa))
            print("f1 on dev data: {}\n".format(val_f1))

            # TODO: 可以给出每一轮的 loss, 但是保存的还是当前轮的
            if val_loss < best_loss:
                print('Best validation loss improved from {} to {}'.format(best_loss, val_loss))
                print()
                # net_copy = copy.deepcopy(net) # save a copy of the model
                best_loss = val_loss
                best_epoch = ep+1
            model_to_save = copy.deepcopy(net) # 每一轮都保存
        else:
            print("Epoch {} complete!".format(ep+1))
            model_to_save = copy.deepcopy(net)  # save a copy of the model
        
        # Save model every epoch
        if (ep+1)%1==0:
            model_save_path = os.path.join(output_dir, '{}_ep_{}.pt'.format(args.dataset_type, ep+1))
            torch.save(model_to_save.state_dict(), model_save_path)
            print("The model of epoch {} has been saved in {}; best epoch is: {}".format(ep+1, model_save_path, best_epoch))

    log_w.close()
    del loss
    torch.cuda.empty_cache()

# Evaluation
def get_probs_from_logits(logits):
    """
    Converts a tensor of logits into an array of probabilities by applying the sigmoid function
    """
    probs = torch.sigmoid(logits.unsqueeze(-1))
    return probs.detach().cpu().numpy()


def evaluation(net, device, dataloader, loss_type, with_labels=True, result_file="output/results.txt"):
    """
    Evaluation on dataset set.
    Calculate metrics including Accuracy, Kappa, P, R, F1
    """
    net.eval()
    w = open(result_file, 'w')
    preds = []
    golden_truth = []

    with torch.no_grad():
        if with_labels:
            for seq, attn_masks, token_type_ids, labels, indexes in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                if loss_type == "BCE":
                    probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
                    pred = np.where(probs > 0.5, 1, 0)
                elif loss_type == "CE":
                    pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
                preds += pred.tolist()
                golden_truth += labels.tolist()
    
    accuracy = accuracy_score(golden_truth, preds)
    kappa = cohen_kappa_score(golden_truth, preds)
    f1 = f1_score(golden_truth, preds, pos_label=1, average='binary')
    precision = precision_score(golden_truth, preds, pos_label=1, average='binary')
    recall = recall_score(golden_truth, preds, pos_label=1, average='binary')
    w.write("Accuracy on test data: {}\n".format(accuracy))
    w.write("Kappa on test data: {}\n".format(kappa))
    w.write("F1 score on test data: {}\n".format(f1))
    w.write("precision score on test data: {}\n".format(precision))
    w.write("recall score on test data: {}\n".format(recall))
    w.close()
    
def predict(net, device, dataloader, dataset, with_labels=True, relation_file="output/relation.tsv", logits_file="output/logits.pt", metric_file="output/metric.txt", loss_type="BCE"):
    """
    Predict on data set. 
    relation_file: for each question, write classified relations
    logits_file: write prediction logits
    metric_file: Accuracy, P, R, F1, Kappa
    loss_type: "BCE" | "CE"
    """
    net.eval()
    question_relations_map = defaultdict(list) 
    logits_collect = torch.randn(0).to(device)
    preds = []
    golden_truth = []

    with torch.no_grad():
        if with_labels:
            for seq, attn_masks, token_type_ids, labels, indexes in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                logits_collect = torch.cat((logits_collect, logits), 0)
                if loss_type == "BCE":
                    probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
                    pred = np.where(probs > 0.5, 1, 0)
                elif loss_type == "CE":
                    pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
                
                preds += pred.tolist()
                golden_truth += labels.tolist()
                indexes = indexes.tolist()
                true_indexes = [indexes[i] for i in range(0, len(indexes)) if pred[i] == 1]
                orig_items = [dataset.get_original_item(index) for index in true_indexes]
                    
                for item in orig_items:
                    question_relations_map[str(item["question"])].append(item["relation"])
    torch.save(logits_collect, logits_file)

    accuracy = accuracy_score(golden_truth, preds)
    kappa = cohen_kappa_score(golden_truth, preds)
    f1 = f1_score(golden_truth, preds, pos_label=1, average='binary')
    precision = precision_score(golden_truth, preds, pos_label=1, average='binary')
    recall = recall_score(golden_truth, preds, pos_label=1, average='binary')
    with open(metric_file, 'w') as f:
        f.write("Accuracy on test data: {}\n".format(accuracy))
        f.write("Kappa on test data: {}\n".format(kappa))
        f.write("F1 score on test data: {}\n".format(f1))
        f.write("Precision score on test data: {}\n".format(precision))
        f.write("Recall score on test data: {}\n".format(recall))
    
    with open(relation_file, 'w') as f:
        json.dump(question_relations_map, fp=f, indent=4)


def train_main(args):
    bert_model = args.cache_dir
    freeze_bert = False
    maxlen = args.max_len
    bs = args.batch_size
    iters_to_accumulate = args.iters_to_accumulate  # the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size
    lr = args.lr # learning rate
    epochs = args.epochs  # number of training epochs
    log_path = os.path.join(args.log_dir, 'log.txt')

    # Creating instances of training and validation set
    print("Reading training data...")
    train_df, dev_df, _ = data_process(args)
    print(train_df.shape)

    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    if args.mask_entity_mention:
        special_tokens_dict = {'additional_special_tokens': [BLANK_TOKEN]}
        tokenizer.add_special_tokens(special_tokens_dict)

    train_set = CustomDataset(train_df, maxlen, tokenizer=tokenizer, bert_model=bert_model)
    val_set = None
    if dev_df is not None:
        print("Reading validation data...")
        val_set = CustomDataset(dev_df, maxlen, tokenizer=tokenizer, bert_model=bert_model)
    # Creating instances of training and validation dataloaders
    train_loader = DataLoader(train_set, batch_size=bs, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=bs, num_workers=2) if val_set else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SentencePairClassifier(bert_model=bert_model, tokenizer=tokenizer, freeze_bert=freeze_bert)
    net.to(device)

    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    opti = AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    num_warmup_steps = 0 # The number of steps for the warmup phase.
    num_training_steps = epochs * len(train_loader)  # The total number of training steps
    t_total = (len(train_loader) // iters_to_accumulate) * epochs  # Necessary to take into account Gradient accumulation
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

    train_bert(args, net, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate, device, log_path, args.output_dir)


def evaluation_main(args):
    bert_model = args.cache_dir
    maxlen = args.max_len
    bs = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_to_model = args.model_save_path
    path_to_output_file = os.path.join(args.output_dir, "output.txt")
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    if args.mask_entity_mention:
        special_tokens_dict = {'additional_special_tokens': [BLANK_TOKEN]}
        tokenizer.add_special_tokens(special_tokens_dict)
    train_df, dev_df, test_df = data_process(args)
    if 'train' in args.predict_split:
        data_set = CustomDataset(train_df, maxlen, tokenizer=tokenizer, bert_model=bert_model)
    elif 'dev' in args.predict_split:
        data_set = CustomDataset(dev_df, maxlen, tokenizer=tokenizer, bert_model=bert_model)
    elif 'test' in args.predict_split:
        data_set = CustomDataset(test_df, maxlen, tokenizer=tokenizer, bert_model=bert_model)
    test_loader = DataLoader(data_set, batch_size=bs, num_workers=2)
    model = SentencePairClassifier(bert_model=bert_model, tokenizer=tokenizer)
    
    print("Loading the weights of the model...")
    model.load_state_dict(torch.load(path_to_model))
    model.to(device)
    
    print("Predicting on test data...")
    evaluation(net=model, device=device, dataloader=test_loader, loss_type=args.loss_type, with_labels=True, result_file=path_to_output_file)


def prediction_main(args):
    bert_model = args.cache_dir
    maxlen = args.max_len
    bs = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_to_model = args.model_save_path
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    if args.mask_entity_mention:
        special_tokens_dict = {'additional_special_tokens': [BLANK_TOKEN]}
        tokenizer.add_special_tokens(special_tokens_dict)
    if args.dataset_type.lower() == 'webqsp':
        print('webqsp')
        train_2hop_df, train_df, test_2hop_df, test_df = data_process(args)
        if args.predict_split == 'train_2hop':
            print('train_2hop_df')
            data_set = CustomDataset(train_2hop_df, maxlen, tokenizer=tokenizer, bert_model=bert_model)
        elif args.predict_split == 'train':
            print('train_df')
            data_set = CustomDataset(train_df, maxlen, tokenizer=tokenizer, bert_model=bert_model)
        elif args.predict_split == 'test_2hop':
            print('test_2hop_df')
            data_set = CustomDataset(test_2hop_df, maxlen, tokenizer=tokenizer, bert_model=bert_model)
        elif args.predict_split == 'test':
            print('test_df')
            data_set = CustomDataset(test_df, maxlen, tokenizer=tokenizer, bert_model=bert_model)
    else:
        train_df, dev_df, test_df = data_process(args)
        if 'train' in args.predict_split:
            data_set = CustomDataset(train_df, maxlen, tokenizer=tokenizer, bert_model=bert_model)
        elif 'dev' in args.predict_split:
            data_set = CustomDataset(dev_df, maxlen, tokenizer=tokenizer, bert_model=bert_model)
        elif 'test' in args.predict_split:
            data_set = CustomDataset(test_df, maxlen, tokenizer=tokenizer, bert_model=bert_model)

    test_loader = DataLoader(data_set, batch_size=bs, num_workers=2)
    model = SentencePairClassifier(bert_model=bert_model, tokenizer=tokenizer)
    
    print("Loading the weights of the model...")
    model.load_state_dict(torch.load(path_to_model))
    model.to(device)
    
    print("Predicting on test data...")
    predict(
        net=model, 
        device=device, 
        dataloader=test_loader, 
        dataset=data_set,
        with_labels=True, 
        relation_file=os.path.join(args.output_dir, 'relation.json'),
        logits_file=os.path.join(args.output_dir, "logits.pt"), 
        metric_file=os.path.join(args.output_dir, "metric.txt"),
        loss_type=args.loss_type
    )


if __name__=='__main__':
    args = _parse_args()
    print(args)
    set_seed(1)

    if args.do_train:
        train_main(args)
    if args.do_predict:
        prediction_main(args)
    if args.do_eval:
        evaluation_main(args)
