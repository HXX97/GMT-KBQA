"""
Build faiss index
Usage of faiss index
Generating relation/question vectors
Construct training/testing data for cross-encoder
"""
from collections import defaultdict
import collections
import json
import csv
from functools import reduce
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import torch
import faiss
import numpy
import pandas as pd
import argparse
import random
import re

from tqdm import tqdm

from biencoder import BiEncoderModule
from faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer

import os
import logging
import sys

BLANK_TOKEN = '[BLANK]'

def load_json(fname, mode="r", encoding="utf8"):
    if "b" in mode:
        encoding = None
    with open(fname, mode=mode, encoding=encoding) as f:
        return json.load(f)

def dump_json(obj, fname, indent=4, mode='w' ,encoding="utf8", ensure_ascii=False):
    if "b" in mode:
        encoding = None
    with open(fname, "w", encoding=encoding) as f:
        return json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def extract_mentioned_relations_from_sparql(sparql:str):
    """extract relation from sparql"""
    sparql = sparql.replace('(',' ( ').replace(')',' ) ')
    toks = sparql.split(' ')
    toks = [x for x in toks if len(x)]
    relation_tokens = []
    for t in toks:
        if (re.match("ns:[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*",t.strip()) 
            or re.match("ns:[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*",t.strip())):
            relation_tokens.append(t[3:])
    
    relation_tokens = list(set(relation_tokens))
    return relation_tokens


def write_json(json_path, data):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('action', type=str, help='Action to operate')
    parser.add_argument('--dataset', help='CWQ or WebQSP')
    parser.add_argument('--split', help='split to operate on') # the split file: ['dev','test','train']
    parser.add_argument('--cache_dir', default='hfcache/bert-base-uncased')

    return parser.parse_args()



def get_logger(output_dir=None):
    if output_dir != None:
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(
                    "{}/log.txt".format(output_dir), mode="a", delay=False
                ),
                logging.StreamHandler(sys.stdout),
            ],
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    logger = logging.getLogger('Blink')
    logger.setLevel(10)
    return logger


class CustomDataset(Dataset):
    def __init__(self, relations, maxlen, tokenizer=None, bert_model='bert-base-uncased'):
        self.relations = relations
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(bert_model)
        self.maxlen = maxlen
    
    def __len__(self):
        return len(self.relations)
    
    def __getitem__(self, index):
        relation = self.relations[index]
        encoded_relation = self.tokenizer(
            relation,
            padding='max_length',
            truncation=True,
            max_length=self.maxlen,
            return_tensors='pt'
        )
        relation_token_ids = encoded_relation['input_ids'].squeeze(0)  # tensor of token ids
        relation_attn_masks = encoded_relation['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        relation_token_type_ids = encoded_relation['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens
        
        return relation_token_ids, relation_attn_masks, relation_token_type_ids


def encode_relations(relations_path, model_path, save_path, add_special_tokens=False, max_len=32, batch_size=128, cache_dir='bert-base-uncased'):
    maxlen = max_len
    bs = batch_size
    bert_model = cache_dir

    if add_special_tokens:
        print('add special tokens')
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        special_tokens_dict = {'additional_special_tokens': [BLANK_TOKEN]}
        tokenizer.add_special_tokens(special_tokens_dict)
    else:
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiEncoderModule(device, bert_model=bert_model, tokenizer=tokenizer if tokenizer else None, freeze_bert=True)
    
    print("Loading the weights of the model...")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    with open(relations_path, 'r') as f:
        relations = json.load(f)
    
    relations_set = CustomDataset(relations, maxlen, bert_model=bert_model, tokenizer=tokenizer if tokenizer else None)
    relations_loader = DataLoader(relations_set, batch_size=bs, num_workers=2)
    
    relation_vectors = torch.zeros(0).to(device)
    for relation_token_ids, relation_attn_masks, relation_token_type_ids in tqdm(relations_loader):
        # print('relation_token_ids: {}'.format(relation_token_ids.shape))
        embedded_relation = model.encode_relation(
            relation_token_ids.to(device), 
            relation_attn_masks.to(device), 
            relation_token_type_ids.to(device)
        )
        relation_vectors = torch.cat((relation_vectors, embedded_relation), 0)
    print('relation_vectors: {}'.format(relation_vectors.shape))
    torch.save(relation_vectors, save_path)


def encode_questions(
    questions_path, 
    entity_linking_file, 
    model_path, 
    save_path, 
    max_len,
    cache_dir='hfcache/bert-base-uncased',
    add_special_tokens=False, 
    mask_mention=False,
    dataset='cwq',
    split=None
):
    maxlen = max_len
    bs = 128
    bert_model = cache_dir
    print(questions_path)
    if entity_linking_file is not None:
        entity_linking_res = load_json(entity_linking_file)
    else:
        entity_linking_res = None

    if add_special_tokens:
        print('add special tokens')
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        special_tokens_dict = {'additional_special_tokens': [BLANK_TOKEN]}
        tokenizer.add_special_tokens(special_tokens_dict)
    else:
        tokenizer = AutoTokenizer.from_pretrained(bert_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiEncoderModule(device, bert_model=bert_model, tokenizer=tokenizer, freeze_bert=True)
    
    print("Loading the weights of the model...")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    data = load_json(questions_path)
    if dataset == 'webqsp' and split in ['train', 'test']:
        data = data['Questions']
    questions = []
    for item in data:
        question = item["question"] if dataset == 'cwq' else item["ProcessedQuestion"]
        qid = item["ID"] if dataset == 'cwq' else item["QuestionId"]
        if mask_mention:
            question = question.lower()
            el_result = entity_linking_res[qid] if qid in entity_linking_res else []
            for eid in el_result:
                mention = el_result[eid]["mention"]
                question = question.replace(mention, BLANK_TOKEN)
        questions.append(question)
    
    questions_set = CustomDataset(questions, maxlen, bert_model=bert_model, tokenizer=tokenizer)
    questions_loader = DataLoader(questions_set, batch_size=bs, num_workers=2)
    
    questions_vectors = torch.zeros(0).to(device)
    for question_token_ids, question_attn_masks, question_token_type_ids in tqdm(questions_loader):
        # print('relation_token_ids: {}'.format(relation_token_ids.shape))
        embedded_question = model.encode_question(
            question_token_ids.to(device), 
            question_attn_masks.to(device), 
            question_token_type_ids.to(device)
        )
        questions_vectors = torch.cat((questions_vectors, embedded_question), 0)
    print('question_vectors: {}'.format(questions_vectors.shape))
    torch.save(questions_vectors, save_path)


def build_index(output_path, relation_vectors_path, index_buffer=50000):
    """
    index_buffer: Temporal memory data buffer size (in samples) for indexer
    """
    output_dir, _ = os.path.split(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = get_logger(output_dir)
    
    logger.info("Loading relation vectors from path: %s" % relation_vectors_path)
    relation_vectors = torch.load(relation_vectors_path).cpu().detach().numpy()
    vector_size = relation_vectors.shape[1]
    
    logger.info("Using Flat index in FAISS")
    index = DenseFlatIndexer(vector_size, index_buffer)
    # logger.info("Using HNSW index in FAISS")
    # index = DenseHNSWFlatIndexer(vector_size, index_buffer)
    
    logger.info("Building index.")
    index.index_data(relation_vectors)
    logger.info("Done indexing data.")
    
    index.serialize(output_path)



def calculate_recall(
    all_relations_path, 
    dataset_file, 
    question_vectors_path, 
    index_file, 
    vector_size=768, 
    index_buffer=50000, 
    top_k=500, 
    hnsw=False
):
    """
    all_relations_path: all relations in Freebase
    calculate relation linking recall
    """
    with open(all_relations_path, 'r') as f:
        all_relations = json.load(f)
    
    golden_relations = []
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    for item in data:
        golden_relations.append(list(item["gold_relation_map"].keys()))
    
    if hnsw:
        index = DenseHNSWFlatIndexer(vector_size, index_buffer)
    else:
        index = DenseFlatIndexer(vector_size, index_buffer)
    index.deserialize_from(index_file) 
    question_vectors = torch.load(question_vectors_path).cpu().detach().numpy()
    _, pred_relation_indexes = index.search_knn(question_vectors, top_k=top_k)
    pred_relations = pred_relation_indexes.tolist()
    pred_relations = [[all_relations[index] for index in indexes ] for indexes in pred_relation_indexes ]
    
    assert len(golden_relations) == len(pred_relations)
    
    recall = 0.0
    for i in range(len(golden_relations)):
        recall += (len([relation for relation in golden_relations[i] if relation in pred_relations[i]]) / len(golden_relations[i]))
    recall /= len(golden_relations)
    
    print('top{} recall: {}'.format(top_k, recall))
    return recall



def retrieve_candidate_relations_cwq(
    all_relations_path, 
    dataset_file, 
    question_vectors_path, 
    index_file, 
    output_path,
    CWQid_index_path,
    entity_linking_path,
    split,
    validation_relations_path=None,
    relation_rich_map_path=None,
    source_rich_relation=False,
    target_rich_relation=True,
    vector_size=768, 
    index_buffer=50000, 
    top_k=200,
    validation_top_k=100,
    mask_mention=True,
):
    """
    validation_relations_path: 2-hop relations from entity, for validation
    validation_top_k: After validation, reserve top-k relations
    """
    all_relations = load_json(all_relations_path)
    entity_linking_res = load_json(entity_linking_path)
    if relation_rich_map_path:
        relation_rich_map = load_json(relation_rich_map_path)
        
    if validation_relations_path:
        validation_relations_map = load_json(validation_relations_path)
    
    index = DenseFlatIndexer(vector_size, index_buffer)
    index.deserialize_from(index_file) 
    question_vectors = torch.load(question_vectors_path).cpu().detach().numpy()
    _, pred_relation_indexes = index.search_knn(question_vectors, top_k=top_k)
    pred_relations = pred_relation_indexes.tolist()
    pred_relations = [
        list([all_relations[index] for index in indexes])
        for indexes in pred_relation_indexes
    ]
    
    input_data = load_json(dataset_file)
    
    samples = []
    count = 0
    CWQid_index = dict()
    debug = True
    for idx in tqdm(range(len(input_data)), total=len(input_data)):
        start = count
        question = input_data[idx]["question"]
        id = input_data[idx]["ID"]
        if mask_mention:
            question = question.lower()
            el_result = entity_linking_res[id] if id in entity_linking_res else []
            for eid in el_result:
                mention = el_result[eid]["mention"]
                question = question.replace(mention, BLANK_TOKEN)
        if source_rich_relation and relation_rich_map:
            golden_relations = list(set([relation_rich_map[relation] if relation in relation_rich_map else relation for relation in input_data[idx]["gold_relation_map"].keys()]))
        else:
            golden_relations = list(set([relation for relation in input_data[idx]["gold_relation_map"].keys()]))
        
        if validation_relations_path:
            validation_relations = reduce(
                lambda x,y: x+validation_relations_map[y] if y in validation_relations_map else x,
                # input_data[idx]["disambiguated_cand_entity"],
                entity_linking_res[id].keys(),
                []
            )

        if validation_relations_path and relation_rich_map and source_rich_relation:
            validation_relations = [relation_rich_map[rel] if rel in relation_rich_map else rel for rel in validation_relations]

        if validation_relations_path:
            missed_relations = []
            for pred_relation in pred_relations[idx]:
                # 之前的 bug 长这样
                # if split == 'train':
                #     if pred_relation in golden_relations:
                #         pred_relation = relation_rich_map[pred_relation] if (pred_relation in relation_rich_map and target_rich_relation) else pred_relation
                #         samples.append([question, pred_relation, '1'])
                #         count += 1
                #     elif pred_relation in validation_relations:
                #         pred_relation = relation_rich_map[pred_relation] if (pred_relation in relation_rich_map and target_rich_relation) else pred_relation
                #         samples.append([question, pred_relation, '0'])
                #         count += 1
                #     else:
                #         missed_relations.append(pred_relation)
                # else: # `dev` or `test`
                #     if pred_relation in validation_relations:
                #         if pred_relation in golden_relations:
                #             pred_relation = relation_rich_map[pred_relation] if (pred_relation in relation_rich_map and target_rich_relation) else pred_relation
                #             samples.append([question, pred_relation, '1'])
                #         else:
                #             pred_relation = relation_rich_map[pred_relation] if (pred_relation in relation_rich_map and target_rich_relation) else pred_relation
                #             samples.append([question, pred_relation, '0'])
                #         count += 1
                #     else:
                #         missed_relations.append(pred_relation)
                if debug:
                    print('pred_relation: {}'.format(pred_relation))
                    print('validation_relations: {}'.format(validation_relations[0]))
                if pred_relation in validation_relations:
                    if split != 'test' and pred_relation in golden_relations:
                        pred_relation_rich = relation_rich_map[pred_relation] if (pred_relation in relation_rich_map and target_rich_relation) else pred_relation
                        samples.append([question, pred_relation_rich, '1'])
                    else:
                        pred_relation_rich = relation_rich_map[pred_relation] if (pred_relation in relation_rich_map and target_rich_relation) else pred_relation
                        samples.append([question, pred_relation_rich, '0'])
                    count += 1
                else:
                    missed_relations.append(pred_relation)

            # 如果筛选后的关系数量不足 validation_top_k， 则补齐
            if count - start < validation_top_k:
                for rel in missed_relations[:validation_top_k-(count - start)]:
                    if split != 'test' and rel in golden_relations:
                        rel_rich = relation_rich_map[rel] if (rel in relation_rich_map and target_rich_relation) else rel
                        samples.append([question, rel_rich, '1'])
                    else:
                        rel_rich = relation_rich_map[rel] if (rel in relation_rich_map and target_rich_relation) else rel
                        samples.append([question, rel_rich, '0'])
                    count += 1
        else:
            for pred_relation in pred_relations[idx]:
                if split != 'test' and pred_relation in golden_relations:
                    pred_relation = relation_rich_map[pred_relation] if (pred_relation in relation_rich_map and target_rich_relation) else pred_relation
                    samples.append([question, pred_relation, '1'])
                    count += 1
                else:
                    pred_relation = relation_rich_map[pred_relation] if (pred_relation in relation_rich_map and target_rich_relation) else pred_relation
                    samples.append([question, pred_relation, '0'])
                    count += 1
        end = count - 1
        CWQid_index[id] = {
            'start': start,
            'end': end
        }

        debug = False


    with open(output_path, 'w') as f:
        header = ['id', 'question', 'relation', 'label']
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        idx = 0
        for line in samples:
            writer.writerow([str(idx)] + line)
            idx += 1
    
    write_json(CWQid_index_path, CWQid_index)


def calculate_recall_cwq(
    dataset_file,
    id_index_path,
    tsv_file
):
    dataset = load_json(dataset_file)
    dataset = {item["ID"]: item for item in dataset}
    id_index_map = load_json(id_index_path)
    df = pd.read_csv(tsv_file, sep='\t', error_bad_lines=False).dropna()
    r_list = []
    for idx in tqdm(id_index_map, total=len(id_index_map)):
        example = dataset[idx]
        golden_relations = example["gold_relation_map"].keys()
        start_idx = id_index_map[idx]["start"]
        end_idx = id_index_map[idx]["end"]
        pred_relations = df[start_idx:end_idx+1]['relation'].unique().tolist()
        pred_relations = [rel.split('|')[0] for rel in pred_relations]
        recall = len(set(pred_relations) & set(golden_relations))/ len(set(golden_relations))
        r_list.append(recall)
    
    print('Recall: {}'.format(sum(r_list)/len(r_list)))


def retrieve_train_candidate_relations_webqsp(
    all_relations_path,
    dataset_path,
    output_path,
    output_id_index_path,
    question_vectors_path,
    index_path,
    biencoder_topk=50,
    vector_size=768, 
    index_buffer=50000, 
):
    """
    新的代码
    对于每个问题: 
    所有 golden 关系作为正例，
    bi-encoder 的 top50 结果，**不是 golden 关系的**，作为负例
    """
    all_relations = load_json(all_relations_path)
    dataset_with_golden_relations = load_json(dataset_path)
    samples = []
    count = 0
    id_index = dict()
    debug = True
    index = DenseFlatIndexer(vector_size, index_buffer)
    index.deserialize_from(index_path) 
    question_vectors = torch.load(question_vectors_path).cpu().detach().numpy()
    print('question_vectors: {}'.format(question_vectors.shape))
    _, pred_relation_indexes = index.search_knn(question_vectors, top_k=biencoder_topk)

    for idx in tqdm(range(len(dataset_with_golden_relations)), total=len(dataset_with_golden_relations)):
        start = count
        example = dataset_with_golden_relations[idx]
        qid = example["QuestionId"]
        question = example["ProcessedQuestion"].lower()
        golden_relations = example["gold_relation_map"].keys()
        if debug:
            print('golden_relations: {}'.format(golden_relations))
            print('question: {}'.format(question))

        # 先处理正例
        for golden_rel in golden_relations:
            # golden_rel 已经变成 rich_relation 形式
            samples.append([question, golden_rel, '1'])
            count += 1
        if debug:
            print(pred_relation_indexes[idx], len(pred_relation_indexes[idx]))
        pred_relations = [
            all_relations[index] for index in pred_relation_indexes[idx]
        ]
        # 处理负例
        if debug:
            print('L518: {} {}'.format(pred_relations, golden_relations))
        for n_rel in pred_relations:
            if n_rel in golden_relations:
                continue
            else:
                samples.append([question, n_rel, '0'])
                count += 1
        
        end = count - 1
        # map 中的 question 应该是原问题
        id_index[qid] = {
            'start': start,
            'end': end
        }
        if debug:
            print('samples: {}'.format(samples))
        debug=False

    with open(output_path, 'w') as f:
        header = ['id', 'question', 'relation', 'label']
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        idx = 0
        for line in samples:
            writer.writerow([str(idx)] + line)
            idx += 1
    
    write_json(output_id_index_path, id_index)


def retrieve_dev_and_test_candidate_relations_webqsp(
    all_relations_path,
    dataset_path,
    output_path,
    output_id_index_path,
    question_vectors_path,
    index_path,
    biencoder_topk=150, 
    vector_size=768, 
    index_buffer=50000
):
    """
    对于验证集和测试集，直接取 Bi-encoder 的 top-k 预测结果，构建数据集
    """
    all_relations = load_json(all_relations_path)
    # 仅用于给预测关系进行标记 '0'/'1'
    dataset_with_golden = load_json(dataset_path) 
    samples = []
    count = 0
    id_index = dict()
    debug = True

    index = DenseFlatIndexer(vector_size, index_buffer)
    index.deserialize_from(index_path) 
    question_vectors = torch.load(question_vectors_path).cpu().detach().numpy()
    print('question_vectors: {}'.format(question_vectors.shape))
    _, pred_relation_indexes = index.search_knn(question_vectors, top_k=biencoder_topk)

    for idx in tqdm(range(len(dataset_with_golden)), total=len(dataset_with_golden)):
        start = count
        example = dataset_with_golden[idx]
        qid = example["QuestionId"]
        question = example["ProcessedQuestion"].lower()
        # 仅用于标记数据
        golden_relations = example["gold_relation_map"].keys()

        if debug:
            print('question: {}'.format(question))
            print('golden_relations: {}'.format(golden_relations))
        
        pred_relations = [
            all_relations[index] for index in pred_relation_indexes[idx]
        ]
        
        for pred_rel in pred_relations:
            if pred_rel in golden_relations:
                samples.append([question, pred_rel, '1'])
            else:
                samples.append([question, pred_rel, '0'])
            count += 1

        end = count - 1
        id_index[qid] = {
            'start': start,
            'end': end
        }
        if debug:
            print('samples: {}'.format(samples))
        debug = False
    
    with open(output_path, 'w') as f:
        header = ['id', 'question', 'relation', 'label']
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        idx = 0
        for line in samples:
            writer.writerow([str(idx)] + line)
            idx += 1
    dump_json(id_index, output_id_index_path)


def retrieve_cross_encoder_data_yhshu(
    dataset_path,
    output_path,
    output_id_index_path,
    negative_sample_num=20
):
    dataset_with_golden_relations = load_json(dataset_path)
    golden_relations_map = dict()
    for example in dataset_with_golden_relations:
        golden_relations_map[example["QuestionId"]] = list(example["gold_relation_map"].keys())
        
    samples = []
    count = 0
    id_index = dict()
    debug = True

    for example in tqdm(dataset_with_golden_relations, total=len(dataset_with_golden_relations)):
        start = count
        qid = example["QuestionId"]
        question = example["ProcessedQuestion"].lower()
        golden_relations = example["gold_relation_map"].keys()

        # 先处理正例
        for golden_rel in golden_relations:
            # golden_rel 已经变成 rich_relation 形式
            samples.append([question, golden_rel, '1'])
            count += 1
        
        # 处理负例
        diff_qids = list(set(golden_relations_map.keys()) - set({qid}))
        random.seed(1)
        negative_samples = random.sample(diff_qids, negative_sample_num)
        if debug:
            print('L652: {}'.format(negative_samples))
        for n_qid in negative_samples:
            n_relations = golden_relations_map[n_qid]
            for n_rel in n_relations:
                if n_rel not in golden_relations:
                    samples.append([question, n_rel, '0'])
                    count += 1
        
        end = count - 1
        # map 中的 question 应该是原问题
        id_index[qid] = {
            'start': start,
            'end': end
        }
        if debug:
            print('samples: {}'.format(samples))
        debug=False

    with open(output_path, 'w') as f:
        header = ['id', 'question', 'relation', 'label']
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        idx = 0
        for line in samples:
            writer.writerow([str(idx)] + line)
            idx += 1
    
    write_json(output_id_index_path, id_index)

def retrieve_cross_encoder_inference_data(
    rng_linking_result_path,
    dataset_path,
    output_path,
    output_id_index_path,
):
    """
    每个问题，把实体链接得到实体的所有二跳关系，都放到数据集中
    """
    dataset = load_json(dataset_path)
    dataset = dataset["Questions"]
    rng_linking_results = load_json(rng_linking_result_path)
    rng_linking_results = {item["id"]: item for item in rng_linking_results}
    samples = []
    count = 0
    id_index = dict()
    debug = True
    for example in dataset:
        start = count
        qid = example["QuestionId"]
        question = example["ProcessedQuestion"].lower()
        assert qid in rng_linking_results, print(qid)
        two_hop_relations = rng_linking_results[qid]["two_hop_relations"]
        
        for rel in two_hop_relations:
            samples.append([question, rel, '0'])
            count += 1

        end = count - 1
        # map 中的 question 应该是原问题
        id_index[qid] = {
            'start': start,
            'end': end
        }
        if debug:
            print('samples: {}'.format(samples))
        debug=False

    with open(output_path, 'w') as f:
        header = ['id', 'question', 'relation', 'label']
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        idx = 0
        for line in samples:
            writer.writerow([str(idx)] + line)
            idx += 1
    
    write_json(output_id_index_path, id_index)

# bug 版本，请勿使用
# def retrieve_candidate_relations_prev(
#     all_relations_path, 
#     dataset_file, 
#     question_vectors_path, 
#     index_file, 
#     entity_linking_path,
#     output_path,
#     id_index_path,
#     prominent_type_path,
#     split,
#     validation_relations_path=None,
#     relation_rich_map_path=None,
#     vector_size=768, 
#     index_buffer=50000, 
#     top_k=200,
#     validation_top_k=100,
#     rich_entity=True,
#     mask_mention=False,
# ):
#     """
#     validation_relations_path: 从实体出发的一跳、二跳关系表，用于验证
#     validation_top_k: 进行KB验证后保留的候选关系数量
#     新的想法：训练集和验证集以选中的那个 parse 的关系作为正例，其他 parse 的关系加到负例中；然后再按照之前做法，添加其他关系
#     测试集：就直接使用 bi-encoder + 二跳验证之后的结果
#     """
#     all_relations = load_json(all_relations_path)
#     entity_linking_res = load_json(entity_linking_path)
#     entity_linking_res = {item["id"]: item for item in entity_linking_res}
#     prominent_type_map = load_json(prominent_type_path)
#     if relation_rich_map_path:
#         relation_rich_map = load_json(relation_rich_map_path)
#     else:
#         relation_rich_map = None
    
#     if validation_relations_path:
#         validation_relations_map = load_json(validation_relations_path)
    
#     index = DenseFlatIndexer(vector_size, index_buffer)
#     index.deserialize_from(index_file) 
#     question_vectors = torch.load(question_vectors_path).cpu().detach().numpy()
#     _, pred_relation_indexes = index.search_knn(question_vectors, top_k=top_k)
#     pred_relations = pred_relation_indexes.tolist()
#     pred_relations = [
#         list([all_relations[index] for index in indexes])
#         for indexes in pred_relation_indexes
#     ]
    
#     input_data = load_json(dataset_file)
    
#     samples = []
#     count = 0
#     id_index = dict()

#     print('split: {}'.format(split))

#     assert len(input_data) == len(pred_relations)
#     for idx in tqdm(range(len(input_data)), total=len(input_data)):
#         id = input_data[idx]["QuestionId"]
#         start = count
#         question = input_data[idx]["ProcessedQuestion"]
#         question = question.lower()
#         if rich_entity:
#             if id not in entity_linking_res:
#                 continue
#             for entity_idx in range(len(entity_linking_res[id]["freebase_ids"])):
#                 entity_id = entity_linking_res[id]["freebase_ids"][entity_idx]
#                 mention = entity_linking_res[id]['pred_tuples_string'][entity_idx][1]
#                 label = entity_linking_res[id]['pred_tuples_string'][entity_idx][0]
#                 prominent_type = prominent_type_map[entity_id][0] if entity_id in prominent_type_map else ''
#                 question = question.replace(mention, '|'.join([mention, label, prominent_type]))
#             # for entity_id in entity_linking_res[id]:
#             #     mention = entity_linking_res[id][entity_id]['mention']
#             #     label = entity_linking_res[id][entity_id]['label']
#             #     prominent_type = prominent_type_map[entity_id][0] if entity_id in prominent_type_map else ''
#             #     question = question.replace(mention, '|'.join([mention, label, prominent_type]))
#         elif mask_mention:
#             for entity_idx in range(len(entity_linking_res[id]["freebase_ids"])):
#                 mention = entity_linking_res[id]['pred_tuples_string'][entity_idx][1]
#                 question = question.replace(mention, BLANK_TOKEN)

#         if relation_rich_map:
#             golden_relations = list(set([relation_rich_map[relation] if relation in relation_rich_map else relation for relation in input_data[idx]["gold_relation_map"].keys()]))
#         else:
#             golden_relations = list(set([relation for relation in input_data[idx]["gold_relation_map"].keys()]))
        
#         if validation_relations_path:
#             validation_relations = reduce(
#                 lambda x,y: x+validation_relations_map[y] if y in validation_relations_map else x,
#                 entity_linking_res[id]["freebase_ids"],
#                 []
#             )

#         if validation_relations_path and relation_rich_map:
#             validation_relations = [relation_rich_map[rel] if rel in relation_rich_map else rel for rel in validation_relations]
        
#         if validation_relations_path:
#             missed_relations = []
#             for pred_relation in pred_relations[idx]:
#                 # 之前的 bug 长这样
#                 # if split == 'train':
#                 #     if pred_relation in golden_relations:
#                 #         samples.append([question, pred_relation, '1'])
#                 #         count += 1
#                 #     elif pred_relation in validation_relations:
#                 #         samples.append([question, pred_relation, '0'])
#                 #         count += 1
#                 #     else:
#                 #         missed_relations.append(pred_relation)
#                 # else:
#                 #     if pred_relation in validation_relations:
#                 #         if pred_relation in golden_relations:
#                 #             samples.append([question, pred_relation, '1'])
#                 #         else:
#                 #             samples.append([question, pred_relation, '0'])
#                 #         count += 1
#                 #     else:
#                 #         missed_relations.append(pred_relation)  
#                 if pred_relation in validation_relations:
#                     if split != 'test':
#                         if pred_relation in golden_relations:
#                             samples.append([question, pred_relation, '1'])
#                         else:
#                             samples.append([question, pred_relation, '0'])
#                     else:
#                         # test 统一标记为 '0'
#                         samples.append([question, pred_relation, '0'])
#                     count += 1
#                 else:
#                     missed_relations.append(pred_relation)  

#             # 不再补齐 validation_top_k
#             # 如果筛选后的关系数量不足 validation_top_k， 则补齐
#             if count - start < validation_top_k:
#                 # print(count - start)
#                 for rel in missed_relations[:validation_top_k-(count - start)]:
#                     if split != 'test':
#                         if rel in golden_relations:
#                             samples.append([question, rel, '1'])
#                         else:
#                             samples.append([question, rel, '0'])
#                     else:
#                         samples.append([question, rel, '0'])
#                     count += 1
#         else:
#             for pred_relation in pred_relations[idx]:
#                 if split != 'test':
#                     if pred_relation in golden_relations:
#                         samples.append([question, pred_relation, '1'])
#                         count += 1
#                     else:
#                         samples.append([question, pred_relation, '0'])
#                         count += 1
#                 else:
#                     samples.append([question, pred_relation, '0'])
#                     count += 1
        
#         # 打一个补丁，不要 richRelation, 就普通 relation
#         for sample in samples:
#             sample[1] = sample[1].split('|')[0]
        
#         end = count - 1
#         # map 中的 question 应该是原问题
#         id_index[id] = {
#             'start': start,
#             'end': end
#         }


#     with open(output_path, 'w') as f:
#         header = ['id', 'question', 'relation', 'label']
#         writer = csv.writer(f, delimiter='\t')
#         writer.writerow(header)
#         idx = 0
#         for line in samples:
#             writer.writerow([str(idx)] + line)
#             idx += 1
    
#     write_json(id_index_path, id_index)
    

def retrieve_candidate_relations_webqsp(
    all_relations_path, 
    dataset_file, 
    question_vectors_path, 
    index_file, 
    entity_linking_path,
    output_path,
    id_index_path,
    split,
    prominent_type_path=None,
    validation_relations_path=None,
    relation_rich_map_path=None,
    vector_size=768, 
    index_buffer=50000, 
    top_k=200,
    validation_top_k=100,
    rich_entity=True,
    mask_mention=False,
):
    """
    validation_relations_path: 从实体出发的一跳、二跳关系表，用于验证
    validation_top_k: 进行KB验证后保留的候选关系数量
    """
    all_relations = load_json(all_relations_path)
    entity_linking_res = load_json(entity_linking_path)
    entity_linking_res = {item["id"]: item for item in entity_linking_res}

    if prominent_type_path:
        prominent_type_map = load_json(prominent_type_path)
    else:
        prominent_type_map = None
    if relation_rich_map_path:
        relation_rich_map = load_json(relation_rich_map_path)
    else:
        relation_rich_map = None
    
    if validation_relations_path:
        validation_relations_map = load_json(validation_relations_path)
    else:
        validation_relations_map = None
    
    index = DenseFlatIndexer(vector_size, index_buffer)
    index.deserialize_from(index_file) 
    question_vectors = torch.load(question_vectors_path).cpu().detach().numpy()
    _, pred_relation_indexes = index.search_knn(question_vectors, top_k=top_k)
    pred_relations = pred_relation_indexes.tolist()
    pred_relations = [
        list([all_relations[index] for index in indexes])
        for indexes in pred_relation_indexes
    ]
    
    dataset = load_json(dataset_file)
    
    samples = []
    count = 0
    id_index = dict()

    print('split: {}'.format(split))

    assert len(dataset) == len(pred_relations)
    for idx in tqdm(range(len(dataset)), total=len(dataset)):
        qid = dataset[idx]["QuestionId"]
        start = count
        question = dataset[idx]["ProcessedQuestion"]
        question = question.lower()
        if rich_entity:
            if qid not in entity_linking_res:
                continue
            for entity_idx in range(len(entity_linking_res[qid]["freebase_ids"])):
                entity_id = entity_linking_res[qid]["freebase_ids"][entity_idx]
                mention = entity_linking_res[qid]['pred_tuples_string'][entity_idx][1]
                label = entity_linking_res[qid]['pred_tuples_string'][entity_idx][0]
                prominent_type = prominent_type_map[entity_id][0] if entity_id in prominent_type_map else ''
                question = question.replace(mention, '|'.join([mention, label, prominent_type]))
        elif mask_mention:
            for entity_idx in range(len(entity_linking_res[qid]["freebase_ids"])):
                mention = entity_linking_res[qid]['pred_tuples_string'][entity_idx][1]
                question = question.replace(mention, BLANK_TOKEN)

        if split != 'test':
            if relation_rich_map:
                golden_relations = list(set([relation_rich_map[relation] if relation in relation_rich_map else relation for relation in dataset[idx]["gold_relation_map"].keys()]))
            else:
                golden_relations = list(set([relation for relation in dataset[idx]["gold_relation_map"].keys()]))
        else:
            golden_relations = None
        
        if validation_relations_map:
            validation_relations = reduce(
                lambda x,y: x+validation_relations_map[y] if y in validation_relations_map else x,
                entity_linking_res[qid]["freebase_ids"],
                []
            )
        if validation_relations_map and relation_rich_map:
            validation_relations = [relation_rich_map[rel] if rel in relation_rich_map else rel for rel in validation_relations]
        
        if validation_relations_map:
            missed_relations = []
            for pred_relation in pred_relations[idx]:
                if pred_relation in validation_relations:
                    if split != 'test':
                        if pred_relation in golden_relations:
                            samples.append([question, pred_relation, '1'])
                        else:
                            samples.append([question, pred_relation, '0'])
                    else:
                        # for `test` split, label is set to '0' uniformly
                        samples.append([question, pred_relation, '0'])
                    count += 1
                else:
                    missed_relations.append(pred_relation)  

            # if sampled relations are less than `validation_top_k`, Filling up using `missed_relations`
            if count - start < validation_top_k:
                for rel in missed_relations[:validation_top_k-(count - start)]:
                    if split != 'test':
                        if rel in golden_relations:
                            samples.append([question, rel, '1'])
                        else:
                            samples.append([question, rel, '0'])
                    else:
                        samples.append([question, rel, '0'])
                    count += 1
        else:
            for pred_relation in pred_relations[idx]:
                if split != 'test':
                    if pred_relation in golden_relations:
                        samples.append([question, pred_relation, '1'])
                        count += 1
                    else:
                        samples.append([question, pred_relation, '0'])
                        count += 1
                else:
                    samples.append([question, pred_relation, '0'])
                    count += 1
        
        # convert rich relation from bi-encoder output to relation
        for sample in samples:
            sample[1] = sample[1].split('|')[0]
        
        end = count - 1
        id_index[qid] = {
            'start': start,
            'end': end
        }


    with open(output_path, 'w') as f:
        header = ['id', 'question', 'relation', 'label']
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        idx = 0
        for line in samples:
            writer.writerow([str(idx)] + line)
            idx += 1
    
    write_json(id_index_path, id_index)


def retrieve_candidate_relations_webqsp_0720(
    all_relations_path, 
    dataset_file,
    question_vectors_path, 
    index_file, 
    entity_linking_path,
    output_path,
    output_id_index_path,
    prominent_type_path,
    split,
    relation_rich_map_path=None,
    vector_size=768, 
    index_buffer=50000, 
    top_k=200,
    validation_top_k=100,
    rich_entity=True
):
    """
    尝试一下: rich entity + rich relation 
    + 训练集和验证集以选中的那个 parse 的关系作为正例，其他 parse 的关系加到负例中；然后再按照之前做法，添加其他关系
    测试集：就直接使用 bi-encoder + 二跳验证之后的结果
    """
    all_relations = load_json(all_relations_path)
    entity_linking_res = load_json(entity_linking_path)
    entity_linking_res = {item["id"]: item for item in entity_linking_res}
    prominent_type_map = load_json(prominent_type_path)
    if relation_rich_map_path:
        relation_rich_map = load_json(relation_rich_map_path)
    index = DenseFlatIndexer(vector_size, index_buffer)
    index.deserialize_from(index_file) 
    question_vectors = torch.load(question_vectors_path).cpu().detach().numpy()
    _, pred_relation_indexes = index.search_knn(question_vectors, top_k=top_k)
    pred_relations = pred_relation_indexes.tolist()
    pred_relations = [
        list([all_relations[index] for index in indexes])
        for indexes in pred_relation_indexes
    ]

    input_data = load_json(dataset_file)
    
    samples = []
    count = 0
    id_index = dict()
    debug=True

    print('split: {}'.format(split))
    for idx in tqdm(range(len(input_data)), total=len(input_data)):
        example = input_data[idx]
        qid = example["QuestionId"]
        start = count
        question = example["ProcessedQuestion"].lower()

        if rich_entity:
            if qid not in entity_linking_res:
                continue
            for entity_idx in range(len(entity_linking_res[qid]["freebase_ids"])):
                entity_id = entity_linking_res[qid]["freebase_ids"][entity_idx]
                mention = entity_linking_res[qid]['pred_tuples_string'][entity_idx][1]
                label = entity_linking_res[qid]['pred_tuples_string'][entity_idx][0]
                prominent_type = prominent_type_map[entity_id][0] if entity_id in prominent_type_map else ''
                question = question.replace(mention, '|'.join([mention, label, prominent_type]))
        
        # if validation_relations_path:
        #     validation_relations = reduce(
        #         lambda x,y: x+validation_relations_map[y] if y in validation_relations_map else x,
        #         entity_linking_res[id]["freebase_ids"],
        #         []
        #     )
        validation_relations = entity_linking_res[qid]["two_hop_relations"]

        if relation_rich_map:
            validation_relations = [relation_rich_map[rel] if rel in relation_rich_map else rel for rel in validation_relations]
        
        sampled_rels = []
        # 训练集和验证集以选中的那个 parse 的关系作为正例，其他 parse 的关系加到负例中
        if split == 'ptrain' or split == 'pdev':
            print('ptrain or pdev!')
            parses = example["Parses"]
            golden_relation_map = []
            shortest_idx = 0
            shortest_len = 9999
            for i in range(len(parses)):
                golden_relation_map.append(extract_mentioned_relations_from_sparql(parses[i]['Sparql']))
                if 'SExpr_execute_right' in parses[i] and parses[i]['SExpr_execute_right']:
                    if len(parses[i]['Sparql']) < shortest_len:
                        shortest_idx = i
                        shortest_len = len(parses[i]['Sparql'])
            # 选中的 parse 中的 golden relation
            golden_relations = golden_relation_map[shortest_idx]
            all_relations = example["gold_relation_map"].keys()
            for rel in all_relations:
                if rel in golden_relations:
                    samples.append([question, relation_rich_map[rel] if rel in relation_rich_map else rel, '1'])
                else:
                    samples.append([question, relation_rich_map[rel] if rel in relation_rich_map else rel, '0'])
                sampled_rels.append(relation_rich_map[rel] if rel in relation_rich_map else rel) # 已采样的关系
        
        if validation_relations:
            missed_relations = []
            for pred_relation in pred_relations[idx]:
                if debug:
                    print(pred_relation)
                if pred_relation not in sampled_rels:
                    if pred_relation in validation_relations:
                        samples.append([question, pred_relation, '0'])
                        count += 1
                    else:
                        missed_relations.append(pred_relation)
            
            if count - start < validation_top_k:
                print(count - start)
                for rel in missed_relations[:validation_top_k-(count - start)]:
                    if pred_relation not in sampled_rels:
                        samples.append([question, rel, '0'])
                        count += 1
        else:
            for pred_relation in pred_relations[idx]:
                if pred_relation not in sampled_rels:
                    samples.append([question, pred_relation, '0'])
                    count += 1
        
        end = count - 1
        # map 中的 question 应该是原问题
        id_index[qid] = {
            'start': start,
            'end': end
        }
        debug=False
    with open(output_path, 'w') as f:
        header = ['id', 'question', 'relation', 'label']
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        idx = 0
        for line in samples:
            writer.writerow([str(idx)] + line)
            idx += 1
    
    write_json(output_id_index_path, id_index)





def calculate_recall_webqsp(
    dataset_file,
    id_index_path,
    tsv_file
):
    dataset = load_json(dataset_file)
    dataset = {item["QuestionId"]: item for item in dataset}
    id_index_map = load_json(id_index_path)
    df = pd.read_csv(tsv_file, sep='\t', error_bad_lines=False).dropna()
    r_list = []
    for idx in tqdm(id_index_map, total=len(id_index_map)):
        example = dataset[idx]
        golden_relations = example["gold_relation_map"].keys()
        start_idx = id_index_map[idx]["start"]
        end_idx = id_index_map[idx]["end"]
        pred_relations = df[start_idx:end_idx+1]['relation'].unique().tolist()
        pred_relations = [rel.split('|')[0] for rel in pred_relations]
        recall = len(set(pred_relations) & set(golden_relations))/ len(set(golden_relations))
        r_list.append(recall)
    
    print('Recall: {}'.format(sum(r_list)/len(r_list)))

def make_partial_train_dev(train_split_path):
    random.seed(17)
    data = load_json(train_split_path)["Questions"]
    random.shuffle(data)
    ptrain = data[:-200]
    pdev = data[-200:]
    print(len(ptrain))
    print(len(pdev))
    dump_json(ptrain, f'data/WebQSP/origin/WebQSP.ptrain.json', indent=4)
    dump_json(pdev, f'data/WebQSP/origin/WebQSP.pdev.json', indent=4)

def split_file(
    data_path,
    train_path,
    dev_path,
    output_train_path,
    output_dev_path
):
    data = load_json(data_path)
    qid_data_map = {item["QuestionId"]: item for item in data}
    prev_train = load_json(train_path)
    prev_dev = load_json(dev_path)

    # 保持问题的顺序不变
    train_data_with_golden = [qid_data_map[example["QuestionId"]] for example in prev_train]
    split_dev_with_golden = [qid_data_map[example["QuestionId"]] for example in prev_dev]
    
    print('prev_train: {}, new_train: {}'.format(len(prev_train), len(train_data_with_golden)))
    print('prev_dev: {}, new_dev: {}'.format(len(prev_dev), len(split_dev_with_golden)))

    dump_json(train_data_with_golden, output_train_path)
    dump_json(split_dev_with_golden, output_dev_path)

def validate_data_sequence(
    data_path_1,
    data_path_2,
):
    data_1 = load_json(data_path_1)
    data_2 = load_json(data_path_2)
    data_1_qids = [example["QuestionId"] for example in data_1]
    data_2_qids = [example["QuestionId"] for example in data_2]
    print(data_1_qids == data_2_qids)

# def split_tsv_file(
#     tsv_file_path,
#     id_index_path,
#     train_data_path,
#     dev_data_path
# ):
#     tsv_df = pd.read_csv(tsv_file_path, sep='\t', error_bad_lines=False).dropna()

def get_cross_encoder_tsv_max_len(tsv_file):
    tsv_df = pd.read_csv(tsv_file, sep='\t', error_bad_lines=False).dropna()
    tokenizer = AutoTokenizer.from_pretrained('hfcache/bert-base-uncased')
    length_dict = defaultdict(int)
    for idx in tqdm(range(len(tsv_df)), total=len(tsv_df)):
        question = tsv_df.loc[idx, 'question']
        relation = tsv_df.loc[idx, 'relation']
        tokenized = tokenizer.tokenize(question, relation)
        length_dict[len(tokenized)] += 1
    print(collections.OrderedDict(sorted(length_dict.items())))

def main():
    # Save all freebase relations as vectors using trained Bi-encoder
    """
    add_special_tokens: True if mask entity mentions
    max_len: 32 for CWQ, 80 for WEBQSP with rich relations
    """
    # CWQ
    # save_relations_vectors(
    #     '../../Data/common_data/freebase_relations_filtered.json',
    #     '../../Data/CWQ/relation_retrieval/bi-encoder/saved_models/mask_entity_ep1.pt',
    #     '../../Data/CWQ/relation_retrieval/bi-encoder/vectors/mask_entity_ep1_relations.pt',
    #     add_special_tokens=True,
    #     max_len=32,
    #     batch_size=128
    # )

    # WEBQSP
    # save_relations_vectors(
    #     '../../Data/common_data/freebase_relations_filtered.json',
    #     '../../Data/WEBQSP/relation_retrieval/bi-encoder/saved_models/richRelation_3epoch/bert-base-uncased_lr_2e-05_ep_1.pt',
    #     '../../Data/WEBQSP/relation_retrieval/bi-encoder/vectors/test/mask_entity_ep1_relations.pt',
    #     add_special_tokens=False,
    #     max_len=80,
    #     batch_size=128
    # )

    # Save questions in dataset as vectors using trained Bi-encoder
    """
    add_special_tokens: True if mask entity mentions
    """
    # CWQ
    # save_questions_vectors(
    #     '../../Data/CWQ/generation/merged/CWQ_test.json',
    #     '../../Data/CWQ/entity_retrieval/linking_results/merged_CWQ_test_linking_results.json',
    #     '../../Data/CWQ/relation_retrieval/bi-encoder/saved_models/mask_entity_ep1.pt',
    #     '../../Data/CWQ/relation_retrieval/bi-encoder/vectors/questions_test.pt',
    #     add_special_tokens=True,
    #     mask_mention=True
    # )

    # WEBQSP
    # save_questions_vectors(
    #     '../../Data/WEBQSP/generation/merged/WebQSP_test.json',
    #     '../../Data/WEBQSP/entity_retrieval/linking_results/merged_WebQSP_test_linking_results.json',
    #     '../../Data/WEBQSP/relation_retrieval/bi-encoder/saved_models/richRelation_3epoch/bert-base-uncased_lr_2e-05_ep_1.pt',
    #     '../../Data/WEBQSP/relation_retrieval/bi-encoder/vectors/test/questions_test.pt',
    #     add_special_tokens=False,
    #     mask_mention=False
    # )

    # build index with FAISS for all relations in freebase
    # CWQ
    # build_index(
    #     '../../Data/CWQ/relation_retrieval/bi-encoder/index/mask_entity_flat.index',
    #     '../../Data/CWQ/relation_retrieval/bi-encoder/vectors/mask_entity_ep1_relations.pt'
    # )

    # WEBQSP
    # build_index(
    #     '../../Data/WEBQSP/relation_retrieval/bi-encoder/index/test/mask_entity_flat.index',
    #     '../../Data/WEBQSP/relation_retrieval/bi-encoder/vectors/test/relation_test.pt'
    # )

    # calculate recall of linked entities

    # CWQ
    # calculate_recall(
    #     '../../Data/common_data/freebase_relations_filtered.json',
    #     '../../Data/CWQ/generation/merged/CWQ_test.json',
    #     '../../Data/CWQ/relation_retrieval/bi-encoder/vectors/questions_test.pt',
    #     '../../Data/CWQ/relation_retrieval/bi-encoder/index/mask_entity_flat.index',
    #     top_k=300
    # )

    # WEBQSP
    calculate_recall(
        '../../Data/common_data/freebase_relations_filtered.json',
        '../../Data/WEBQSP/generation/merged/WebQSP_test.json',
        '../../Data/WEBQSP/relation_retrieval/bi-encoder/vectors/test/questions_test.pt',
        '../../Data/WEBQSP/relation_retrieval/bi-encoder/index/test/mask_entity_flat.index',
        top_k=300
    )

    # CWQ, create crossEncoder source data using trained Bi-encoder
    """
    relation_rich_map_path: None as default
    source_rich_relation: whether relations inferenced by faiss has been enriched -- False for CWQ, True for WEBQSP
    target_rich_relation: whether relations in cross-encoder source data will be enriched -- both True for CWQ and WEBQSP
    """
    # CWQ_create_CrossEncoder_sample_tsv(
    #     '../../Data/common_data/freebase_relations_filtered.json',
    #     '../../Data/CWQ/generation/merged/CWQ_test.json',
    #     '../../Data/CWQ/relation_retrieval/bi-encoder/vectors/questions_test.pt',
    #     '../../Data/CWQ/relation_retrieval/bi-encoder/index/mask_entity_flat.index',
    #     '../../Data/CWQ/relation_retrieval/cross-encoder/maskMention.richRelation.2hopValidation.test.tsv',
    #     '../../Data/CWQ/relation_retrieval/cross-encoder/maskMention.richRelation.2hopValidation.test.question_index_map.json',
    #     '../../Data/CWQ/entity_retrieval/linking_results/merged_CWQ_test_linking_results.json',
    #     validation_relations_path='../../Data/CWQ/relation_retrieval/bi-encoder/CWQ.2hopRelations.candEntities.json',
    #     relation_rich_map_path='../../Data/common_data/fb_relation_rich_map.json',
    #     source_rich_relation=False,
    #     target_rich_relation=True,
    #     mask_mention=True,
    #     rich_entity=False
    # )


if __name__=='__main__':
    args = _parse_args()
    action = args.action

    if action.lower() == 'encode_relation':
        if args.dataset.lower() == 'cwq':
            encode_relations(
                'data/common_data/freebase_relations_filtered.json',
                'data/CWQ/relation_retrieval_0723/bi-encoder/saved_models/CWQ_ep_1.pt',
                'data/CWQ/relation_retrieval_0723/bi-encoder/vectors/CWQ_ep_1_relations.pt',
                add_special_tokens=True,
                max_len=32,
                batch_size=128,
                cache_dir=args.cache_dir
            )
        elif args.dataset.lower() == 'webqsp':
            encode_relations(
                'data/common_data/freebase_richRelations_filtered.json', # depends on `normal relation` or `rich relation` biencoder was trained on
                'data/WebQSP/relation_retrieval_final/bi-encoder/saved_models/rich_relation_3epochs/WebQSP_ep_3.pt',
                'data/WebQSP/relation_retrieval_final/bi-encoder/vectors/rich_relation_3epochs/WebQSP_ep_3_relations.pt',
                add_special_tokens=False,
                max_len=60, # consistent with training scripts, e.g.`run_bi_encoder_WebQSP.sh`
                batch_size=128,
                cache_dir=args.cache_dir
            )
    if action.lower() == 'build_index':
        if args.dataset.lower() == 'cwq':
            build_index(
                'data/CWQ/relation_retrieval_0723/bi-encoder/index/ep_1_flat.index',
                'data/CWQ/relation_retrieval_0723/bi-encoder/vectors/CWQ_ep_1_relations.pt'
            )
        elif args.dataset.lower() == 'webqsp':
            build_index(
                'data/WebQSP/relation_retrieval_final/bi-encoder/index/rich_relation_3epochs/ep_3_flat.index',
                'data/WebQSP/relation_retrieval_final/bi-encoder/vectors/rich_relation_3epochs/WebQSP_ep_3_relations.pt'
            )
    if action.lower() == 'encode_question':
        if args.dataset.lower() == 'cwq':
            encode_questions(
                'data/CWQ/origin/ComplexWebQuestions_{}.json'.format(args.split),
                'data/CWQ/entity_retrieval/merged_linking_results/merged_CWQ_{}_linking_results.json'.format(args.split),
                'data/CWQ/relation_retrieval_0723/bi-encoder/saved_models/CWQ_ep_1.pt',
                'data/CWQ/relation_retrieval_0723/bi-encoder/vectors/CWQ_{}_questions.pt'.format(args.split),
                max_len=32,
                cache_dir='hfcache/bert-base-uncased',
                add_special_tokens=True,
                mask_mention=True,
                dataset=args.dataset.lower(),
                split=args.split
            )
        elif args.dataset.lower() == 'webqsp':
            encode_questions(
                'data/WebQSP/origin/WebQSP.{}.json'.format(args.split),
                None,
                'data/WebQSP/relation_retrieval_final/bi-encoder/saved_models/rich_relation_3epochs/WebQSP_ep_3.pt',
                'data/WebQSP/relation_retrieval_final/bi-encoder/vectors/rich_relation_3epochs/WebQSP_{}_ep3_questions.pt'.format(args.split),
                max_len=60, # consistent with training scripts, e.g.`run_bi_encoder_WebQSP.sh`
                cache_dir='hfcache/bert-base-uncased',
                add_special_tokens=False,
                mask_mention=False,
                dataset=args.dataset.lower(),
                split=args.split
            )
    if action.lower() == 'retrieve_relations':
        if args.dataset.lower() == 'cwq':
            retrieve_candidate_relations_cwq(
                'data/common_data/freebase_relations_filtered.json', 
                'data/CWQ/relation_retrieval_0723/bi-encoder/CWQ.{}.goldenRelation.json'.format(args.split),
                'data/CWQ/relation_retrieval_0723/bi-encoder/vectors/CWQ_{}_questions.pt'.format(args.split),
                'data/CWQ/relation_retrieval_0723/bi-encoder/index/ep_1_flat.index',
                'data/CWQ/relation_retrieval_0723/cross-encoder/mask_mention_1epoch_question_relation/CWQ.{}.tsv'.format(args.split),
                'data/CWQ/relation_retrieval_0723/cross-encoder/mask_mention_1epoch_question_relation/CWQ_{}_id_index_map.json'.format(args.split),
                'data/CWQ/entity_retrieval/merged_linking_results/merged_CWQ_{}_linking_results.json'.format(args.split),
                args.split,
                validation_relations_path='data/CWQ/relation_retrieval_0723/bi-encoder/CWQ.2hopRelations.candEntities.json',
                relation_rich_map_path='data/common_data/fb_relation_rich_map.json',
                source_rich_relation=False,
                # target_rich_relation=True,
                # mask_mention=True,
                target_rich_relation=False,
                mask_mention=False
            )
            calculate_recall_cwq(
                'data/CWQ/relation_retrieval_0723/bi-encoder/CWQ.{}.goldenRelation.json'.format(args.split),
                'data/CWQ/relation_retrieval_0723/cross-encoder/mask_mention_1epoch_question_relation/CWQ_{}_id_index_map.json'.format(args.split),
                'data/CWQ/relation_retrieval_0723/cross-encoder/mask_mention_1epoch_question_relation/CWQ.{}.tsv'.format(args.split),
                # 'data/CWQ/relation_retrieval/cross-encoder/CWQ.test.biEncoder.train_all.maskMention.crossEncoder.2hopValidation.maskMention.richRelation.top100_CWQid_index_map.json',
                # 'data/CWQ/relation_retrieval/cross-encoder/CWQ.test.biEncoder.train_all.maskMention.crossEncoder.2hopValidation.maskMention.richRelation.top100.tsv'
            )
            # Help to decide max length of cross-encoder training, 50 is suggested
            # get_cross_encoder_tsv_max_len(
            #     'data/CWQ/relation_retrieval_0723/cross-encoder/mask_mention_1epoch_question_relation/CWQ.{}.tsv'.format(args.split),
            # )
        elif args.dataset.lower() == 'webqsp':
            # 如果测试集还没划分好的话，先划分
            # if not os.path.exists('data/WebQSP/origin/WebQSP.pdev.json'):
            #     print('dev split')
            #     make_partial_train_dev('data/WebQSP/origin/WebQSP.train.json')
            # 相应的, goldenRelation.json 也要有相同的划分
            # if not os.path.exists('data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.pdev.goldenRelation.json'):
            #     print('splitting dev')
            #     split_file(
            #         'data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.train.goldenRelation.json',
            #         'data/WebQSP/origin/WebQSP.ptrain.json',
            #         'data/WebQSP/origin/WebQSP.pdev.json',
            #         'data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.ptrain.goldenRelation.json',
            #         'data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.pdev.goldenRelation.json',
            #     )
            #     for split in ['ptrain', 'pdev']:
            #         validate_data_sequence(
            #             f'data/WebQSP/origin/WebQSP.{split}.json',
            #             f'data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.{split}.goldenRelation.json',
            #         )
            # if args.split in ['ptrain']:
            #     retrieve_train_candidate_relations_webqsp(
            #         'data/common_data/freebase_relations_filtered.json',
            #         f'data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.{args.split}.goldenRelation.json',
            #         f'data/WebQSP/relation_retrieval_0717/cross-encoder/rich_relation_3epochs_question_relation_top100/WebQSP.{args.split}.tsv',
            #         f'data/WebQSP/relation_retrieval_0717/cross-encoder/rich_relation_3epochs_question_relation_top100/WebQSP_{args.split}_id_index_map.json',
            #         f'data/WebQSP/relation_retrieval_0717/bi-encoder/vectors/rich_relation_3epochs/WebQSP_{args.split}_ep3_questions.pt',
            #         'data/WebQSP/relation_retrieval_0717/bi-encoder/index/rich_relation_3epochs/ep_3_flat.index',
            #         biencoder_topk=100
            #     )
            # elif args.split in ['train', 'pdev', 'test']:
            #     # 为什么 train 也调用这个函数？因为生成任务训练集也需要 top10 的关系，需要和测试集保持一致
            #     print('retrieve test relations')
            #     retrieve_dev_and_test_candidate_relations_webqsp(
            #         'data/common_data/freebase_relations_filtered.json',
            #         f'data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.{args.split}.goldenRelation.json',
            #         f'data/WebQSP/relation_retrieval_0717/cross-encoder/rich_relation_3epochs_question_relation_top100/WebQSP.{args.split}.tsv',
            #         f'data/WebQSP/relation_retrieval_0717/cross-encoder/rich_relation_3epochs_question_relation_top100/WebQSP_{args.split}_id_index_map.json',
            #         f'data/WebQSP/relation_retrieval_0717/bi-encoder/vectors/rich_relation_3epochs/WebQSP_{args.split}_ep3_questions.pt',
            #         'data/WebQSP/relation_retrieval_0717/bi-encoder/index/rich_relation_3epochs/ep_3_flat.index',
            #         biencoder_topk=100
            #     )

            # retrieve_cross_encoder_data_yhshu(
            #     f'data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.{args.split}.goldenRelation.json',
            #     f'data/WebQSP/relation_retrieval_0717/cross-encoder/rich_relation_3epochs_question_relation_yhshu/WebQSP.{args.split}.tsv',
            #     f'data/WebQSP/relation_retrieval_0717/cross-encoder/rich_relation_3epochs_question_relation_yhshu/WebQSP_{args.split}_id_index_map.json',
            #     negative_sample_num=20
            # )
            
            # 该方法得到WebQSP cross-encoder的推理数据；已确认得到的数据完全一致
            if args.split in ['train_2hop', 'test_2hop']:
                # 将 rng 实体链接结果中的所有二跳关系作为候选关系，利用 cross-encoder 来预测
                subsp = 'train' if args.split == 'train_2hop' else 'test'
                retrieve_cross_encoder_inference_data(
                    f'data/WebQSP/relation_retrieval_final/cross-encoder/rng_kbqa_linking_results/webqsp_{subsp}_rng_el_two_hop_relations.json',
                    f'data/WebQSP/origin/WebQSP.{subsp}.json',
                    f'data/WebQSP/relation_retrieval_final/cross-encoder/rich_relation_3epochs_question_relation/WebQSP.{args.split}.tsv',
                    f'data/WebQSP/relation_retrieval_final/cross-encoder/rich_relation_3epochs_question_relation/WebQSP_{args.split}_id_index_map.json',
                )

            # 该方法得到 WebQSP cross-encoder 的训练、验证数据
            # 第一个参数传 rich_relation, 得到 rich_relation; 传普通 relation, 得到普通 relation
            # 这就是生成 rich_relation_3epochs_question_relation 下采样数据的方法(会用到的是 train, 用于模型训练，后面应该是 ptrain 和 pdev)
            # 已确认用这个方法可以生成完全一致的采样数据
            if args.split in ['train', 'ptrain', 'pdev', 'test']:
                # TODO: 2hop 关系直接利用数据集里头的 "2hop_relations"
                retrieve_candidate_relations_webqsp(
                    'data/common_data/freebase_relations_filtered.json', # determines what kind of sampled relations will be generated (normal/rich)
                    'data/WebQSP/relation_retrieval_final/bi-encoder/WebQSP.{}.goldenRelation.json'.format(args.split),
                    'data/WebQSP/relation_retrieval_final/bi-encoder/vectors/rich_relation_3epochs/WebQSP_{}_ep3_questions.pt'.format(args.split),
                    'data/WebQSP/relation_retrieval_final/bi-encoder/index/rich_relation_3epochs/ep_3_flat.index',
                    'data/WebQSP/relation_retrieval_final/cross-encoder/rng_kbqa_linking_results/webqsp_train_rng_el_two_hop_relations.json' if args.split in ['train', 'ptrain', 'pdev'] else 'data/WebQSP/relation_retrieval_final/cross-encoder/rng_kbqa_linking_results/webqsp_test_rng_el_two_hop_relations.json',
                    # 'data/WebQSP/entity_retrieval/linking_results/merged_WebQSP_{}_linking_results.json'.format(args.split),
                    'data/WebQSP/relation_retrieval_final/cross-encoder/rich_relation_3epochs_question_relation/WebQSP.{}.tsv'.format(args.split),
                    'data/WebQSP/relation_retrieval_final/cross-encoder/rich_relation_3epochs_question_relation/WebQSP_{}_id_index_map.json'.format(args.split),
                    args.split,
                    prominent_type_path=None,
                    validation_relations_path='data/WebQSP/relation_retrieval_final/cross-encoder/rng_kbqa_linking_results/WebQSP.2hopRelations.rng.elq.candEntities.json',
                    # relation_rich_map_path='data/common_data/fb_relation_rich_map.json',
                    rich_entity=False,
                    # rich_entity=False,
                    top_k=200,
                    validation_top_k=100,
                    mask_mention=False,
                )

            # Help to decide max length of cross-encoder training, 34 is suggested
            # get_cross_encoder_tsv_max_len(
            #     'data/WebQSP/relation_retrieval_final/cross-encoder/rich_relation_3epochs_question_relation/WebQSP.train.tsv'
            # )

            # subsplit = 'train' if args.split in ['train', 'ptrain', 'pdev'] else 'test'
            # retrieve_candidate_relations_webqsp_0720(
            #     'data/common_data/freebase_richRelations_filtered.json',
            #     'data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.{}.goldenRelation.json'.format(args.split),
            #     'data/WebQSP/relation_retrieval_0717/bi-encoder/vectors/rich_relation_3epochs/WebQSP_{}_ep3_questions.pt'.format(args.split),
            #     'data/WebQSP/relation_retrieval_0717/bi-encoder/index/rich_relation_3epochs/ep_3_flat.index',
            #     'data/WebQSP/relation_retrieval_0717/cross-encoder/rng_kbqa_linking_results/webqsp_{}_rng_el_two_hop_relations.json'.format(subsplit),
            #     'data/WebQSP/relation_retrieval_0717/cross-encoder/rich_relation_3epochs_rich_entity_rich_relation_1parse/WebQSP.{}.tsv'.format(args.split),
            #     'data/WebQSP/relation_retrieval_0717/cross-encoder/rich_relation_3epochs_rich_entity_rich_relation_1parse/WebQSP_{}_id_index_map.json'.format(args.split),
            #     'data/WebQSP/relation_retrieval/cross-encoder/rng_linking_results/WebQSP.entity.prominentType.map.rng.elq.json',
            #     args.split,
            #     relation_rich_map_path='data/common_data/fb_relation_rich_map.json',
            #     top_k=200,
            #     validation_top_k=100,
            #     rich_entity=True
            # )

            # calculate_recall_webqsp(
            #     'data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.{}.goldenRelation.json'.format(args.split),
            #     'data/WebQSP/relation_retrieval_0717/cross-encoder/rich_relation_3epochs_question_relation_2hop_validation_top300/WebQSP_{}_id_index_map.json'.format(args.split),
            #     'data/WebQSP/relation_retrieval_0717/cross-encoder/rich_relation_3epochs_question_relation_2hop_validation_top300/WebQSP.{}.tsv'.format(args.split),
            #     # 'data/WebQSP/relation_retrieval/cross-encoder/WebQSP.test.biEncoder.train_all.richRelation.crossEncoder.train_all.richRelation.2hopValidation.richEntity.top100.1parse_WebQSPid_index_map.json',
            #     # 'data/WebQSP/relation_retrieval/cross-encoder/WebQSP.test.biEncoder.train_all.richRelation.crossEncoder.train_all.richRelation.2hopValidation.richEntity.top100.1parse.tsv'
            # )
            # calculate_recall(
            #     'data/common_data/freebase_relations_filtered.json',
            #     'data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.test.goldenRelation.json',
            #     'data/WebQSP/relation_retrieval_0717/bi-encoder/vectors/rich_relation_3epochs/WebQSP_test_ep3_questions.pt',
            #     'data/WebQSP/relation_retrieval_0717/bi-encoder/index/rich_relation_3epochs/ep_3_flat.index',
            #     top_k=150
            # )

            
