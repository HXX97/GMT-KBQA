"""
Build faiss index
Usage of faiss index
Generating relation/question vectors
Construct training/testing data for cross-encoder
"""
import json
import csv
from functools import reduce
from torch.utils.data import DataLoader, Dataset
# from transformers import AutoTokenizer
import torch
# import faiss
import numpy
import pandas as pd

from tqdm import tqdm

# from biencoder import BiEncoderModule
from faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer

import os
import logging
import sys

BLANK_TOKEN = '[BLANK]'


def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def write_json(json_path, data):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


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
    def __init__(self, relations, maxlen, tokenizer=None, bert_model='/home/home2/xwu/bert-base-uncased'):
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


def save_relations_vectors(relations_path, model_path, save_path, add_special_tokens=False, max_len=32, batch_size=128):
    maxlen = max_len
    bs = batch_size
    bert_model = "/home3/xwu/bertModels/bert-base-uncased"

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


def save_questions_vectors(questions_path, entity_linking_file, model_path, save_path, add_special_tokens=False, mask_mention=False):
    maxlen = 32 # 28 for LC, 32 for CWQ, 80 for WebQSP with richRelation
    bs = 128
    bert_model = "/home3/xwu/bertModels/bert-base-uncased"
    print(questions_path)
    entity_linking_res = read_json(entity_linking_file)

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
    
    with open(questions_path, 'r') as f:
        data = json.load(f)
    questions = []
    for item in data:
        question = item["question"]
        qid = item["ID"]
        if mask_mention:
            question = question.lower()
            el_result = entity_linking_res[qid] if qid in entity_linking_res else []
            for eid in el_result:
                mention = el_result[eid]["mention"]
                question = question.replace(mention, BLANK_TOKEN)
        questions.append(question)
        # questions.append(item["corrected_question"]) # LC_1
        # questions.append(item["ProcessedQuestion"]) # WebQSP
    
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



def CWQ_create_CrossEncoder_sample_tsv(
    all_relations_path, 
    dataset_file, 
    question_vectors_path, 
    index_file, 
    output_path,
    CWQid_index_path,
    entity_linking_path,
    validation_relations_path=None,
    relation_rich_map_path=None,
    source_rich_relation=False,
    target_rich_relation=True,
    vector_size=768, 
    index_buffer=50000, 
    top_k=200,
    validation_top_k=100,
    mask_mention=False,
    rich_entity=True
):
    """
    validation_relations_path: 1-hop/2hop relations from entity, for validation
    validation_top_k: After validation, reserve top-k relations
    """
    all_relations = read_json(all_relations_path)
    entity_linking_res = read_json(entity_linking_path)
    if relation_rich_map_path:
        relation_rich_map = read_json(relation_rich_map_path)
        
    if validation_relations_path:
        validation_relations_map = read_json(validation_relations_path)
    
    index = DenseFlatIndexer(vector_size, index_buffer)
    index.deserialize_from(index_file) 
    question_vectors = torch.load(question_vectors_path).cpu().detach().numpy()
    _, pred_relation_indexes = index.search_knn(question_vectors, top_k=top_k)
    pred_relations = pred_relation_indexes.tolist()
    pred_relations = [
        list(set([all_relations[index] for index in indexes])) 
        for indexes in pred_relation_indexes
    ]
    
    input_data = read_json(dataset_file)
    
    samples = []
    count = 0
    CWQid_index = dict()
    for idx in range(len(input_data)):
        start = count
        question = input_data[idx]["question"]
        id = input_data[idx]["ID"]
        if mask_mention:
            question = question.lower()
            el_result = entity_linking_res[id] if id in entity_linking_res else []
            for eid in el_result:
                mention = el_result[eid]["mention"]
                question = question.replace(mention, BLANK_TOKEN)
        if rich_entity:
            question = question.lower()
            for richEntity in input_data[idx]["candidate_rich_entities"]:
                question = question.replace(richEntity.split('|')[0], richEntity)
        if source_rich_relation and relation_rich_map:
            golden_relations = list(set([relation_rich_map[relation] if relation in relation_rich_map else relation for relation in input_data[idx]["gold_relation_map"].keys()]))
        else:
            golden_relations = list(set([relation for relation in input_data[idx]["gold_relation_map"].keys()]))
        
        if validation_relations_path:
            validation_relations = reduce(
                lambda x,y: x+validation_relations_map[y["id"]] if y["id"] in validation_relations_map else x,
                input_data[idx]["disambiguated_cand_entity"],
                []
            )

        if validation_relations_path and relation_rich_map and source_rich_relation:
            validation_relations = [relation_rich_map[rel] if rel in relation_rich_map else rel for rel in validation_relations]

        if validation_relations_path:
            missed_relations = []
            for pred_relation in pred_relations[idx]:
                if pred_relation in golden_relations:
                    pred_relation = relation_rich_map[pred_relation] if (pred_relation in relation_rich_map and target_rich_relation) else pred_relation
                    samples.append([question, pred_relation, '1'])
                    count += 1
                elif pred_relation in validation_relations:
                    pred_relation = relation_rich_map[pred_relation] if (pred_relation in relation_rich_map and target_rich_relation) else pred_relation
                    samples.append([question, pred_relation, '0'])
                    count += 1
                else:
                    missed_relations.append(pred_relation)
            if count - start < validation_top_k:
                for rel in missed_relations[:validation_top_k-(count - start)]:
                    rel = relation_rich_map[rel] if (rel in relation_rich_map and target_rich_relation) else rel
                    samples.append([question, rel, '0'])
                    count += 1
        else:
            for pred_relation in pred_relations[idx]:
                if pred_relation in golden_relations:
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


    with open(output_path, 'w') as f:
        header = ['id', 'question', 'relation', 'label']
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        idx = 0
        for line in samples:
            writer.writerow([str(idx)] + line)
            idx += 1
    
    write_json(CWQid_index_path, CWQid_index)


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
    main()