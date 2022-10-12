from collections import defaultdict
import collections
import json
import csv
from functools import reduce
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import torch
import pandas as pd
import argparse

from tqdm import tqdm

from biencoder import BiEncoderModule
from faiss_indexer import DenseFlatIndexer

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

def write_json(json_path, data):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('action', type=str, help='Action to operate')
    parser.add_argument('--dataset', help='CWQ or WebQSP')
    parser.add_argument('--split', help='split to operate on') # the split file: ['dev','test','train']
    parser.add_argument('--cache_dir', default='bert-base-uncased')

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
    cache_dir='bert-base-uncased',
    add_special_tokens=False, 
    mask_mention=False,
    dataset='cwq',
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
    if dataset == 'webqsp' and 'Questions' in data:
        data = data['Questions']
    questions = []
    for item in data:
        question = item["question"] if dataset == 'cwq' else item["ProcessedQuestion"]
        qid = item["ID"] if dataset == 'cwq' else item["QuestionId"]
        if mask_mention:
            question = question.lower()
            el_result = entity_linking_res[qid] if qid in entity_linking_res else {}
            el_result = {item["id"]: item for item in el_result}
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
    vector_size=768, 
    index_buffer=50000, 
    top_k=200,
    validation_top_k=100,
):
    """
    validation_relations_path: 2-hop relations from entity, for validation
    validation_top_k: After validation, reserve top-k relations
    """
    all_relations = load_json(all_relations_path)
    entity_linking_res = load_json(entity_linking_path)
        
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
        
        golden_relations = list(set([relation for relation in input_data[idx]["gold_relation_map"].keys()]))
        
        if validation_relations_path:
            validation_relations = reduce(
                lambda x,y: x+validation_relations_map[y] if y in validation_relations_map else x,
                map(lambda item: item['id'], entity_linking_res[id]),
                []
            )

        if validation_relations_path:
            missed_relations = []
            for pred_relation in pred_relations[idx]:
                if debug:
                    print('pred_relation: {}'.format(pred_relation))
                    print('validation_relations: {}'.format(validation_relations[0]))
                if pred_relation in validation_relations:
                    if split != 'test' and pred_relation in golden_relations:
                        samples.append([question, pred_relation, '1'])
                    else:
                        samples.append([question, pred_relation, '0'])
                    count += 1
                else:
                    missed_relations.append(pred_relation)

            # Fulfilling `validation_top_k` candidate relations
            if count - start < validation_top_k:
                for rel in missed_relations[:validation_top_k-(count - start)]:
                    if split != 'test' and rel in golden_relations:
                        samples.append([question, rel, '1'])
                    else:
                        samples.append([question, rel, '0'])
                    count += 1
        else:
            for pred_relation in pred_relations[idx]:
                if split != 'test' and pred_relation in golden_relations:
                    samples.append([question, pred_relation, '1'])
                    count += 1
                else:
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
    
    dump_json(CWQid_index, CWQid_index_path)


def retrieve_cross_encoder_inference_data(
    rng_linking_result_path,
    dataset_path,
    output_path,
    output_id_index_path,
):
    """
    For each question, two hop relations of entity linking result will be used as inference data
    """
    dataset = load_json(dataset_path)
    if "Questions" in dataset:
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
    

def retrieve_candidate_relations_webqsp(
    all_relations_path, 
    dataset_file, 
    question_vectors_path, 
    index_file, 
    entity_linking_path,
    output_path,
    id_index_path,
    split,
    validation_relations_path=None,
    vector_size=768, 
    index_buffer=50000, 
    top_k=200,
    validation_top_k=100,
):
    """
    validation_relations_path: 1-hop/2-hop relations of linked entities
    validation_top_k: reserved relations amount after validation
    """
    all_relations = load_json(all_relations_path)
    entity_linking_res = load_json(entity_linking_path)
    entity_linking_res = {item["id"]: item for item in entity_linking_res}

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
    debug = True

    print('split: {}'.format(split))

    assert len(dataset) == len(pred_relations)
    for idx in tqdm(range(len(dataset)), total=len(dataset)):
        qid = dataset[idx]["QuestionId"]
        start = count
        question = dataset[idx]["ProcessedQuestion"]
        question = question.lower()

        if split != 'test':
            golden_relations = list(set([relation for relation in dataset[idx]["gold_relation_map"].keys()]))
        else:
            golden_relations = None
        
        if validation_relations_map:
            validation_relations = reduce(
                lambda x,y: x+validation_relations_map[y] if y in validation_relations_map else x,
                entity_linking_res[qid]["freebase_ids"],
                []
            )
        
        if validation_relations_map:
            missed_relations = []
            for pred_relation in pred_relations[idx]:
                if debug:
                    print(f'pred_relation: {pred_relation}')
                    print(f'validation_relation: {validation_relations[0]}')
                if pred_relation in validation_relations:
                    if split != 'test':
                        if pred_relation in golden_relations:
                            samples.append([question, pred_relation, '1'])
                        else:
                            samples.append([question, pred_relation, '0'])
                    else:
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
        
        end = count - 1
        id_index[qid] = {
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
    
    write_json(id_index_path, id_index)


if __name__=='__main__':
    args = _parse_args()
    action = args.action

    if action.lower() == 'encode_relation':
        if args.dataset.lower() == 'cwq':
            encode_relations(
                'data/common_data/freebase_relations_filtered.json',
                'data/CWQ/relation_retrieval/bi-encoder/saved_models/mask_mention/CWQ_ep_1.pt',
                'data/CWQ/relation_retrieval/bi-encoder/vectors/mask_mention/CWQ_ep_1_relations.pt',
                add_special_tokens=True, # consistent with bi-encoder training script
                max_len=32, # consistent with bi-encoder training script
                batch_size=128,
                cache_dir=args.cache_dir
            )
        elif args.dataset.lower() == 'webqsp':
            encode_relations(
                'data/common_data/freebase_richRelations_filtered.json', # depends on `normal relation` or `rich relation` biencoder was trained on
                'data/WebQSP/relation_retrieval/bi-encoder/saved_models/rich_relation_3epochs/WebQSP_ep_3.pt',
                'data/WebQSP/relation_retrieval/bi-encoder/vectors/rich_relation_3epochs/WebQSP_ep_3_relations.pt',
                add_special_tokens=False,
                max_len=60, # consistent with training scripts, e.g.`run_bi_encoder_WebQSP.sh`
                batch_size=128,
                cache_dir=args.cache_dir
            )
    if action.lower() == 'build_index':
        if args.dataset.lower() == 'cwq':
            build_index(
                'data/CWQ/relation_retrieval/bi-encoder/index/mask_mention/ep_1_flat.index',
                'data/CWQ/relation_retrieval/bi-encoder/vectors/mask_mention/CWQ_ep_1_relations.pt'
            )
        elif args.dataset.lower() == 'webqsp':
            build_index(
                'data/WebQSP/relation_retrieval/bi-encoder/index/rich_relation_3epochs/ep_3_flat.index',
                'data/WebQSP/relation_retrieval/bi-encoder/vectors/rich_relation_3epochs/WebQSP_ep_3_relations.pt'
            )
    if action.lower() == 'encode_question':
        if args.dataset.lower() == 'cwq':
            encode_questions(
                'data/CWQ/origin/ComplexWebQuestions_{}.json'.format(args.split),
                'data/CWQ/entity_retrieval/disamb_entities/CWQ_merged_{}_disamb_entities.json'.format(args.split),
                'data/CWQ/relation_retrieval/bi-encoder/saved_models/mask_mention/CWQ_ep_1.pt',
                'data/CWQ/relation_retrieval/bi-encoder/vectors/mask_mention/CWQ_{}_questions.pt'.format(args.split),
                max_len=32,  # consistent with bi-encoder training script
                cache_dir='bert-base-uncased',
                add_special_tokens=True,
                mask_mention=True,
                dataset=args.dataset.lower()
            )
        elif args.dataset.lower() == 'webqsp':
            encode_questions(
                'data/WebQSP/origin/WebQSP.{}.json'.format(args.split),
                None,
                'data/WebQSP/relation_retrieval/bi-encoder/saved_models/rich_relation_3epochs/WebQSP_ep_3.pt',
                'data/WebQSP/relation_retrieval/bi-encoder/vectors/rich_relation_3epochs/WebQSP_{}_ep3_questions.pt'.format(args.split),
                max_len=60, # consistent with training scripts, e.g.`run_bi_encoder_WebQSP.sh`
                cache_dir='bert-base-uncased',
                add_special_tokens=False,
                mask_mention=False,
                dataset=args.dataset.lower()
            )
    if action.lower() == 'retrieve_relations':
        if args.dataset.lower() == 'cwq':
            retrieve_candidate_relations_cwq(
                'data/common_data/freebase_relations_filtered.json', 
                'data/CWQ/relation_retrieval/bi-encoder/CWQ.{}.goldenRelation.json'.format(args.split),
                'data/CWQ/relation_retrieval/bi-encoder/vectors/mask_mention/CWQ_{}_questions.pt'.format(args.split),
                'data/CWQ/relation_retrieval/bi-encoder/index/mask_mention/ep_1_flat.index',
                'data/CWQ/relation_retrieval/cross-encoder/mask_mention_1epoch_question_relation/CWQ.{}.tsv'.format(args.split),
                'data/CWQ/relation_retrieval/cross-encoder/mask_mention_1epoch_question_relation/CWQ_{}_id_index_map.json'.format(args.split),
                'data/CWQ/entity_retrieval/disamb_entities/CWQ_merged_{}_disamb_entities.json'.format(args.split),
                args.split,
                validation_relations_path='data/CWQ/relation_retrieval/bi-encoder/CWQ.2hopRelations.candEntities.json',
            )
        elif args.dataset.lower() == 'webqsp':
            # Get data for cross-encoder training and validation
            if args.split in ['train', 'ptrain', 'pdev', 'test']:
                retrieve_candidate_relations_webqsp(
                    'data/common_data/freebase_relations_filtered.json', # determines what kind of sampled relations will be generated (normal/rich)
                    'data/WebQSP/relation_retrieval/bi-encoder/WebQSP.{}.goldenRelation.json'.format(args.split),
                    'data/WebQSP/relation_retrieval/bi-encoder/vectors/rich_relation_3epochs/WebQSP_{}_ep3_questions.pt'.format(args.split),
                    'data/WebQSP/relation_retrieval/bi-encoder/index/rich_relation_3epochs/ep_3_flat.index',
                    'data/WebQSP/relation_retrieval/cross-encoder/rng_kbqa_linking_results/webqsp_train_rng_el_two_hop_relations.json' if args.split in ['train', 'ptrain', 'pdev'] else 'data/WebQSP/relation_retrieval/cross-encoder/rng_kbqa_linking_results/webqsp_test_rng_el_two_hop_relations.json',
                    'data/WebQSP/relation_retrieval/cross-encoder/rich_relation_3epochs_question_relation/WebQSP.{}.tsv'.format(args.split),
                    'data/WebQSP/relation_retrieval/cross-encoder/rich_relation_3epochs_question_relation/WebQSP_{}_id_index_map.json'.format(args.split),
                    args.split,
                    validation_relations_path='data/WebQSP/relation_retrieval/cross-encoder/rng_kbqa_linking_results/WebQSP.2hopRelations.rng.elq.candEntities.json',
                )
            
            # Get data for cross-encoder inference
            if args.split in ['train_2hop', 'test_2hop']:
                subsp = 'train' if args.split == 'train_2hop' else 'test'
                retrieve_cross_encoder_inference_data(
                    f'data/WebQSP/relation_retrieval/cross-encoder/rng_kbqa_linking_results/webqsp_{subsp}_rng_el_two_hop_relations.json',
                    f'data/WebQSP/origin/WebQSP.{subsp}.json',
                    f'data/WebQSP/relation_retrieval/cross-encoder/rich_relation_3epochs_question_relation/WebQSP.{args.split}.tsv',
                    f'data/WebQSP/relation_retrieval/cross-encoder/rich_relation_3epochs_question_relation/WebQSP_{args.split}_id_index_map.json',
                )


            
