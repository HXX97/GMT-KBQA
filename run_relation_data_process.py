import argparse
from collections import defaultdict
import collections
import os
import random
import csv
from executor.sparql_executor import get_2hop_relations_with_odbc, get_2hop_relations_with_odbc_wo_filter
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer

from components.utils import (
    extract_mentioned_relations_from_sparql, 
    load_json, 
    dump_json,
    _textualize_relation
) 

BLANK_TOKEN = '[BLANK]'

def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('action',type=str,help='Action to operate')
    parser.add_argument('--dataset', default='CWQ', help='dataset to perform entity linking, should be CWQ or WebQSP')
    parser.add_argument('--split', default='test', help="split to operate on; ['train', 'dev', 'test']") # the split file: ['dev','test','train']

    return parser.parse_args()


def extract_golden_relations_cwq(src_path, tgt_path):
    """
    Extracting golden relations from sparql.
    """
    dataset_with_sexpr = load_json(src_path)
    merged_data = []
    for example in tqdm(dataset_with_sexpr, total=len(dataset_with_sexpr), desc=f'Extracting golden relations'):
        sparql = example['sparql']
        gold_relations = extract_mentioned_relations_from_sparql(sparql)
        gold_rel_label_map = {}
        for rel in gold_relations:
            linear_rel = _textualize_relation(rel)
            gold_rel_label_map[rel] = linear_rel
        example['gold_relation_map'] = gold_rel_label_map
        merged_data.append(example)

    print(f'Wrinting merged data to {tgt_path}...')
    dump_json(merged_data,tgt_path,indent=4)
    print('Writing finished')

def extract_golden_relations_webqsp(src_path, tgt_path):
    origin_dataset = load_json(src_path)
    if "Questions" in origin_dataset:
        origin_dataset = origin_dataset["Questions"]
    merged_data = []
    for example in tqdm(origin_dataset, total=len(origin_dataset), desc=f'Extracting golden relations'):
        gold_rel_label_map = {}
        for parse in example['Parses']:
            sparql = parse['Sparql']
            gold_relations = extract_mentioned_relations_from_sparql(sparql)
            
            for rel in gold_relations:
                linear_rel = _textualize_relation(rel)
                gold_rel_label_map[rel] = linear_rel # dictionary 本身有去重功能

        example['gold_relation_map'] = gold_rel_label_map
        merged_data.append(example)

    print(f'Wrinting merged data to {tgt_path}...')
    dump_json(merged_data,tgt_path,indent=4)
    print('Writing finished')

    



def sample_data_mask_entity_mention(
    golden_file, 
    entity_linking_file, 
    all_relations_file, 
    output_path, 
    sample_size=100
):
    """
    This method mask entity mentions in question accroding to entity linking results
    """
    print('output_path: {}'.format(output_path))
    golden_NLQ_relations = dict()
    all_relations = load_json(all_relations_file)
    entity_linking_res = load_json(entity_linking_file)
    items = load_json(golden_file)
    # TODO: 应该用 qid 作为标识符
    for item in items:
        # mask entity mention in question
        question = item["question"].lower()
        qid = item["ID"]
        el_result = entity_linking_res[qid] if qid in entity_linking_res else []
        for eid in el_result:
            mention = el_result[eid]["mention"]
            question = question.replace(mention, BLANK_TOKEN)
        golden_NLQ_relations[question] = list(item["gold_relation_map"].keys())

    samples = []
    for question in tqdm(golden_NLQ_relations, total=len(golden_NLQ_relations), desc="Sampling data for Bi-encoder"): 
        relations = golden_NLQ_relations[question]
        diff_rels = list(set(all_relations) - set(relations))
        
        negative_rels = random.sample(diff_rels, (sample_size-1) * len(relations))
        # Make sure each batch contains 1 golden relation
        for idx in range(len(relations)):
            sample = []
            sample.append([question, relations[idx], '1'])
            for n_rel in negative_rels[idx * (sample_size-1): (idx+1) * (sample_size-1)]:
                sample.append([question, n_rel, '0'])
            random.shuffle(sample)
            samples.extend(sample)
        
    with open(output_path, 'w') as f:
        header = ['id', 'question', 'relation', 'label']
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        idx = 0
        for line in samples:
            writer.writerow([str(idx)] + line)
            idx += 1

def sample_data_rich_relation(
    golden_file, 
    relations_file, 
    relation_rich_map_path,
    output_path, 
    sample_size=100
):
    """
    sample_data() create training/test dataset based relations
    sample_data_rich_relation() create training/test dataset based on enriched relations, i.e., relation|label|domain|range
    """
    golden_rich_relations_map = dict()
    relation_rich_map = load_json(relation_rich_map_path)
    all_relations = load_json(relations_file)
    all_rich_relations = list(set(map(
        lambda item: relation_rich_map[item] if item in relation_rich_map else item, 
        all_relations
    )))
    examples = load_json(golden_file)

    for example in examples:
        golden_rich_relations_map[example["QuestionId"]] = list(set(map(
            lambda item: relation_rich_map[item] if item in relation_rich_map else item, list(example["gold_relation_map"].keys())
        )))
    
    qid_example_map = {item["QuestionId"]: item for item in examples}

    samples = []
    for qid in tqdm(golden_rich_relations_map, total=len(golden_rich_relations_map)): 
        question = qid_example_map[qid]["ProcessedQuestion"].lower()
        golden_rich_relations = golden_rich_relations_map[qid]
        diff_rich_relations = list(set(all_rich_relations) - set(golden_rich_relations))
        
        negative_rich = random.sample(diff_rich_relations, (sample_size-1) * len(golden_rich_relations))
        # Make sure each batch contains 1 golden relation
        for idx in range(len(golden_rich_relations)):
            sample = []
            sample.append([question, golden_rich_relations[idx], '1'])
            for n_lab in negative_rich[idx * (sample_size-1): (idx+1) * (sample_size-1)]:
                sample.append([question, n_lab, '0'])
            assert len(sample) == sample_size
            random.shuffle(sample)
            samples.extend(sample)
        
    with open(output_path, 'w') as f:
        header = ['id', 'question', 'relation', 'label']
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        idx = 0
        for line in samples:
            writer.writerow([str(idx)] + line)
            idx += 1

def sample_data_normal(
    golden_data_path,
    all_relations_path,
    output_path, 
    sample_size=100
):
    all_relations = load_json(all_relations_path)
    examples = load_json(golden_data_path)

    # 应该用 qid 作为标识符
    golden_relations_map = {
        example["QuestionId"]: list(example["gold_relation_map"].keys())
        for example in examples
    }

    qid_example_map = {item["QuestionId"]: item for item in examples}

    samples = []
    for qid in tqdm(golden_relations_map, total=len(golden_relations_map)):
        question = qid_example_map[qid]["ProcessedQuestion"].lower()
        gold_relations = golden_relations_map[qid]
        diff_relations = list(set(all_relations) - set(gold_relations))
        negative_relations = random.sample(diff_relations, (sample_size-1) * len(gold_relations))
        for idx in range(len(gold_relations)):
            sample = []
            sample.append([question, gold_relations[idx], '1'])
            for n_lab in negative_relations[idx * (sample_size-1): (idx+1) * (sample_size-1)]:
                sample.append([question, n_lab, '0'])
            assert len(sample) == sample_size
            random.shuffle(sample)
            samples.extend(sample)

    with open(output_path, 'w') as f:
        header = ['id', 'question', 'relation', 'label']
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        idx = 0
        for line in samples:
            writer.writerow([str(idx)] + line)
            idx += 1

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

def validate_data_sequence(
    data_path_1,
    data_path_2,
):
    data_1 = load_json(data_path_1)
    data_2 = load_json(data_path_2)
    if "Questions" in data_1:
        data_1 = data_1["Questions"]
    if "Questions" in data_2:
        data_2 = data_2["Questions"]
    data_1_qids = [example["QuestionId"] for example in data_1]
    data_2_qids = [example["QuestionId"] for example in data_2]
    print(data_1_qids == data_2_qids)


def get_bi_encoder_tsv_max_len(tsv_file):
    tsv_df = pd.read_csv(tsv_file, delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
    tokenizer = AutoTokenizer.from_pretrained('hfcache/bert-base-uncased')
    length_dict = defaultdict(int)
    for idx in tqdm(range(len(tsv_df)), total=len(tsv_df)):
        question = tsv_df.loc[idx, 'question']
        relation = tsv_df.loc[idx, 'relation']
        tokenized_question = tokenizer.tokenize(question)
        length_dict[len(tokenized_question)] += 1
        tokenized_relation = tokenizer.tokenize(relation)
        length_dict[len(tokenized_relation)] += 1
    print(collections.OrderedDict(sorted(length_dict.items())))

def sample_data(dataset, split):
    if dataset.lower() == 'cwq':
        extract_golden_relations_cwq(
            'data/CWQ/sexpr/CWQ.{}.expr.json'.format(split),
            'data/CWQ/relation_retrieval/bi-encoder/CWQ.{}.relation.json'.format(split)
        )
        if split != 'test':
            sample_data_mask_entity_mention(
                'data/CWQ/relation_retrieval/bi-encoder/CWQ.{}.relation.json'.format(split),
                'data/CWQ/entity_retrieval/merged_linking_results/merged_CWQ_{}_linking_results.json'.format(split),
                'data/common_data/freebase_relations_filtered.json',
                'data/CWQ/relation_retrieval/bi-encoder/CWQ.{}.sampled.tsv'.format(split)
            )
    elif dataset.lower() == 'webqsp':
        if not os.path.exists('data/WebQSP/origin/WebQSP.pdev.json'):
            print('Dividing ptrain and pdev')
            make_partial_train_dev('data/WebQSP/origin/WebQSP.train.json')
        if not os.path.exists('data/WebQSP/relation_retrieval_final/bi-encoder/WebQSP.{}.goldenRelation.json'.format(split)):
            for sp in ['train', 'ptrain', 'pdev', 'test']:
                print('extract golden relations')
                extract_golden_relations_webqsp(
                    'data/WebQSP/origin/WebQSP.{}.json'.format(sp),
                    'data/WebQSP/relation_retrieval_final/bi-encoder/WebQSP.{}.goldenRelation.json'.format(sp)
                )
                validate_data_sequence(
                    f'data/WebQSP/origin/WebQSP.{sp}.json',
                    f'data/WebQSP/relation_retrieval_final/bi-encoder/WebQSP.{sp}.goldenRelation.json',
                )
                
        if split != 'test':
            sample_data_rich_relation(
                'data/WebQSP/relation_retrieval_final/bi-encoder/WebQSP.{}.goldenRelation.json'.format(split),
                'data/common_data/freebase_relations_filtered.json',
                'data/common_data/fb_relation_rich_map.json',
                'data/WebQSP/relation_retrieval_final/bi-encoder/WebQSP.{}.sampled.tsv'.format(split)
            )
        # if split != 'test':
        #     sample_data_normal(
        #         'data/WebQSP/relation_retrieval_0722/bi-encoder/WebQSP.{}.goldenRelation.json'.format(split),
        #         'data/common_data/freebase_relations_filtered.json',
        #         'data/WebQSP/relation_retrieval_0722/bi-encoder/WebQSP.{}.sampled.tsv'.format(split)
        #     )

        # prepare_2hop_relations(dataset)

        # Utility function, help to set max_len for training bi-encoder
        # Here the max length suggested is 60
        # get_bi_encoder_tsv_max_len('data/WebQSP/relation_retrieval_final/bi-encoder/WebQSP.train.sampled.tsv')


def get_unique_entity_ids(
    train_split_path,
    dev_split_path,
    test_split_path,
    output_path
):
    """
    得到所有实体链接结果中的 entity ids, 便于下一步查询每个实体的二跳关系
    """
    if train_split_path is not None:
        train_data = load_json(train_split_path)
    else:
        train_data = None
    
    if dev_split_path is not None:
        dev_data = load_json(dev_split_path)
    else:
        dev_data = None
    
    if test_split_path is not None:
        test_data = load_json(test_split_path)
    else:
        test_data = None
    
    unique_entity_ids = set()
    for data in [train_data, dev_data, test_data]:
        if data is None: 
            continue
        for example in data:
            for entity_id in example["freebase_ids"]:
                unique_entity_ids.add(entity_id)
    dump_json(list(unique_entity_ids), output_path)

def query_2hop_relations(entity_ids_path, output_path):
    entity_ids = load_json(entity_ids_path)
    res = dict()
    for eid in tqdm(entity_ids, total=len(entity_ids), desc="querying 2 hop relations"):
        in_relations, out_relations, _ = get_2hop_relations_with_odbc_wo_filter(eid)
        # in_relations, out_relations, _ = get_2hop_relations_with_odbc(eid)
        relations = list(set(in_relations) | set(out_relations))
        res[eid] = relations
    dump_json(res, output_path)

def construct_question_2hop_relations(
    train_split_path,
    dev_split_path,
    test_split_path,
    entities_2hop_relations_path
):
    if train_split_path is not None:
        train_data = load_json(train_split_path)
    else:
        train_data = None
    
    if dev_split_path is not None:
        dev_data = load_json(dev_split_path)
    else:
        dev_data = None
    
    if test_split_path is not None:
        test_data = load_json(test_split_path)
    else:
        test_data = None
    
    entity_relation_map = load_json(entities_2hop_relations_path)
    for item in [(train_data, train_split_path), (dev_data, dev_split_path), (test_data, test_split_path)]:
        data, path = item
        if data is None: 
            continue
        enhanced_linking_results = []
        for example in tqdm(data, total=len(data)):
            two_hop_relations = []
            for entity_id in example["freebase_ids"]:
                if entity_id in entity_relation_map:
                    two_hop_relations.extend(entity_relation_map[entity_id])
            example["two_hop_relations"] = list(set(two_hop_relations)) # 去重
            enhanced_linking_results.append(example)
        print('split: {}, length: {}'.format(path, len(enhanced_linking_results)))
        dump_json(enhanced_linking_results, path[:-5] + '_two_hop_relations.json')




def prepare_2hop_relations(
    dataset
):
    if dataset.lower() == 'webqsp':
        if not os.path.exists('data/WebQSP/relation_retrieval_final/cross-encoder/rng_kbqa_linking_results/unique_entity_ids.json'):
            print('generating unique entity')
            get_unique_entity_ids(
                'data/WebQSP/relation_retrieval_final/cross-encoder/rng_kbqa_linking_results/webqsp_train_rng_el.json',
                None,
                'data/WebQSP/relation_retrieval_final/cross-encoder/rng_kbqa_linking_results/webqsp_test_rng_el.json',
                'data/WebQSP/relation_retrieval_final/cross-encoder/rng_kbqa_linking_results/unique_entity_ids.json'
            )
        if not os.path.exists('data/WebQSP/relation_retrieval_final/cross-encoder/rng_kbqa_linking_results/entities_2hop_relations.json'):
            print('quering 2hop relations')
            query_2hop_relations(
                'data/WebQSP/relation_retrieval_final/cross-encoder/rng_kbqa_linking_results/unique_entity_ids.json',
                'data/WebQSP/relation_retrieval_final/cross-encoder/rng_kbqa_linking_results/entities_2hop_relations.json'
            )
        if not os.path.exists('data/WebQSP/relation_retrieval_final/cross-encoder/rng_kbqa_linking_results/webqsp_test_rng_el_two_hop_relations.json'):
            print('adding two hop relations')
            # 把问题到二跳关系的映射，添加到实体链接文件中
            construct_question_2hop_relations(
                'data/WebQSP/relation_retrieval_final/cross-encoder/rng_kbqa_linking_results/webqsp_train_rng_el.json',
                None,
                'data/WebQSP/relation_retrieval_final/cross-encoder/rng_kbqa_linking_results/webqsp_test_rng_el.json',
                'data/WebQSP/relation_retrieval_final/cross-encoder/rng_kbqa_linking_results/entities_2hop_relations.json'
            )




if __name__=='__main__':
    args = _parse_args()
    action = args.action

    if action.lower() == 'sample_data':
        sample_data(dataset=args.dataset, split=args.split)
        
