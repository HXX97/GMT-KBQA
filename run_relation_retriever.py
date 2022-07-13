import argparse
from collections import defaultdict
import os
import random
import csv
from tqdm import tqdm

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
    parser.add_argument('--dataset', required=True, default='CWQ', help='dataset to perform entity linking, should be CWQ or WebQSP')
    parser.add_argument('--split', required=True, default='test', help="split to operate on; ['train', 'dev', 'test']") # the split file: ['dev','test','train']

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
    dataset_with_sexpr = load_json(src_path)
    merged_data = []
    for example in tqdm(dataset_with_sexpr, total=len(dataset_with_sexpr), desc=f'Extracting golden relations'):
        # for WebQSP choose 
        # 1. shortest sparql
        # 2. s-expression converted from this sparql should leads to same execution results.
        parses = example['Parses']
        shortest_idx = 0
        shortest_len = 9999
        for i in range(len(parses)):
            if 'SExpr_execute_right' in parses[i] and parses[i]['SExpr_execute_right']:
                if len(parses[i]['Sparql']) < shortest_len:
                    shortest_idx = i
                    shortest_len = len(parses[i]['Sparql'])

        sparql = parses[shortest_idx]['Sparql']
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
    golden_NLQ_richs = dict()
    relation_rich_map = load_json(relation_rich_map_path)
    all_relations = load_json(relations_file)
    all_rich = list(set(map(lambda item: relation_rich_map[item], all_relations)))
    items = load_json(golden_file)
    for item in items:
        # For LC1
        golden_NLQ_richs[item["ProcessedQuestion"].lower()] = list(set(map(
            lambda item: relation_rich_map[item] if item in relation_rich_map else item, list(item["gold_relation_map"].keys())
        )))

    samples = []
    for question in tqdm(golden_NLQ_richs, total=len(golden_NLQ_richs)): 
        rich = golden_NLQ_richs[question]
        diff_rich = list(set(all_rich) - set(rich))
        
        negative_rich = random.sample(diff_rich, (sample_size-1) * len(rich))
        # Make sure each batch contains 1 golden relation
        for idx in range(len(rich)):
            sample = []
            sample.append([question, rich[idx], '1'])
            for n_lab in negative_rich[idx * (sample_size-1): (idx+1) * (sample_size-1)]:
                sample.append([question, n_lab, '0'])
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

def sample_data(dataset, split):
    if dataset.lower() == 'cwq':
        extract_golden_relations_cwq(
            'data/CWQ/sexpr/CWQ.{}.expr.json'.format(split),
            'data/CWQ/relation_retrieval/bi-encoder/xwu_test_0713/CWQ.{}.relation.json'.format(split)
        )
        sample_data_mask_entity_mention(
            'data/CWQ/relation_retrieval/bi-encoder/xwu_test_0713/CWQ.{}.relation.json'.format(split),
            'data/CWQ/entity_retrieval/merged_linking_results/merged_CWQ_{}_linking_results.json'.format(split),
            'data/common_data/freebase_relations_filtered.json',
            'data/CWQ/relation_retrieval/bi-encoder/xwu_test_0713/CWQ.{}.sampled.tsv'.format(split)
        )
    elif dataset.lower() == 'webqsp':
        extract_golden_relations_webqsp(
            'data/WebQSP/sexpr/WebQSP.{}.expr.json'.format(split),
            'data/WebQSP/relation_retrieval/bi-encoder/xwu_test_0713/WebQSP.{}.relation.json'.format(split)
        )
        sample_data_rich_relation(
            'data/WebQSP/relation_retrieval/bi-encoder/xwu_test_0713/WebQSP.{}.relation.json'.format(split),
            'data/common_data/freebase_relations_filtered.json',
            'data/common_data/fb_relation_rich_map.json',
            'data/WebQSP/relation_retrieval/bi-encoder/xwu_test_0713/WebQSP.{}.sampled.tsv'.format(split)
        )



if __name__=='__main__':
    args = _parse_args()
    action = args.action

    if action.lower() == 'sample_data':
        sample_data(dataset=args.dataset, split=args.split)