import os
import torch
import numpy as np

import json
import pandas as pd
from collections import defaultdict
"""
Evaluate the results of cross-encoder.
Combine the result of cross-encoder, to construct source data file of GMT-KBQA
"""

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def write_json(json_path, data):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def make_logits_result(logits_folder, tsv_path, id_index_map_path, loss_type="CE"):
    """
    Get retrieved relations based on logits.
    """
    logits = torch.load(os.path.join(logits_folder, 'logits.pt'), map_location=torch.device('cpu'))
    rich_relation_path = '../../Data/common_data/fb_rich_relation_map.json'
    rich_relation_map = read_json(rich_relation_path)
    if loss_type == "CE":
        logits = logits[:, 1] # probability of '1'
    print('logits: {}'.format(logits.shape))
    logits_list = list(logits.squeeze().numpy())
    print('Logits len:',len(logits_list))

    split_df = pd.read_csv(tsv_path, sep='\t'
                            ,error_bad_lines=False).dropna()
                
    print('Dataframe len:',len(split_df))
    print('0: {}'.format(split_df.loc[0]))

    # length check
    assert len(logits_list) == len(split_df)

    id_index_map = read_json(id_index_map_path)

    res_dict = defaultdict(list)
    
    print('len logits: {}'.format(len(logits_list)))

    for qid in id_index_map:
        print(qid)
        start = int(id_index_map[qid]["start"])
        end = int(id_index_map[qid]["end"])  
            
        for idx in range(start, end+1):
            rel = split_df.loc[idx]['relation']
            assert len(rich_relation_map[rel]) == 1
            rel = rich_relation_map[rel][0]
            res_dict[qid].append([
                rel,
                str(logits_list[idx])
            ])
        
        sorted_logits = dict()
        for id in res_dict:
            logits = res_dict[id]
            logits.sort(key=lambda item: float(item[1]), reverse=True)
            sorted_logits[id] = logits
    
    write_json(os.path.join(logits_folder, 'logits_sorted.json'), sorted_logits)


def check_sorted_dataset_coverage(logits_folder, split='test'):
    sorted_rel_file = os.path.join(logits_folder, 'logits_sorted.json')
    sorted_rel_bank = read_json(sorted_rel_file)
    golden_file = '../../Data/CWQ/generation/merged/CWQ_{}.json'.format(split)
    golden = read_json(golden_file)
    assert len(golden) == len(sorted_rel_bank)
    golden_bank = dict()
    for item in golden:
        golden_bank[item["ID"]] = item["gold_relation_map"].keys()
    top1_pre = 0
    top1_recall = 0
    top3_pre = 0
    top3_recall = 0
    top5_pre = 0
    top5_recall = 0
    top10_pre = 0
    top10_recall = 0
    gt0_pre = 0
    gt0_recall = 0
    total = len(sorted_rel_bank)
    for qid in sorted_rel_bank:
        cand_rel_list = sorted_rel_bank[qid]
        gt0_cand = list(filter(
            lambda item: float(item[1]) > 0.0,
            cand_rel_list
        ))
        gt0_cand = set(list(map(
            lambda item: item[0],
            gt0_cand
        )))
        cand_rel_list = list(map(
            lambda item: item[0],
            cand_rel_list
        ))
        gold_rel_set = set(golden_bank[qid])

        top1_cand = set(cand_rel_list[:1])
        top3_cand = set(cand_rel_list[:3])
        top5_cand = set(cand_rel_list[:5])
        top10_cand = set(cand_rel_list[:10])
       

        top1_pre += len(top1_cand & gold_rel_set) / len(top1_cand) if top1_cand else 0
        top1_recall += len(top1_cand & gold_rel_set) / len(gold_rel_set) if gold_rel_set else 0
        
        top3_pre += len(top3_cand & gold_rel_set) / len(top3_cand) if top3_cand else 0
        top3_recall += len(top3_cand & gold_rel_set) / len(gold_rel_set) if gold_rel_set else 0

        top5_pre += len(top5_cand & gold_rel_set) / len(top5_cand) if top5_cand else 0
        top5_recall += len(top5_cand & gold_rel_set) / len(gold_rel_set) if gold_rel_set else 0

        top10_pre += len(top10_cand & gold_rel_set) / len(top10_cand) if top10_cand else 0
        top10_recall += len(top10_cand & gold_rel_set) / len(gold_rel_set) if gold_rel_set else 0
        
        gt0_pre += len(gt0_cand & gold_rel_set) / len(gt0_cand) if gt0_cand else 0
        gt0_recall += len(gt0_cand & gold_rel_set) / len(gold_rel_set) if gold_rel_set else 0
    
    print(split)
    print(f'GT 0 pre: {gt0_pre/total}, GT 0 recall: {gt0_recall/total}') 
    print(f'TOP 1 pre: {top1_pre/total}, TOP 1 recall: {top1_recall/total}') 
    print(f'TOP 3 pre: {top3_pre/total}, TOP 3 recall: {top3_recall/total}') 
    print(f'TOP 5 pre: {top5_pre/total}, TOP 5 recall: {top5_recall/total}') 
    print(f'TOP 10 pre: {top10_pre/total}, TOP 10 recall: {top10_recall/total}') 


def _normalize_relation(r):
    r = r.replace('_', ' ')
    r = r.replace('.', ' , ')
    return r


def add_cand_relations_to_dataset(logits_folder, split, output_path, topK=10):
    """
    Add retrieved relations to source dataset for GMT-KBQA
    """
    dataset_path = '../../Data/CWQ/generation/merged/CWQ_{}.json'.format(split)
    dataset = read_json(dataset_path)
    sorted_logits_path = os.path.join(logits_folder, 'logits_sorted.json')
    sorted_logits = read_json(sorted_logits_path)
    new_dataset = []
    for item in dataset:
        item.pop('cand_relation_list', None)
        id = item["ID"]
        relation_list = sorted_logits[id][:topK]
        for rel in relation_list:
            rel.append(_normalize_relation(rel[0]))
        item["cand_relation_list"] = relation_list
        new_dataset.append(item)
    assert len(new_dataset) == len(dataset)

    write_json(output_path, new_dataset)

def main():
    """
    Get retrieved relations based on logits.
    """
    # make_logits_result(
    #     '../../Data/CWQ/relation_retrieval/cross-encoder/saved_models/final',
    #     '../../Data/CWQ/relation_retrieval/cross-encoder/CWQ.test.biEncoder.train_all.maskMention.crossEncoder.2hopValidation.maskMention.richRelation.top100.tsv',
    #     '../../Data/CWQ/relation_retrieval/cross-encoder/CWQ.test.biEncoder.train_all.maskMention.crossEncoder.2hopValidation.maskMention.richRelation.top100_CWQid_index_map.json',
    # )

    """
    Evaluation of P,R,F1 on retrieved relations under different settings
    """
    # check_sorted_dataset_coverage(
    #     '../../Data/CWQ/relation_retrieval/cross-encoder/saved_models/final',
    #     'test'
    # )

    """
    Add retrieved relations to dataset of GMT-KBQA
    """
    add_cand_relations_to_dataset(
        '../../Data/CWQ/relation_retrieval/cross-encoder/saved_models/final',
        'test',
        'CWQ_test.json'
    )

if __name__=='__main__':
    main()