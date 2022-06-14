#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :    data_process_final.py
@Time    :    2022/06/12 14:26:37
@Author  :    Xixin Hu
@Version :    1.0
@Contact :    xixinhu97@foxmail.com
@Desc    :    
'''

from collections import defaultdict
import random

from sklearn import datasets
from components.utils import load_json, dump_json
import argparse
from tqdm import tqdm
import os
import torch
import pandas as pd


def _parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('action',type=str,help='Action to operate')
    parser.add_argument('--dataset', required=True, default='CWQ', help='dataset to perform entity linking, should be CWQ or WebQSP')
    parser.add_argument('--split', required=True, default='test', help='split to operate on') # the split file: ['dev','test','train']
    
    

    return parser.parse_args()


def combine_entities_from_FACC1_and_elq(dataset, split, sample_size=10):
    """ Combine the linking results from FACC1 and ELQ """
    entity_dir = f'data/{dataset}/entity_retrieval/candidate_entities'

    facc1_disamb_res = load_json(f'{entity_dir}/{dataset}_{split}_cand_entities_facc1.json')
    elq_res = load_json(f'{entity_dir}/{dataset}_{split}_cand_entities_elq.json')

    print('lens: {}'.format(len(elq_res.keys())))

    combined_res = dict()

    train_entities_elq = {}
    elq_res_train = load_json(f'{entity_dir}/{dataset}_train_cand_entities_elq.json')
    for qid,cand_ents in elq_res_train.items():
        for ent in cand_ents:
            train_entities_elq[ent['id']] = ent['label']
        
    
    train_entities_elq = [{"id":mid,"label":label} for mid,label in train_entities_elq.items()]


    for qid in tqdm(elq_res,total=len(elq_res),desc=f'Merging candidate entities of {split}'):
        cur = dict() # unique by mid

        elq_result = elq_res[qid]
        facc1_result = facc1_disamb_res.get(qid,[])
        # facc1_result = [item for mention_res in facc1_disamb_res[qid] for item in mention_res] if qid in facc1_disamb_res else []
        
        # sort by score
        elq_result = sorted(elq_result, key=lambda d: d.get('score', -20.0), reverse=True)
        facc1_result = sorted(facc1_result, key=lambda d: d.get('logit', 0.0), reverse=True)

        # merge the linking results of ELQ and FACC1 one by one
        idx = 0
        while len(cur.keys()) < sample_size:
            if idx < len(elq_result):
                cur[elq_result[idx]["id"]] = elq_result[idx]
            if len(cur.keys()) < sample_size and idx < len(facc1_result):
                cur[facc1_result[idx]["id"]] = facc1_result[idx]
            if idx >= len(elq_result) and idx >= len(facc1_result):
                break
            idx += 1

        if len(cur.keys()) < sample_size:
            # sample some entities to reach the sample size
            diff_entities = list(filter(lambda v: v["id"] not in cur.keys(), train_entities_elq))
            random_entities = random.sample(diff_entities, 10 - len(cur.keys()))
            for ent in random_entities:
                cur[ent["id"]] = ent

        assert len(cur.keys()) == sample_size, print(qid)
        combined_res[qid] = list(cur.values())

    
    merged_file_path = f'{entity_dir}/{dataset}_{split}_merged_cand_entities_elq_facc1.json'
    print(f'Writing merged candidate entities to {merged_file_path}')
    dump_json(combined_res, merged_file_path, indent=4)


def make_sorted_relation_dataset_from_logits(dataset, split):

    assert dataset in ['CWQ','WebQSP']
    if dataset == 'WebQSP':
        assert split in ['test','train']
    else:
        assert split in ['test','train','dev']

    output_dir = f'data/{dataset}/relation_retrieval/candidate_relations'
    logits_file = f'data/{dataset}/relation_retrieval/cross-encoder/saved_models/final/{split}/logits.pt'
    
    if dataset=='CWQ':
        tsv_file = f'data/CWQ/relation_retrieval/cross-encoder/CWQ.{split}.biEncoder.train_all.maskMention.crossEncoder.2hopValidation.maskMention.richRelation.top100.tsv'
    elif dataset=='WebQSP':
        tsv_file = f'data/WebQSP/relation_retrieval/cross-encoder/WebQSP.{split}.biEncoder.train_all.richRelation.crossEncoder.train_all.richRelation.2hopValidation.richEntity.top100.1parse.tsv'


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logits = torch.load(logits_file,map_location=torch.device('cpu'))

    # print(logits)
    # print(len(logits))
    logits_list = list(logits.squeeze().numpy())
    print('Logits len:',len(logits_list))

    tsv_df = pd.read_csv(tsv_file, delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
                            #quoting=csv.QUOTE_NONE,
                            

    print('Tsv len:', len(tsv_df))
    # print(tsv_df.head())
    print('Question Num:',len(tsv_df['question'].unique()))

    # the length of predicted logits must match the num of input examples
    assert(len(logits_list)==len(tsv_df))

    
    # if dataset.lower()=='webqsp':
    #     split_dataset = load_json(f'data/WebQSP/origin/WebQSP.{split}.json')
    # else:
    #     split_dataset = load_json(f'data/CWQ/sexpr/ComplexWebQuestions_{split}.json')

    split_dataset = load_json(f'data/{dataset}/sexpr/{dataset}.{split}.expr.json')
    # question2id = {x['question']:x['ID'] for x in split_dataset}


    rowid2qid = {} # map rowid to qid
    ''

    if dataset=='CWQ':
        idmap = load_json(f'data/CWQ/relation_retrieval/cross-encoder/CWQ.{split}.biEncoder.train_all.maskMention.crossEncoder.2hopValidation.maskMention.richRelation.top100_CWQid_index_map.json')
    elif dataset=='WebQSP':
        idmap = load_json(f'data/WebQSP/relation_retrieval/cross-encoder/WebQSP.{split}.biEncoder.train_all.richRelation.crossEncoder.train_all.richRelation.2hopValidation.richEntity.top100.1parse_WebQSPid_index_map.json')

    for qid in idmap:
        rowid_start = idmap[qid]['start']
        rowid_end = idmap[qid]['end']
        #rowid2qid[rowid]=qid
        for i in range(rowid_start,rowid_end+1):
           rowid2qid[i]=qid


    # cand_rel_bank = {} # Dict[Question, Dict[Relation:logit]]
    cand_rel_bank = defaultdict(dict)
    for idx,logit in tqdm(enumerate(logits_list),total=len(logits_list),desc=f'Reading logits of {split}'):
        logit = float(logit[1])
        row_id = tsv_df.loc[idx]['id']
        question = tsv_df.loc[idx]['question']
        rel = tsv_df.loc[idx]['relation'].split("|")[0]
        #cwq_id = question2id.get(question,None)
        qid = rowid2qid[row_id]

        if not qid:
            print(question)
            cand_rel_bank[qid]= {}
        else:
            cand_rel_bank[qid][rel]=logit

    cand_rel_logit_map = {}
    for qid in tqdm(cand_rel_bank,total=len(cand_rel_bank),desc='Sorting rels...'):
        cand_rel_maps = cand_rel_bank[qid]
        cand_rel_list = [(rel,logit) for rel,logit in cand_rel_maps.items()]
        cand_rel_list.sort(key=lambda x:x[1],reverse=True)
        
        cand_rel_logit_map[qid]=cand_rel_list

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dump_json(cand_rel_logit_map,os.path.join(output_dir,f'{dataset}_{split}_cand_rel_logits.json'),indent=4)

    final_candRel_map = defaultdict(list) # Dict[Question,List[Rel]]   sorted by logits

    for ori_data in tqdm(split_dataset,total=len(split_dataset),desc=f'{split} Dumping... '):
        if dataset=='CWQ':
            qid = ori_data['ID']
        else:
            qid = ori_data['QuestionId']
        # cand_rel_map = cand_rel_bank.get(qid,None)
        cand_rel_list = cand_rel_logit_map.get(qid,None)
        if not cand_rel_list:
            final_candRel_map[qid]=[]
        else:
            # cand_rel_list = list(cand_rel_map.keys())
            # cand_rel_list.sort(key=lambda x:float(cand_rel_map[x]),reverse=True)
            final_candRel_map[qid]=[x[0] for x in cand_rel_list]

    sorted_cand_rel_name = os.path.join(output_dir,f'{dataset}_{split}_cand_rels_sorted.json')
    dump_json(final_candRel_map,sorted_cand_rel_name,indent=4)   




if __name__=='__main__':
    
    
    args = _parse_args()
    action = args.action

    if action.lower()=='merge_entity':
        combine_entities_from_FACC1_and_elq(dataset=args.dataset, split=args.split)
    elif action.lower()=='merge_relation':
        make_sorted_relation_dataset_from_logits(dataset=args.dataset, split=args.split)
    elif action.lower()=='merge_all':
        pass
    else:
        print('usage: data_process.py action [--dataset DATASET] --split SPLIT ')
