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
from typing import Dict, List
from sklearn import datasets
from components.utils import (
    _textualize_relation,
    load_json, 
    dump_json, 
    extract_mentioned_entities_from_sparql, 
    extract_mentioned_relations_from_sparql,
    vanilla_sexpr_linearization_method
)
import argparse
from tqdm import tqdm
import os
import torch
import pandas as pd
from executor.sparql_executor import (
    execute_query_with_odbc_filter_answer,
    get_label_with_odbc,    
    get_types_with_odbc,
    get_1hop_relations_with_odbc
)


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
    rel_info_map = defaultdict(str)
    for idx,logit in tqdm(enumerate(logits_list),total=len(logits_list),desc=f'Reading logits of {split}'):
        logit = float(logit[1])
        row_id = tsv_df.loc[idx]['id']
        question = tsv_df.loc[idx]['question']
        rich_rel = tsv_df.loc[idx]['relation']
        rel = rich_rel.split("|")[0]
        rel_info = " | ".join(rich_rel.split("|")).replace("."," , ").replace("_"," ")
        #cwq_id = question2id.get(question,None)
        qid = rowid2qid[row_id]

        if not qid:
            print(question)
            cand_rel_bank[qid]= {}
        else:
            cand_rel_bank[qid][rel]=logit

        if not rel in rel_info_map:
            rel_info_map[rel] = rel_info
            

    cand_rel_logit_map = {}
    for qid in tqdm(cand_rel_bank,total=len(cand_rel_bank),desc='Sorting rels...'):
        cand_rel_maps = cand_rel_bank[qid]
        cand_rel_list = [(rel,logit,rel_info_map[rel]) for rel,logit in cand_rel_maps.items()]
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


def get_all_unique_candidate_entities(dataset)->List[str]:
    """Get unique candidate entity ids of {dataset}"""

    ent_dir = f'data/{dataset}/entity_retrieval/candidate_entities'
    unique_entities_file = f'{ent_dir}/{dataset}_candidate_entity_ids_unique.json'
    
    if os.path.exists(unique_entities_file):
        print(f'Loading unique candidate entities from {unique_entities_file}')
        unique_entities = load_json(unique_entities_file)
    else:
        print(f'Processing candidate entities...')

        train_data = load_json(f'{ent_dir}/{dataset}_train_merged_cand_entities_elq_facc1.json')
        test_data = load_json(f'{ent_dir}/{dataset}_test_merged_cand_entities_elq_facc1.json')

        if dataset=='CWQ':
            dev_data = load_json(f'{ent_dir}/{dataset}_dev_merged_cand_entities_elq_facc1.json')
        else:
            dev_data = None

        unique_entities = set()
        for qid in train_data.keys():
            for ent in train_data[qid]:
                unique_entities.add(ent["id"])
        
        
        for qid in test_data.keys():
            for ent in test_data[qid]:
                unique_entities.add(ent["id"])

        if dev_data:
            for qid in dev_data.keys():
                for ent in dev_data[qid]:
                    unique_entities.add(ent["id"])
                
        print(f'Wrinting unique candidate entities to {unique_entities_file}')
        dump_json(list(unique_entities), unique_entities_file ,indent=4)
    
    return unique_entities


def get_entities_in_out_relations(dataset,unique_candidate_entities)->Dict[str,Dict[str,List[str]]]:
    
    ent_dir = f'data/{dataset}/entity_retrieval/candidate_entities'
    in_out_rels_file = f'{ent_dir}/{dataset}_candidate_entities_in_out_relations.json'

    if os.path.exists(in_out_rels_file):
        print(f'Loading cached 1hop relations from {in_out_rels_file}')
        in_out_rels = load_json(in_out_rels_file)
    else:

        if unique_candidate_entities:
            entities = unique_candidate_entities
        else:
            unique_entities_file = f'{ent_dir}/{dataset}_candidate_entity_ids_unique.json'
            entities = load_json(unique_entities_file)            
        
        IGONORED_DOMAIN_LIST = ['type', 'common', 'kg', 'dataworld', 'freebase', 'user']
        in_out_rels = dict()
        
        for ent in tqdm(entities,total=len(entities),desc='Fetching 1hop relations of candidate entities'):            
            
            relations = get_1hop_relations_with_odbc(ent)
            relations = [x for x in relations if x.split('.')[0] not in IGONORED_DOMAIN_LIST]
            in_out_rels[ent] = {
                '1hop_relations': relations
            }
        
        print(f'Writing 1hop relations to {in_out_rels_file}')
        dump_json(in_out_rels, in_out_rels_file)

    return in_out_rels
    

def merge_all_data_for_logical_form_generation(dataset, split):

    dataset_with_sexpr = load_json(f'data/{dataset}/sexpr/{dataset}.{split}.expr.json')
    candidate_entities_map = load_json(f'data/{dataset}/entity_retrieval/candidate_entities/{dataset}_{split}_merged_cand_entities_elq_facc1.json')
    candidate_relations_map = load_json(f'data/{dataset}/relation_retrieval/candidate_relations/{dataset}_{split}_cand_rel_logits.json')

    # get unique candidate entity ids
    unique_cand_entities = get_all_unique_candidate_entities(dataset)
    # get in and out rels of candidate entities
    in_out_rels_map = get_entities_in_out_relations(dataset, unique_cand_entities)
    # disambiguated entities
    disamb_ent_map = get_merged_disambiguated_entities(dataset,split)
    

    global_ent_label_map = {}
    global_rel_label_map = {}
    global_type_label_map = {}
    
    merged_data_all = []

    for example in tqdm(dataset_with_sexpr, total=len(dataset_with_sexpr), desc=f'Processing {dataset}_{split}'):
        
        new_example = {}
        
        qid = example["ID"]
        question = example['question'] if dataset=='CWQ' else example['ProcessedQuestion']
        comp_type = example["compositionality_type"] if dataset=='CWQ' else None                
        sexpr = example['SExpr']
        sparql = example['sparql']
        
        if dataset=='CWQ':
            if split=='test':
                answer = list(execute_query_with_odbc_filter_answer(sparql))
            else:
                answer = [x['answer_id'] for x in example['answers']]
        elif dataset=='WebQSP':
            pass
        
        
        normed_sexpr = vanilla_sexpr_linearization_method(sexpr)

        gold_entities = extract_mentioned_entities_from_sparql(sparql)
        gold_relations = extract_mentioned_relations_from_sparql(sparql)

        gold_ent_label_map = {}
        gold_rel_label_map = {}
        gold_type_label_map = {}
        

        for entity in gold_entities:
            is_type = False
            entity_types = get_types_with_odbc(entity)
            if "type.type" in entity_types:
                is_type = True

            entity_label = get_label_with_odbc(entity)
            gold_ent_label_map[entity] = entity_label
            global_ent_label_map[entity] = entity_label

            if is_type:
                gold_type_label_map[entity] = entity_label
                global_type_label_map[entity] = entity_label

                
        for rel in gold_relations:
            linear_rel = _textualize_relation(rel)
            gold_rel_label_map[rel] = linear_rel
            global_rel_label_map[rel] = linear_rel
        
        
        cand_relation_list = candidate_relations_map[qid]

        cand_entities = candidate_entities_map[qid]
        cand_ent_list = []

        for ent in cand_entities:
            new_ent = {}
            new_ent['id']=ent['id']
            new_ent['label']=ent['label']
            new_ent['mention']=ent.get('mention',"")
            new_ent['1hop_relations']=in_out_rels_map[new_ent['id']]['1hop_relations']
            cand_ent_list.append(new_ent)            
        


        

        # disambiguated_cand_entity = {}
        # linked_entity_id_set = set()
        # for ent in cand_entities:
        #     mention = ent.get("mention","")
        #     if mention == "": continue # entity with no mention is randomly sampled

        #     # for each mention, retain the entity with highest score
        #     if (mention not in disambiguated_cand_entity
        #             and ent['id'] not in linked_entity_id_set):
                
        #         disambiguated_cand_entity[mention] = {
        #             "id": ent["id"],
        #             "label": ent['label'],
        #             "mention": mention
        #         }
        #         linked_entity_id_set.add(ent['id'])
        
        
        # disambiguated_cand_entity = [ent for (_,ent) in disambiguated_cand_entity.items()]

        disambiguated_cand_entity = disamb_ent_map.get(qid,[])

        
        new_example['ID']=qid
        new_example['question'] = question
        new_example['comp_type'] = comp_type
        new_example['answer'] = answer
        new_example['sparql'] = sparql
        new_example['sexpr'] = sexpr
        new_example['normed_sexpr'] = normed_sexpr
        new_example['gold_entity_map'] = gold_ent_label_map
        new_example['gold_relation_map'] = gold_rel_label_map
        new_example['gold_type_map'] = gold_type_label_map
        new_example['cand_relation_list'] = cand_relation_list[:10]
        new_example['cand_entity_list'] = cand_ent_list[:10]
        new_example['disambiguated_cand_entity'] = disambiguated_cand_entity


        merged_data_all.append(new_example)
    
    merged_data_dir = f'data/{dataset}/generation/merged'
    if not os.path.exists(merged_data_dir):
        os.makedirs(merged_data_dir)
    merged_data_file = f'{merged_data_dir}/{dataset}_{split}.json'

    print(f'Wrinting merged data to {merged_data_file}...')
    dump_json(merged_data_all,merged_data_file,indent=4)
    print('Writing finished')
                

def get_merged_disambiguated_entities(dataset, split):
    """Get disambiguated entities by entity retrievers (one entity for one mention)"""
    
    disamb_ent_dir = f"data/{dataset}/entity_retrieval/disamb_entities"
    
    disamb_ent_file = f"{disamb_ent_dir}/{dataset}_merged_{split}_disamb_entities.json"

    if os.path.exists(disamb_ent_file):
        print(f'Loading disamb entities from {disamb_ent_file}')
        disamb_ent_map = load_json(disamb_ent_file)
        return disamb_ent_map
    else:
        cand_ent_dir = f"data/{dataset}/entity_retrieval/candidate_entities"
        elq_cand_ent_file = f"{cand_ent_dir}/{dataset}_{split}_cand_entities_elq.json"
        facc1_cand_ent_file = f"{cand_ent_dir}/{dataset}_{split}_cand_entities_facc1.json"
        
        elq_cand_ents = load_json(elq_cand_ent_file)
        facc1_cand_ents = load_json(facc1_cand_ent_file)

        # entities linked and ranked by elq
        elq_disamb_ents = {}        
        for qid,cand_ents in elq_cand_ents.items():
            mention_cand_map = {}
            for ent in cand_ents:
                if ent['mention'] not in mention_cand_map:
                    mention_cand_map[ent['mention']]=ent
            
            elq_disamb_ents[qid] = [ent for (_,ent) in mention_cand_map.items()]

        # entities linked and ranked by facc1
        facc1_disamb_ents = {}
        for qid,cand_ents in facc1_cand_ents.items():
            mention_cand_map = {}
            for ent in cand_ents:
                if ent['mention'] not in mention_cand_map:
                    mention_cand_map[ent['mention']]=ent            
            facc1_disamb_ents[qid] = [ent for (_,ent) in mention_cand_map.items()]


        disamb_ent_map = {}
        
        # merge the disambed entities        
        for qid in elq_disamb_ents:
            disamb_entities = {}
            linked_entity_id_set = set()
            
            
            facc1_entities = facc1_disamb_ents[qid]
            elq_entities = elq_disamb_ents[qid]
            
            for ent in facc1_entities:
                disamb_entities[ent['mention']]={
                    "id":ent["id"],
                    "label":ent["label"],
                    "mention":ent["mention"],
                    "perfect_match":ent["perfect_match"]
                }
                linked_entity_id_set.add(ent['id'])

            for ent in elq_entities:
                if ent['id'] not in linked_entity_id_set: 
                    if ent['mention'] not in disamb_entities:
                        disamb_entities[ent['mention']]={
                            "id":ent["id"],
                            "label":ent["label"],
                            "mention":ent["mention"],
                            "perfect_match":ent["perfect_match"]
                        }
                        linked_entity_id_set.add(ent['id'])
                    else:
                        # same label, different mid                        
                        if not disamb_entities[ent['mention']]['perfect_match']:
                            # if not perfect match, pop facc1 result, retain elq result
                            disamb_entities[ent['mention']]={
                                "id":ent["id"],
                                "label":ent["label"],
                                "mention":ent["mention"],
                                "perfect_match":ent["perfect_match"]
                            }
                            linked_entity_id_set.add(ent['id'])

                
            disamb_entities = [ent for (_,ent) in disamb_entities.items()]

            disamb_ent_map[qid] = disamb_entities

        print(f'Writing disamb entities into {disamb_ent_file}')
        dump_json(disamb_ent_map, disamb_ent_file, indent=4)
        
        return disamb_ent_map
        



def extract_entity_relation_type_label_from_dataset(dataset, split):
    
    
    train_databank =load_json(f"data/{dataset}/sexpr/{dataset}.{split}.expr.json")

    global_ent_label_map = {}
    global_rel_label_map = {}
    global_type_label_map = {}

    dataset_merged_label_map = {}

    for data in tqdm(train_databank, total=len(train_databank), desc=f"Processing {split}"):
        # print(data)
        qid = data['ID']
        sparql = data['sparql']

        ent_label_map = {}
        rel_label_map = {}
        type_label_map = {}

        # extract entity labels
        gt_entities = extract_mentioned_entities_from_sparql(sparql=sparql)
        for entity in gt_entities:
            is_type = False
            entity_types = get_types_with_odbc(entity)
            if "type.type" in entity_types:
                is_type = True

            entity_label = get_label_with_odbc(entity)
            ent_label_map[entity] = entity_label
            global_ent_label_map[entity] = entity_label

            if is_type:
                type_label_map[entity] = entity_label
                global_type_label_map[entity] = entity_label

        # extract relation labels
        gt_relations = extract_mentioned_relations_from_sparql(sparql)
        for rel in gt_relations:
            linear_rel = _textualize_relation(rel)
            rel_label_map[rel] = linear_rel
            global_rel_label_map[rel] = linear_rel
        
        dataset_merged_label_map[qid] = {
            'entity_label_map':ent_label_map,
            'rel_label_map':rel_label_map,
            'type_label_map':type_label_map
        }

    dir_name = "data/label_maps"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    dump_json(dataset_merged_label_map,f'{dir_name}/CWQ_{split}_label_maps.json',indent=4)    

    dump_json(global_ent_label_map, f'{dir_name}/CWQ_{split}_entity_label_map.json',indent=4)
    dump_json(global_rel_label_map, f'{dir_name}/CWQ_{split}_relation_label_map.json',indent=4)
    dump_json(global_type_label_map, f'{dir_name}/CWQ_{split}_type_label_map.json',indent=4)

    print("done")


if __name__=='__main__':
    
    
    args = _parse_args()
    action = args.action

    if action.lower()=='merge_entity':
        combine_entities_from_FACC1_and_elq(dataset=args.dataset, split=args.split)
    elif action.lower()=='merge_relation':
        make_sorted_relation_dataset_from_logits(dataset=args.dataset, split=args.split)
    elif action.lower()=='merge_all':
        merge_all_data_for_logical_form_generation(dataset=args.dataset, split=args.split)
    else:
        print('usage: data_process.py action [--dataset DATASET] --split SPLIT ')
