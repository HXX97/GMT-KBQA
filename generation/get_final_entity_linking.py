#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   get_final_entity_linking.py
@Time    :   2022/02/25 14:16:14
@Author  :   Xixin Hu 
@Version :   1.0
@Contact :   xixinhu97@foxmail.com
@Desc    :   Merge the entity linking results from multiple linkers
'''

# here put the import lib

from components.utils import dump_json, load_json, extract_mentioned_entities_from_sparql, extract_mentioned_relations_from_sparql
from executor.sparql_executor import execute_query_with_odbc, get_label, get_label_with_odbc, get_types, get_types_with_odbc
import os


from tqdm import tqdm

def _textualize_relation(r):
    """return a relation string with '_' and '.' replaced"""
    if "_" in r: # replace "_" with " "
        r = r.replace("_", " ")
    if "." in r: # replace "." with " , "
        r = r.replace(".", " , ")
    return r


def load_entity_relation_type_label_from_dataset(split):
    train_databank =load_json(f"data/origin/ComplexWebQuestions_{split}.json")

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


def merge_entity_linking_results(split):
    # print(f'Processing {split}')

    # get entity linking by facc1 and ranking
    # pred_file = f"results/disamb/CWQ_{split}/predictions.json"
    # facc1_el_results = arrange_disamb_results_in_lagacy_format(split, pred_file)
    # dump_json(facc1_el_results,f"data/linking_results/{split}_facc1_el_results.json", indent=4)

    elq_el_results = load_json(f"data/linking_results/CWQ_{split}_entities_elq.json")
    facc1_el_results = load_json(f"data/linking_results/CWQ_{split}_facc1_el_results.json")

    merged_el_results = {}

    for qid in tqdm(facc1_el_results, total=len(facc1_el_results), desc=f"Processing {split}"):
        facc1_pred = facc1_el_results[qid]['entities']
        elq_pred = elq_el_results[qid]

        ent_map = {}
        label_mid_map = {}
        for ent in facc1_pred:
            label = get_label_with_odbc(ent)
            ent_map[ent]={
                "label": label,
                "mention": facc1_pred[ent]["mention"],
                "perfect_match": label.lower()==facc1_pred[ent]["mention"].lower()
            }
            label_mid_map[label] = ent
        
        for ent in elq_pred:
            if ent["id"] not in ent_map:
                mid = ent['id']
                label = get_label_with_odbc(mid)

                if label in label_mid_map: # same label, different mid
                    ent_map.pop(label_mid_map[label]) # pop facc1 result, retain elq result

                if label:
                    ent_map[ent["id"]]= {
                        "label": label,
                        "mention": ent["mention"],
                        "perfect_match": label.lower()==ent['mention'].lower()
                    }
                        
        merged_el_results[qid] = ent_map

    dump_json(merged_el_results, f"data/linking_results/merged_CWQ_{split}_linking_results.json", indent=4)

    # print(el_results)


def evaluate_linking_res(split, linker="facc1"):
    gold_et_bank = load_json(f"data/label_maps/CWQ_{split}_label_maps.json")
    if linker.lower()=="facc1":
        facc1_el_results = load_json(f"data/linking_results/CWQ_{split}_facc1_el_results.json")
    elif linker.lower()=="elq":
        elq_el_results = load_json(f"data/linking_results/CWQ_{split}_entities_elq.json")
    elif linker.lower()=="merged":
        merged_el_results = load_json(f"data/linking_results/merged_CWQ_{split}_linking_results.json")
    else:
        facc1_el_results = load_json(f"data/linking_results/CWQ_{split}_facc1_el_results.json")
        elq_el_results = load_json(f"data/linking_results/CWQ_{split}_entities_elq.json")

    avg_p = 0
    avg_r = 0
    avg_f = 0
    total = len(gold_et_bank)
    for qid in gold_et_bank:
        gold_et_map = gold_et_bank[qid]["entity_label_map"]
        gold_type_map = gold_et_bank[qid]["type_label_map"]
        gold_ets = gold_et_map.keys() - gold_type_map.keys()
        if linker=="facc1":
            pred_ets = facc1_el_results[qid]['entities'].keys()
        elif linker.lower()=="elq":
            pred_ets = set([x["id"] for x in elq_el_results[qid]])
        elif linker.lower()=="merged":
            pred_ets = merged_el_results[qid].keys()
        else:
            pred_ets = facc1_el_results[qid]['entities'].keys()
            pred_ets = pred_ets | set([x["id"] for x in elq_el_results[qid]])
        if len(pred_ets)==0:
            local_p,local_r,local_f = (1.0,1.0,1.0) if len(gold_ets)==0 else (0.0,0.0,0.0)
        elif len(gold_ets)==0:
            local_p,local_r,local_f = (1.0,1.0,1.0) if len(pred_ets)==0 else (0.0,0.0,0.0)
        else:
            local_p = len(pred_ets&gold_ets) /len(pred_ets)
            local_r = len(pred_ets&gold_ets) /len(gold_ets)
            local_f = 2*local_p*local_r/(local_p+local_r) if local_p+local_r >0 else 0
        avg_p+=local_p
        avg_r+=local_r
        avg_f+=local_f
    
    
    print(f"{linker.upper()}: AVG P:{avg_p/total}, AVG R:{avg_r/total}, AVG F:{avg_f/total}")

  


def get_all_type_class():
    get_all_type_sparql = """
    SELECT distinct ?resource WHERE{
    ?resource  rdf:type rdfs:Class .
    }
    """
    all_types = execute_query_with_odbc(get_all_type_sparql)
    print(f'TOTAL:{len(all_types)}')

    type_label_class = {}
    
    for fb_typ in tqdm(all_types,total=len(all_types),desc="Processing"):
        try:
            get_label_sparql = ("""
            SELECT ?label WHERE{
                """
                f'<{fb_typ}> rdfs:label ?label'
                """
                FILTER (lang(?label) = 'en')
            }
            """)
            # label = get_label_with_odbc(fb_typ)
            label = list(execute_query_with_odbc(get_label_sparql))[0]
            type_label_class[fb_typ]=label
        except Exception:
            continue
    
    

    # print(f"Get Label:{type_label_class}")

    # label_type_class = {l:t for t,l in type_label_class.items()}
    label_type_class = {}
    for t,l in type_label_class.items():
        if l not in label_type_class:
            label_type_class[l]=t
        else:
            if "ns/m." in t and not "ns/m." in label_type_class[l]:
                label_type_class[l]=t
            else:
                continue


    dump_json(type_label_class, "data/fb_type_label_class.json", indent=4)
    dump_json(label_type_class, "data/fb_label_type_class.json", indent=4)

    print("Done")

def test():
    data_bank = load_json('data/label_maps/CWQ_train_entity_label_map.json')
    print(data_bank['m.0y80cnb'])

if __name__=='__main__':
    
    
    # test()
    
    # 处理数据集
    # split_list = ['test','dev','train']
    # for split in split_list:
    #     load_entity_relation_type_label_from_dataset(split)

    # 从Freebase获取所有type
    # get_all_type_class()

    
    # 处理实体链接结果
    # merge_entity_linking_results(split='test')
    # merge_entity_linking_results(split='dev')
    # merge_entity_linking_results(split='train')

    # 按链接器评估实体链接结果
    # evaluate_linking_res('test',linker="merged")
    pass

    
    




