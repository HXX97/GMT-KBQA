# from typing import final
from textwrap import indent
from tqdm import tqdm
import json
import argparse
from executor.sparql_executor import get_freebase_mid_from_wikiID, get_label, get_label_with_odbc
from entity_retrieval.aqqu_entity_linker import IdentifiedEntity
from entity_retrieval import surface_index_memory
from entity_retrieval.bert_entity_linker import BertEntityLinker
from components.utils import dump_json, load_json, clean_str
import requests
from nltk.tokenize import word_tokenize
import os
from config import ELQ_SERVICE_URL

"""
This file performs candidate entity linking for CWQ and WebQSP,
using BERT_NER+FACC1 or ELQ.
"""

def _parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default='CWQ', help='dataset to perform entity linking, should be CWQ or WebQSP')
    parser.add_argument('--split', required=True, help='split to operate on') # the split file: ['dev','test','train']
    parser.add_argument('--linker', default='FACC1', help='linker, should be FACC1 or ELQ')
    
    parser.add_argument('--server_ip',default=None,required=False, help='server ip for debugger to attach')
    parser.add_argument('--server_port',default=None,required=False, help='server port for debugger to attach')

    return parser.parse_args()


def to_output_data_format(identified_entity):
    """Transform an identified entity to a dict"""
    data = {}
    data['label'] = identified_entity.name
    data['mention'] = identified_entity.mention
    data['pop_score'] = identified_entity.score
    data['surface_score'] = identified_entity.surface_score
    data['id'] = identified_entity.entity.id
    data['aliases'] = identified_entity.entity.aliases
    data['perfect_match'] = identified_entity.perfect_match
    return data


def get_all_entity_candidates(linker, utterance):
    """get all the entity candidates given an utterance

    @param linker: entity linker
    @param utterance: natural language utterance
    @return: a list of all candidate entities
    """
    mentions = linker.get_mentions(utterance) # get all the mentions detected by ner model
    identified_entities = []
    mids = set()
    all_entities = []
    for mention in mentions:
        results_per_mention = []
        # use facc1
        entities = linker.surface_index.get_entities_for_surface(mention)
        # use google kg api
        if len(entities) == 0 and len(mention) > 3 and mention.split()[0] == 'the':
            mention = mention[3:].strip()
            entities = linker.surface_index.get_entities_for_surface(mention)

        elif len(entities) == 0 and f'the {mention}' in utterance:
            mention = f'the {mention}'
            entities = linker.surface_index.get_entities_for_surface(mention) 

        if len(entities) == 0:
            continue

        entities = sorted(entities, key=lambda x:x[1], reverse=True)
        for i, (e, surface_score) in enumerate(entities):
            if e.id in mids:
                continue
            # Ignore entities with low surface score. But if even the top 1 entity is lower than the threshold,
            # we keep it
            perfect_match = False
            # Check if the main name of the entity exactly matches the text.
            # I only use the label as surface, so the perfect match is always True
            if linker._text_matches_main_name(e, mention):
                perfect_match = True
            ie = IdentifiedEntity(mention,
                                    e.name,
                                    e, 
                                    e.score, 
                                    surface_score,
                                    perfect_match)
            # self.boost_entity_score(ie)
            # identified_entities.append(ie)
            mids.add(e.id)
            results_per_mention.append(to_output_data_format(ie))
        results_per_mention.sort(key=lambda x: x['surface_score'], reverse=True)
        all_entities.append(results_per_mention)

    return all_entities


def dump_entity_linking_results_for_CWQ(split,keep=10):
    
    # 1. build and load entity linking surface index
    # surface_index_memory.EntitySurfaceIndexMemory(entity_list_file, surface_map_file, output_prefix)
    surface_index = surface_index_memory.EntitySurfaceIndexMemory(
        "data/common_data/facc1/entity_list_file_freebase_complete_all_mention", 
        "data/common_data/facc1/surface_map_file_freebase_complete_all_mention",
        "data/common_data/facc1/freebase_complete_all_mention")

    # 2. load BERTEntityLinker
    entity_linker = BertEntityLinker(surface_index, model_path="/BERT_NER/trained_ner_model/", device="cuda:0")
    # sanity check
    sanity_checking = get_all_entity_candidates(entity_linker, "the music video stronger was directed by whom")
    print('RUNNING Sanity Checking on untterance')
    print('\t', "the music video stronger was directed by whom")
    print('Checking result', sanity_checking[0][:2])
    print('Checking result should successfully link stronger to some nodes in Freebase (MIDs)')
    print('If checking result does not look good please check if the linker has been set up successfully')

    # 3. Load dataset split file
    #datafile = f'data/origin/ComplexWebQuestions_{split}.json'
    datafile = f'data/CWQ/sexpr/CWQ.{split}.expr.json'
    data = load_json(datafile, encoding='utf8')
    print(len(data))

    # 4. do entity linking
    el_results = {}
    for ex in tqdm(data, total=len(data)):
        question = ex['question']
        question = clean_str(question)
        qid = ex['ID']
        all_candidates = get_all_entity_candidates(entity_linker, question)
        all_candidates = [x[:keep] for x in all_candidates]
        for instance in all_candidates:
            for x in instance:
                x['label'] =  get_label_with_odbc(x['id'])
        el_results[qid]=all_candidates
    
    # 5. dump the entity linking results
    cand_entity_dir = 'data/CWQ/entity_retrieval/candidate_entities'
    with open(f'{cand_entity_dir}/CWQ_{split}_entities_facc1_unranked.json',encoding='utf8', mode='w') as f:
        json.dump(el_results, f, indent=4)


def dump_entity_linking_results_for_WebQSP(split,keep=10):
    
    # 1. build and load entity linking surface index
    # surface_index_memory.EntitySurfaceIndexMemory(entity_list_file, surface_map_file, output_prefix)
    surface_index = surface_index_memory.EntitySurfaceIndexMemory(
        "data/common_data/facc1/entity_list_file_freebase_complete_all_mention", 
        "data/common_data/facc1/surface_map_file_freebase_complete_all_mention",
        "data/common_data/facc1/freebase_complete_all_mention")

    # 2. load BERTEntityLinker
    entity_linker = BertEntityLinker(surface_index, model_path="/BERT_NER/trained_ner_model/", device="cuda:0")
    # sanity check
    sanity_checking = get_all_entity_candidates(entity_linker, "the music video stronger was directed by whom")
    print('RUNNING Sanity Checking on untterance')
    print('\t', "the music video stronger was directed by whom")
    print('Checking result', sanity_checking[0][:2])
    print('Checking result should successfully link stronger to some nodes in Freebase (MIDs)')
    print('If checking result does not look good please check if the linker has been set up successfully')

    # 3. Load dataset split file
    datafile = f'data/WebQSP/origin/WebQSP.{split}.json'
    data = load_json(datafile)['Questions']
    print(len(data))
    
    # 4. do entity linking
    el_results = {}
    for ex in tqdm(data, total=len(data)):
        question = ex['RawQuestion']
        question = clean_str(question)
        qid = ex['QuestionId']
        all_candidates = get_all_entity_candidates(entity_linker, question)
        all_candidates = [x[:keep] for x in all_candidates]
        for instance in all_candidates:
            for x in instance:
                x['label'] =  get_label_with_odbc(x['id'])
        el_results[qid]=all_candidates
    
    # 5. dump the entity linking results
    with open(f'data/WebQSP/entity_retrieval/candidate_entities/WebQSP_{split}_entities_facc1_unranked.json',encoding='utf8',mode='w') as f:
        json.dump(el_results, f, indent=4)


def get_entity_linking_from_elq(question:str):
    res = requests.post(
        url=ELQ_SERVICE_URL
        , data=json.dumps({'question':question})
        )
    

    cand_ent_list = []
    if res.text:
        try:
            el_res = json.loads(res.text)
        except Exception:
            el_res = None
        detection_res = el_res.get('detection_res',None) if el_res else None

        if detection_res:
            detect = detection_res[0]
            mention_num = len(detect['dbpedia_ids'])
            
            for i in range(mention_num):
                cand_num = len(detect['dbpedia_ids'][i])
                for j in range(cand_num):
                    wiki_id = detect['dbpedia_ids'][i][j]
                    fb_mid = get_freebase_mid_from_wikiID(wiki_id)
                    if fb_mid=='': # empty id
                        continue
                    label = detect['pred_tuples_string'][i][j][0]
                    mention = detect['pred_tuples_string'][i][j][1]
                    # mention_start = detect['pred_triples'][i][1]
                    # mention_end = detect['pred_triples'][i][2]
                    # mention = " ".join(question_tokens[max(0,mention_start):min(len(question_tokens),mention_end)])
                    score = detect['scores'][i][j]
                    
                    el_data = {}
                    el_data['id']=fb_mid
                    el_data['label']=label
                    el_data['mention']=mention
                    el_data['score']=score
                    el_data['perfect_match']= (label==mention or label.lower()==mention.lower())

                    
                    cand_ent_list.append(el_data)
        
        cand_ent_list.sort(key=lambda x:x['score'],reverse=True)

    return cand_ent_list


def dump_entity_linking_results_from_elq_for_CWQ(split, keep=10):
    datafile = f'data/CWQ/sexpr/CWQ.{split}.expr.json'
    data = load_json(datafile,encoding='utf8')
    print(len(data))
    
    # 1. do entity linking by elq
    el_results = {}
    for ex in tqdm(data, total=len(data), desc='Detecting Entities from ELQ for CWQ'):
        question = ex['question']
        question = clean_str(question)
        qid = ex['ID']
        all_candidates = get_entity_linking_from_elq(question)
        el_results[qid]=all_candidates
    
    # 2. dump the entity linking results
    cand_entity_dir = 'data/CWQ/entity_retrieval/candidate_entities'
    if not os.path.exists(cand_entity_dir):
        os.makedirs(cand_entity_dir)
    with open(f'{cand_entity_dir}/CWQ_{split}_cand_entities_elq.json',encoding='utf8',mode='w') as f:
        json.dump(el_results, f, indent=4)


def dump_entity_linking_results_from_elq_for_WebQSP(split, keep=10):
    datafile = f'data/WebQSP/origin/WebQSP.{split}.json'
    data = load_json(datafile)['Questions']
    print(len(data))
    
    # 1. do entity linking
    el_results = {}
    for ex in tqdm(data, total=len(data),desc="Detecting Entities from ELQ for WebQSP"):
        question = ex['RawQuestion']
        question = clean_str(question)
        qid = ex['QuestionId']
        all_candidates = get_entity_linking_from_elq(question)
        el_results[qid]=all_candidates
    
    # 2. dump the entity linking results
    cand_entity_dir = 'data/WebQSP/entity_retrieval/candidate_entities'
    if not os.path.exists(cand_entity_dir):
        os.makedirs(cand_entity_dir)        
    with open(f'{cand_entity_dir}/WebQSP_{split}_cand_entities_elq.json',encoding='utf8',mode='w') as f:
        json.dump(el_results, f, indent=4)


if __name__=='__main__':
    
    
    # question = "what religions are practiced in the country that has the national anthem Afghan National Anthem"
    # print(get_entity_linking_from_elq(question))

    # question = "china"
    # el_list = get_entity_linking_from_elq(question)
    # print(len(el_list))
    # print(el_list)

    
    args = _parse_args()

    # for debugger to attach
    # args.server_ip = '0.0.0.0'
    # args.server_port = 12345

    if args.server_ip and args.server_port:
        import ptvsd
        print('Waiting for debugger to attach...')
        ptvsd.enable_attach(address=(args.server_ip,args.server_port),redirect_output=True)
        ptvsd.wait_for_attach()

    if args.dataset.lower() == 'cwq':
        if args.linker.lower() == 'elq':
            dump_entity_linking_results_from_elq_for_CWQ(args.split)
        else:
            dump_entity_linking_results_for_CWQ(args.split)
    elif args.dataset.lower() == 'webqsp':
        if args.linker.lower() == 'elq':
            dump_entity_linking_results_from_elq_for_WebQSP(args.split)
        else:
            dump_entity_linking_results_for_WebQSP(args.split)
        
    
    print(f'Successfully detected entities for {args.dataset}_{args.split} from {args.linker}')
    