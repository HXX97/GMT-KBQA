# from typing import final
import sys
sys.path.append('..')
from tqdm import tqdm
import json
import argparse
from executor.sparql_executor import get_freebase_mid_from_wikiID, get_wikipage_id_from_dbpedia_uri
from entity_retrieval.aqqu_entity_linker import IdentifiedEntity
# from entity_retrieval import surface_index_memory
import entity_retrieval.surface_index_memory
from entity_retrieval.bert_entity_linker import BertEntityLinker
from components.utils import dump_json, load_json, clean_str
import requests
from nltk.tokenize import word_tokenize
import os



def _parse_args():
    parser = argparse.ArgumentParser()
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
    # print(mentions)
    all_entities = []
    for mention in mentions:
        results_per_mention = []
        # use facc1
        entities = linker.surface_index.get_entities_for_surface(mention)
        # use google kg api
        # entities = get_entity_from_surface(mention)
        # if len(entities) == 0:
        #     entities = get_entity_from_surface(mention)
        # print('A Mention', mention)
        # print('Init Surface Entitis', len(entities), entities)
        if len(entities) == 0 and len(mention) > 3 and mention.split()[0] == 'the':
            mention = mention[3:].strip()
            entities = linker.surface_index.get_entities_for_surface(mention)
            # print('Removing the then Surface Entitis', len(entities), entities)
        elif len(entities) == 0 and f'the {mention}' in utterance:
            mention = f'the {mention}'
            entities = linker.surface_index.get_entities_for_surface(mention)
            # print('Adding the then Surface Entitis', len(entities), entities)

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
                                    e.name, e, e.score, surface_score,
                                    perfect_match)
            # self.boost_entity_score(ie)
            # identified_entities.append(ie)
            mids.add(e.id)
            results_per_mention.append(to_output_data_format(ie))
        results_per_mention.sort(key=lambda x: x['surface_score'], reverse=True)
        # print(results_per_mention[:5])
        all_entities.append(results_per_mention)

    return all_entities


def dump_entity_linking_results(split,keep=10):
    
    # 1. build and load entity linking surface index
    # surface_index_memory.EntitySurfaceIndexMemory(entity_list_file, surface_map_file, output_prefix)
    surface_index = surface_index_memory.EntitySurfaceIndexMemory(
        "entity_linker/data/entity_list_file_freebase_complete_all_mention", "entity_linker/data/surface_map_file_freebase_complete_all_mention",
        "entity_linker/data/freebase_complete_all_mention")

    # 2. load BERTEntityLinker
    entity_linker = BertEntityLinker(surface_index, model_path="/BERT_NER/trained_ner_model/", device="cuda:2")
    # sanity check
    sanity_checking = get_all_entity_candidates(entity_linker, "the music video stronger was directed by whom")
    print('RUNNING Sanity Checking on untterance')
    print('\t', "the music video stronger was directed by whom")
    print('Checking result', sanity_checking[0][:2])
    print('Checking result should successfully link stronger to some nodes in Freebase (MIDs)')
    print('If checking result does not look good please check if the linker has been set up successfully')


    # 3. Load dataset split file
    #datafile = f'data/origin/ComplexWebQuestions_{split}.json'
    datafile = f'data/sexpr/CWQ.{split}.expr.json'
    data = load_json(datafile)
    print(len(data))
    
    with open(datafile,'r',encoding='utf8') as f:
        data = json.load(f)
        print(len(data))
    
    # 4. do entity linking
    el_results = {}
    for ex in tqdm(data, total=len(data)):
        question = ex['question']
        question = clean_str(question)
        qid = ex['ID']
        all_candidates = get_all_entity_candidates(entity_linker, question)
        all_candidates = [x[:keep] for x in all_candidates]
        el_results[qid]=all_candidates
    
    # 5. dump the entity linking results
    with open(f'data/CWQ_{split}_entities.json',encoding='utf8',mode='w') as f:
        json.dump(el_results, f)



EARL_cache_file = "cache/EARL_cache.json"
EARL_cache = {}
added_num = 0
if os.path.exists(EARL_cache_file):
    EARL_cache = load_json(EARL_cache_file)

def save_EARL_cache():
    print(f'Saving EARL cache, added {added_num} records')
    dump_json(EARL_cache,EARL_cache_file)


def get_top1_entity_linking_from_earl(question:str):
    global EARL_cache
    global added_num
    if question in EARL_cache:
        return EARL_cache[question]
    else:

        headers = {'Content-type': 'application/json'}
        res = requests.post(
            url="http://114.212.190.19:4999/processQuery"
            , headers=headers
            , data=json.dumps({'nlquery':question,'pagerankflag':'false'})
            )
        # print(res)
        # print(res.text)
        result_json = json.loads(res.text)
        reranked_lists = result_json["rerankedlists"]
        # print(len(reranked_lists))
        res_list = []
        if len(reranked_lists)>0:
            cand_ent_list = reranked_lists["0"]
            for cand_ent in cand_ent_list:
                dbpedia_uri = cand_ent[1]
                wikipage_id = get_wikipage_id_from_dbpedia_uri(dbpedia_uri)
                freebase_mid = get_freebase_mid_from_wikiID(wikipage_id)
                res_list.append(freebase_mid)
                break # retain top 1
        
        EARL_cache[question] = res_list

        added_num+=1
        if added_num>0 and added_num%100==0:
            save_EARL_cache()

    return res_list
    


def get_entity_linking_from_elq(question:str):
    res = requests.post(
        url="http://210.28.134.34:5685/entity_linking"
        , data=json.dumps({'question':question})
        )
    
    # print(res.text)

    cand_ent_list = []
    if res.text:
        try:
            el_res = json.loads(res.text)
        except Exception:
            el_res = None
        print(el_res)
        # mention_num = len(el_res['scores'])
        # for i in range(mention_num):
        #     pass
        # question_tokens = question.strip().split(" ")
        # question_tokens = word_tokenize(question.strip())
        detection_res = el_res.get('detection_res',None) if el_res else None
        if detection_res:
            detect = detection_res[0]
            cand_num = len(detect['dbpedia_ids'])

            
            for i in range(cand_num):
                wiki_id = detect['dbpedia_ids'][i]
                fb_mid = get_freebase_mid_from_wikiID(wiki_id)
                label = detect['pred_tuples_string'][i][0]
                mention = detect['pred_tuples_string'][i][1]
                # mention_start = detect['pred_triples'][i][1]
                # mention_end = detect['pred_triples'][i][2]
                # mention = " ".join(question_tokens[max(0,mention_start):min(len(question_tokens),mention_end)])
                score = detect['scores'][i]
                
                el_data = {}
                el_data['id']=fb_mid
                el_data['label']=label
                el_data['mention']=mention
                el_data['score']=score
                el_data['perfect_match']= (label==mention or label.lower()==mention.lower())
                #print(el_data)
                
                cand_ent_list.append(el_data)
            
        # print(final_res_list)
    return cand_ent_list
    
def dump_entity_linking_results_from_elq(split,keep=10):
    datafile = f'data/sexpr/CWQ.{split}.expr.json'
    data = load_json(datafile)
    print(len(data))
    
    with open(datafile,'r',encoding='utf8') as f:
        data = json.load(f)
        print(len(data))
    
    # 4. do entity linking
    el_results = {}
    for ex in tqdm(data, total=len(data)):
        question = ex['question']
        question = clean_str(question)
        qid = ex['ID']
        all_candidates = get_entity_linking_from_elq(question)
        el_results[qid]=all_candidates
    
     # 5. dump the entity linking results
    with open(f'data/linking_results/CWQ_{split}_entities_elq.json',encoding='utf8',mode='w') as f:
        json.dump(el_results, f)

if __name__=='__main__':
    
    # question = "which boxing stance is used by michael tyson?"
    # question = "Who was the president in 1980 of the country that has Azad Kashmir?"
    # question = "Lou Seal is the mascot for the team that last won the World Series when?"
    
    # question = "what religions are practiced in the country that has the national anthem Afghan National Anthem"
    # print(get_entity_linking_from_elq(question))

    question = "china"
    # print(get_entity_linking_from_elq(question))

    print(get_top1_entity_linking_from_earl(question))
    
    """
    args = _parse_args()

    # for debugger to attach
    # args.server_ip = '0.0.0.0'
    # args.server_port = 12345

    if args.server_ip and args.server_port:
        import ptvsd
        print('Waiting for debugger to attach...')
        ptvsd.enable_attach(address=(args.server_ip,args.server_port),redirect_output=True)
        ptvsd.wait_for_attach()

    if args.linker.lower() == 'elq':
        dump_entity_linking_results_from_elq(args.split)
    else:
        dump_entity_linking_results(args.split)
    
    print(f'Successfully detected entities for CWQ_{args.split} from {args.linker}')
    """
    