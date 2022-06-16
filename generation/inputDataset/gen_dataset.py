#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   gen_dataset.py
@Time    :   2022/01/06 14:56:57
@Author  :   Xixin Hu 
@Version :   1.0
@Contact :   xixinhu97@foxmail.com
@Desc    :   None
"""

# here put the import lib
import os
from tkinter import N
from typing import List, Optional
from functools import reduce
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from components.utils import extract_mentioned_entities_from_sparql, extract_mentioned_relations_from_sparql
from components.utils import load_json
from executor.sparql_executor import get_label, get_label_with_odbc
from transformers import (BartTokenizer)
from nltk import word_tokenize
from nltk.metrics import edit_distance


class ListDataset(Dataset):
    """
    Dataset for logical form generation
    """

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def __iter__(self):
        return iter(self.examples)


class GenerationExample:
    """
    Generation Example from a raw query to the logcial form
    """

    def __init__(self, qid, query, gt, qdt, 
                entity_label_map, 
                candidate_entity_map={}, 
                candidate_relation_list=[],
                gold_relation_list=[],
                linear_origin_map={},
                answers=[]) -> None:
        self.qid = qid
        self.query = query  # raw question
        self.gt = gt  # ground truth logical form
        self.qdt = qdt
        self.entity_label_map = entity_label_map
        self.candidate_entity_map = candidate_entity_map
        self.candidate_relation_list = candidate_relation_list
        self.gold_relation_list = gold_relation_list
        self.linear_origin_map=linear_origin_map
        self.answers = answers

    def __str__(self):
        return "{}\n\t->{}\n".format(self.query, self.gt.normed_expr)

    def __repr__(self):
        return self.__str__()


class GenerationFeature:
    """
    Feature for generation
    """

    def __init__(self, ex, src_inputs_ids, tgt_input_ids):
        self.ex = ex
        self.src_input_ids = src_inputs_ids
        self.tgt_input_ids = tgt_input_ids


class LFCandidate:
    """The class of logical form candidates

    Attributes:
        s_expr: s-expression
        normed_expr: normalized s-expression
        ex: A boolean indicating exact match
        f1: A double score indicating the f1 score
        edist: A double score indicating the edit distance
    """

    def __init__(self, s_expr, normed_expr, ex=None, f1=None, edist=None) -> None:
        self.s_expr = s_expr
        self.normed_expr = normed_expr
        self.ex = ex
        self.f1 = f1
        self.edist = edist

    def __str__(self):
        return "{}\n\t->{}\n".format(self.s_expr, self.normed_expr)

    def __repr__(self):
        return self.__str__()


def qdt_serialization(qdt_tree):
    """
    serialize a qdt tree into a sequence
    """
    # print(qdt_tree)
    if not isinstance(qdt_tree, list):
        if 'inner_questions' not in qdt_tree:
            return '[DES] ' + qdt_tree["description"]
        else:
            return '[DES] ' + qdt_tree["description"].replace(
                "[IQ1]",
                '[INQ] ' + qdt_serialization(qdt_tree['inner_questions']['IQ1']) + ' [INQ]'
            )
    else:
        return reduce(lambda x, y: x + ' ' + qdt_serialization(y), qdt_tree, '')


def _vanilla_linearization_method(expr, entity_label_map={}, relation_label_map={}, linear_origin_map={}):
    """
    textualize a logical form, replace mids with labels

    Returns:
        (str): normalized s_expr
    """
    expr = expr.replace("(", " ( ") # add space for parantheses
    expr = expr.replace(")", " ) ")
    toks = expr.split(" ") # split by space
    toks = [x for x in toks if len(x)]

    norm_toks = []
    for t in toks:

        # original token
        origin_t = t

        if t.startswith("m.") or t.startswith("g."): # replace entity with its name
            if t in entity_label_map:
                t = entity_label_map[t]
            else:
                # name = get_label(t)
                name = get_label_with_odbc(t)
                if name is not None:
                    entity_label_map[t] = name
                    t = name
            t = '[ '+t+' ]'
        elif "XMLSchema" in t: # remove xml type
            format_pos = t.find("^^")
            t = t[:format_pos]
        elif t == "ge": # replace ge/gt/le/lt
            t = "GREATER EQUAL"
        elif t == "gt":
            t = "GREATER THAN"
        elif t == "le":
            t = "LESS EQUAL"
        elif t == "lt":
            t = "LESS THAN"
        else:
            # TODO 对于没有xml类型的float型数字，如"1.8"，会错误拆解
            t = t.replace("_", " ") # replace "_" with " "
            t = t.replace(".", " , ") # replace "." with " , "
            
            if "." in origin_t: # relation
                t = "[ "+t+" ]"
                relation_label_map[origin_t]=t
        
        norm_toks.append(t)
        linear_origin_map[t] = origin_t # for reverse transduction
        
    return " ".join(norm_toks)

def _textualize_relation(r):
    """return a relation string with '_' and '.' replaced"""
    if "_" in r: # replace "_" with " "
        r = r.replace("_", " ")
    if "." in r: # replace "." with " , "
        r = r.replace(".", " , ")
    return r

def _tokenize_relation(r):
    """return a token list of the relation"""
    return r.replace('.', ' ').replace('_', ' ').split()

def cwq_read_gen_examples_from_json(
    data_setfile, split_file, qdt_file, 
    candidate_entity_file, candidate_relation_file,
    is_eval=False, use_qdt=False, 
    add_entity=False, add_relation=False,
    gold_entity=False, gold_relation=False,
    candidate_relation_num=5,
    gold_relation_num=0
    ) -> List[GenerationExample]:
    """Read cwq dataset file to generate examples"""

    if use_qdt:
        qdt_bank = load_json(qdt_file)        
        if isinstance(qdt_bank, list): # old qdt
            qdt_bank = {x['ID']:x for x in qdt_bank}
        else:
            # new qdt
            pass

    # data_bank = load_json(data_setfile) # raw data
    # data_bank = {x["ID"]: x for x in data_bank}
    split_bank = load_json(split_file) # raw data with s_expr

    if add_entity:
        # candidate entities
        candidate_entity_bank = load_json(candidate_entity_file)

    if add_relation:
        # candidate relations
        candidate_relation_data = load_json(candidate_relation_file)
        # candidate_relation_bank = {x['ID']:x['Candidate relations'] for x in candidate_relation_data}
        candidate_relation_bank = candidate_relation_data

    
    examples = []
    for data in tqdm(split_bank, desc="Reading", total=len(split_bank)):
        if use_qdt:
            qdt_data = qdt_bank[data["ID"]]
        else:
            qdt_data = None
        
        if add_entity:
            cand_entities = candidate_entity_bank[data["ID"]]
        else:
            cand_entities = None
        
        if add_relation:
            candidate_relation_logit_list = candidate_relation_bank.get(data["ID"],None)
            if not candidate_relation_logit_list:
                cand_relations = []
            else:
                if candidate_relation_num>0:
                    # retain top n cand relations by the setting
                    cand_relations = [x[0] for x in candidate_relation_logit_list[:candidate_relation_num]]
                else:
                    # candidate_relation_num == 0
                    # retain relations with logtis greater than 0
                    cand_relations = [x[0] for x in candidate_relation_logit_list if x[1]>=0]
                    if not cand_relations:
                        # no logits greater than 0 , retain top 1
                        cand_relations = [x[0] for x in candidate_relation_logit_list[:1]]
        else:
            cand_relations = []

        ex = proc_cwq_gen_exs(data, qdt_data, cand_entities, cand_relations, is_eval, use_qdt, add_entity, add_relation, gold_entity, gold_relation, gold_relation_num)
        
        if ex is None:
            continue
        examples.append(ex)

    return examples


def proc_cwq_gen_exs(example:dict, 
                     qdt_data:Optional[dict], 
                     candidate_entities:Optional[dict],
                     candidate_relations:Optional[List],
                     is_eval:bool, 
                     use_qdt=False,
                     add_entity=False,
                     add_relation=False,
                     gold_entity=False,
                     gold_relation=False,
                     gold_relation_num=0
                     ) -> GenerationExample:
    """process a cwq example into a Generation Example"""
    qid = example["ID"]
    query = example["question"]
    gt_expr = example["SExpr"]  # ground truth s-expr
    entity_label_map = {} # Dict[mid->entity_label]
    relation_label_map = {} # Dict[relation->linearized_relation]
    linear_origin_map = {} # Dict[linearized_token->original_token]

    # failed to generate the example, skip it
    # if is_eval and (gt_expr == "" or gt_expr == "null"):
    if (gt_expr == "" or gt_expr == "null") and not is_eval:
        # for training, abandon examples without gt s_expr
        return None

    # normalize the ground truth s-expression
    norm_gt = _vanilla_linearization_method(gt_expr, entity_label_map, relation_label_map, linear_origin_map)
    if len(entity_label_map) == 0:
        # no entity detected, use sparql
        sparql = example['sparql']
        gt_entities = extract_mentioned_entities_from_sparql(sparql)
        gt_relations = extract_mentioned_relations_from_sparql(sparql)
        entity_label_map = {ent:get_label_with_odbc(ent) for ent in gt_entities}
        relation_label_map = {r:_textualize_relation(r) for r in gt_relations}
        linear_origin_map = {l:e for e,l in entity_label_map.items()}.update({l:r for r,l in relation_label_map.items()})
        

    # ground truth logical form
    gt = LFCandidate(gt_expr, norm_gt, True, 1.0, 0.0)


    if use_qdt:
        if isinstance(qdt_data,dict) and "decomposition" in qdt_data: # old qdt
            linear_qdt= qdt_serialization(qdt_data['decomposition']['root_question'])
        else:
            # new qdt
            linear_qdt = qdt_data
    else:
        linear_qdt= None
        
    if add_entity:
        # candidate entity label map
        cand_ent_map = {} # Dict[mid->label]
        for mid in candidate_entities:
            label = candidate_entities[mid]['label']
            cand_ent_map[mid] = label.lower()
            # cand_ent_map[label.lower()] = mid
        # for cand in candidate_entities: # retain top5
        #     for ent in cand[:5]:
        #         token = ent['label']
        #         mid = ent['id']
        #         cand_ent_map[token] = mid    
    else:
        cand_ent_map = None
    
    # gold entity label map
    gold_ent_map = entity_label_map

    if add_relation:
        # candidate relation list
        cand_rel_list = list(map(_textualize_relation,candidate_relations))
        # gold_rel_list = list(map(_textualize_relation, gold_rel_list))
        # query = query.lower()
        # question_tokens = word_tokenize(query.lower())
        # cand_rel_list = sorted(cand_rel_list, key=lambda x: edit_distance(x,query), reverse=True)
        # cand_rel_list = cand_rel_list[:10] # retain 10 relations
    else:
        cand_rel_list = None
    # gold relation list
    gold_rel_list = []
    for rel in relation_label_map:
        gold_rel_list.append(_textualize_relation(rel))
        # gold_rel_list.extend(relation_label_map[token])
    
    if gold_relation_num>0:
        # only add partial gold relations
        gold_rel_list = gold_rel_list[:gold_relation_num]


    return GenerationExample(qid, query, gt, linear_qdt, gold_ent_map,
                        candidate_entity_map=cand_ent_map, 
                        candidate_relation_list=cand_rel_list,
                        gold_relation_list=gold_rel_list,
                        linear_origin_map=linear_origin_map,
                        answers=[])

def _extract_gen_feature_from_example(args, tokenizer, ex, 
        add_prefix_space=False):
    """Use Huggingface Tokenizer to encode input and output"""
    qid = ex.qid
    q = ex.query
    qdt = ex.qdt
    gt_lf = ex.gt.normed_expr
    cand_ent_map = ex.candidate_entity_map
    gold_ent_label_map = ex.entity_label_map
    cand_rel_list = ex.candidate_relation_list
    gold_rel_list = ex.gold_relation_list

    # do lower case
    if args.do_lower_case:
        q = q.lower().strip()
        gt_lf = gt_lf.lower().strip()
    
    if not args.use_qdt:
        src_text = q
    else:
        if args.qdt_only:
            src_text = qdt
        else:
            src_text = q+" "+ qdt.strip()
    

    if args.add_entity:
        if args.gold_entity:
            # append gold entity labels
            for ent in gold_ent_label_map:
                src_text = src_text.strip()+" [ENT] "+ gold_ent_label_map[ent].lower().strip()
        else:
            # append candidate entity labels
            for cand in cand_ent_map:
                src_text = src_text.strip()+" [ENT] "+ cand_ent_map[cand].lower().strip()

    if args.add_relation:
        if args.gold_relation:
            for rel in gold_rel_list:
                src_text = src_text.strip()+" [REL] " + rel.lower().strip()
        else:    
            for rel in cand_rel_list:
                src_text = src_text.strip()+" [REL] " + rel.lower().strip()
        # pass
    
    # src_text = q if not use_qdt else f'{q} ; {qdt}'
    src_text = src_text.lower()
    dst_text = gt_lf.lower()

    # encoding = tokenizer(
    #     src_text,
    #     dst_text,
    #     max_length=args.max_source_length,
    #     max_target_length=args.max_target_length,
    #     return_tensors='pt',
    #     add_prefix_space=add_prefix_space
    # ).data

    # src_encoding = tokenizer(
    #     src_text,
    #     max_length=args.max_source_length,
    #     truncation=True,
    #     return_tensors='pt',
    #     # add_prefix_space=add_prefix_space
    # )

    # target_encoding = tokenizer(
    #     dst_text,
    #     max_length=args.max_target_length,
    #     truncation=True,
    #     return_tensors='pt',
    #     # add_prefix_space=add_prefix_space
    # )

    # input_ids, labels = src_encoding['input_ids'], target_encoding['input_ids']
    if add_prefix_space:
        batch_encoding = tokenizer.prepare_seq2seq_batch(
            [src_text],
            [dst_text],
            max_length=args.max_source_length,
            max_target_length=args.max_target_length,
            return_tensors='pt',
            add_prefix_space=add_prefix_space
        ).data
    else:
        batch_encoding = tokenizer.prepare_seq2seq_batch(
            [src_text],
            [dst_text],
            max_length=args.max_source_length,
            max_target_length=args.max_target_length,
            return_tensors='pt'
        ).data


    input_ids, labels = batch_encoding['input_ids'][0], batch_encoding['labels'][0]

    return GenerationFeature(ex,input_ids, labels)


def extract_gen_features_from_examples(args, tokenizer, examples)->List[GenerationFeature]:
    """Extract Generation Features from examples with Huggingface Tokenizer"""
    features = []
    add_prefix_space = isinstance(tokenizer, BartTokenizer) # whether to add prefix space
    
    # indexing the examples to generate features
    for ex in tqdm(examples, desc='Indexing', total=len(examples)):
        feat = _extract_gen_feature_from_example(args, tokenizer, ex, add_prefix_space=add_prefix_space)
        features.append(feat)
    return features


def cwq_load_and_cache_gen_examples(args, tokenizer, evaluate=False)-> ListDataset:
    """Load and cache generation examples of CWQ, return a ListDataset"""
    # load CWQ generate examples
    logger = args.logger

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    # split_id = args.split
    split_file = (
        args.predict_file if evaluate else args.train_file
    )  # if evaluate, use predict file
    dataset_id = os.path.basename(split_file).split("_")[0]  # CWQ, Grail, WebQSP
    split_id = os.path.basename(split_file).split("_")[1]  # dev, test, train

    # make feature cache dir
    if not os.path.exists("feature_cache"):
        os.mkdir("feature_cache")
    

    cachefile_name = "gen_{}_{}_{}".format(dataset_id, split_id, args.model_type, args.use_qdt)

    
    if args.use_qdt:
        cachefile_name+="_useQdt"
        if args.new_qdt:
            cachefile_name+="_newQdt"
        if args.qdt_only:
            cachefile_name+="_qdtOnly"
    

    if args.add_entity:
        
        if args.gold_entity:
            cachefile_name = cachefile_name+"_goldEntity"
        else:
            cachefile_name = cachefile_name+"_candEntity"
    
    candidate_relation_num = args.cand_relation_num
    gold_relation_num = args.gold_relation_num

    if args.add_relation:
        if args.gold_relation:
            cachefile_name = cachefile_name+"_goldRelation_"+str(gold_relation_num)
        else:
            cachefile_name = cachefile_name+"_candRelation_top"+str(candidate_relation_num)


    cached_features_file = os.path.join(
        "feature_cache",cachefile_name
    )

    # cached_features_file = os.path.join(
    #     "feature_cache", "gen_{}_{}_{}_{}_{}_{}_{}".format(
    #         dataset_id, split_id, args.model_type, args.use_qdt, args.qdt_only,
    #         'addEntity' if args.add_entity else 'noEntity',
    #         'addRelation' if args.add_relation else 'noRelation'
    #         )
    # )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:  # feature cache not exists
        logger.info("Creating features from dataset file at %s", input_dir)
        dataset_file = os.path.join(
            input_dir, "origin", f"ComplexWebQuestions_{split_id}.json"
        )
        
        if args.new_qdt:
            qdt_file = os.path.join(
                input_dir, "qdt", f"CWQ_{split_id}_new_qdt.json"
            )
        else:
            qdt_file = os.path.join(
                input_dir, "qdt", f"CWQ_{split_id}_qdt_predict.json"
            )

        candidate_entity_file = os.path.join(
            input_dir, "linking_results", f"merged_CWQ_{split_id}_linking_results.json"
        )

        
        candidate_relation_file = os.path.join(
            input_dir, "rel_match","relations","sorted_results_new",f"CWQ_{split_id}_cand_rel_logits.json"
        )

        
        examples = cwq_read_gen_examples_from_json(dataset_file, 
                                                split_file, 
                                                qdt_file, 
                                                candidate_entity_file, 
                                                candidate_relation_file, 
                                                is_eval=evaluate, 
                                                use_qdt=args.use_qdt,
                                                add_entity=args.add_entity,
                                                add_relation=args.add_relation,
                                                gold_entity=args.gold_entity,
                                                gold_relation=args.gold_relation,
                                                candidate_relation_num = candidate_relation_num,
                                                gold_relation_num = gold_relation_num
                                                )
        features = extract_gen_features_from_examples(args, tokenizer, examples)

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    
    return ListDataset(features)


def generation_collate_fn(data,tokenizer):
    """For dynamic padding"""
    all_input_ids = []
    all_labels = []
    
    for feat in data:
        all_input_ids.append(feat.src_input_ids)
        all_labels.append(feat.tgt_input_ids)

    src_encoded = tokenizer.pad({'input_ids': all_input_ids},return_tensors='pt')
    tgt_encoded = tokenizer.pad({'input_ids': all_labels},return_tensors='pt')

    return {
        'input_ids': src_encoded['input_ids'],
        'attention_mask': src_encoded['attention_mask'],
        'labels': tgt_encoded['input_ids']
    }

