


#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   rank_dataset.py
@Time    :   2022/02/14 14:28:14
@Author  :   Xixin Hu 
@Version :   1.0
@Contact :   xixinhu97@foxmail.com
@Desc    :   None
'''

# here put the import lib
import torch
import os
from torch.utils.data import Dataset
from tqdm import tqdm
from components.utils import load_json
from components.grail_utils import extract_mentioned_entities
from components.grail_utils import extract_mentioned_entities_from_sparql
from executor.sparql_executor import get_in_relations_with_odbc, get_label_with_odbc, get_out_relations_with_odbc
from nltk.tokenize import word_tokenize


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

class CWQEntityCandidate:
    """Entity Candidate for CWQ"""
    def __init__(self, id, label, facc_label, surface_score, pop_score, relations):
        self.id = id
        self.label = label
        self.facc_label = facc_label
        self.surface_score = surface_score
        self.pop_score = pop_score
        self.relations = relations

    def __str__(self):
        # return self.id + ':' + self.label
        return '{}:{}:{:.2f}'.format(self.id, self.label, self.surface_score)

    def __repr__(self):
        return self.__str__()


class CWQEntityDisambProblem:
    """EntityDsiambProblem, one mention, one target_id, several candidates"""
    def __init__(self, pid, query, mention, target_id, candidates):
        self.pid = pid # problem id
        self.qid = pid.split('#')[0]
        self.query = query
        self.mention = mention
        self.target_id = target_id
        self.candidates = candidates


class CWQEntityDisambInstance:
    """EntityDisambInstance, one question, several DisambProblems"""
    def __init__(self, qid, query, s_expr, target_entities, disamb_problems):
        self.qid = qid
        self.query = query
        self.s_expr = s_expr
        self.target_entities = target_entities
        self.target_labels = [get_label_with_odbc(x) for x in target_entities]
        self.disamb_problems = disamb_problems


class CWQEntityDisambFeature:
    def __init__(self, pid, input_ids, token_type_ids, target_idx):
        self.pid = pid
        self.candidate_input_ids = input_ids
        self.candidate_token_type_ids = token_type_ids
        self.target_idx = target_idx


def proc_instance_cwq(ex,linking_results, cutoff=10):
    """Process origin question examples and linking results, return EntityDisambugation Dataset
    Args:
        ex (dict): origin dataset instance
        linking_results (dict): entity linking candidates
        cutoff (int): maxium number of instances for a singlge question
    
    Returns:

    """
    qid = ex['ID']
    query = ex['question']
    if 'sparql' in ex:
        sparql = ex['sparql']
        entities_in_gt = set(extract_mentioned_entities_from_sparql(sparql))
    

    if 'SExpr' in ex:
        s_expr = ex['SExpr']
        if len(entities_in_gt)==0:
            entities_in_gt = set(extract_mentioned_entities(s_expr))
    
    ranking_problems = []

    for idx, entities_per_mention in enumerate(linking_results):
        entities_per_mention = entities_per_mention[:cutoff]

        # no linked entity in this mention
        if not entities_per_mention:
            continue
        
        entities_included = set([e['id'] for e in entities_per_mention])
        candidates = []
        for entity in entities_per_mention:
            eid = entity['id']
            fb_label = get_label_with_odbc(eid) # label in freebase
            in_relations = get_in_relations_with_odbc(eid) # in relations
            out_relations = get_out_relations_with_odbc(eid) # out relations
            
            # create CWQEntityCandidate
            candidates.append(
                CWQEntityCandidate(
                    eid, fb_label, entity['label'], 
                    entity['surface_score'], entity['pop_score'],
                    in_relations | out_relations
                )
            )
        
        target = next((x for x in entities_included if x in entities_in_gt), None)
        problem_id = f'{qid}#{idx}'
        single_problem = CWQEntityDisambProblem(problem_id, query, entities_per_mention[0]['mention'], target, candidates)
        ranking_problems.append(single_problem)

    entity_ex = CWQEntityDisambInstance(qid,query,s_expr, entities_in_gt, ranking_problems)
    return entity_ex


def proc_instance_webqsp(ex, linking_results, cutoff=10):
    qid = ex['QuestionId']
    question = ex['RawQuestion']

    parse = ex['Parses'][0] # take the top 0 parse

    if parse['TopicEntityMid'] is None:
        entities_in_gt = set() # empty entity set
    else:
        entities_in_gt = set([parse['TopicEntityMid']])

    ranking_problems = []

    for idx, entities_per_mention in enumerate(linking_results):
        entities_per_mention = entities_per_mention[:cutoff]

        # no linked entity in this mention
        if not entities_per_mention:
            continue
        
        entities_included = set([e['id'] for e in entities_per_mention])
        candidates = []
        for entity in entities_per_mention:
            eid = entity['id']
            fb_label = get_label_with_odbc(eid) # label in freebase
            in_relations = get_in_relations_with_odbc(eid) # in relations
            out_relations = get_out_relations_with_odbc(eid) # out relations
            
            # create CWQEntityCandidate
            candidates.append(
                CWQEntityCandidate(
                    eid, fb_label, entity['label'], 
                    entity['surface_score'], entity['pop_score'],
                    in_relations | out_relations
                )
            )
        
        target = next((x for x in entities_included if x in entities_in_gt), None)
        problem_id = f'{qid}#{idx}'
        single_problem = CWQEntityDisambProblem(problem_id, question, entities_per_mention[0]['mention'], target, candidates)
        ranking_problems.append(single_problem)

    entity_ex = CWQEntityDisambInstance(qid, question, None, entities_in_gt, ranking_problems)
    return entity_ex


def read_disamb_instances_from_entity_candidates(dataset_file, candidate_file):
    """read entity disambugation instances from entity candidates
    Returns:
        instances (list[CWQEntityDisambInstance]): a list of disambugation instances
    """
    # origin dataset
    dataset = load_json(dataset_file)
    # print(dataset_file)
    # if 'webqsp' in dataset_file.lower():
    #     dataset = dataset['Questions']
    
    # entity linking candidate dataset
    entity_linking_results = load_json(candidate_file)

    instances = []
    dataset = dataset
    
    # for debug
    # dataset = dataset[:10]

    # TODO different processing for WebQSP and CWQ
    if 'webqsp' in dataset_file.lower():
        for data in tqdm(dataset, total=len(dataset), desc='Read Exapmles'):
            qid = data['QuestionId']
            res = entity_linking_results[qid]
            instances.append(proc_instance_webqsp(data, res))

    else:
        for data in tqdm(dataset, total=len(dataset), desc='Read Exapmles'):
            qid = str(data['ID'])
            res = entity_linking_results[qid]
            instances.append(proc_instance_cwq(data, res))

    return instances


class _MODULE_DEFAULT:
    IGONORED_DOMAIN_LIST = ['type', 'common', 'kg', 'dataworld']
    RELATION_FREQ_FILE = 'data/common_data/relation_freq.json'
    RELATION_FREQ = None

def _tokenize_relation(r):
    """return a token list of the relation"""
    return r.replace('.', ' ').replace('_', ' ').split()

def _normalize_relation(r):
    """returan a normalized relation"""
    r = r.replace('.', ' , ')
    r = r.replace('_', ' ')
    return r

def _construct_disamb_context(args, tokenizer, candidate, proc_query_tokens):
    """
    Args:
        args : model arguments
        tokenizer (AutoTokenizer) : model tokenizer
        candidate (CWQEntityCandidate): candidate entity
        proc_query_tokens (list[str]): tokens of original question

    Returns:
        relations_str (str) : a string joining the ranked relaions
    """
    if _MODULE_DEFAULT.RELATION_FREQ is None:
        _MODULE_DEFAULT.RELATION_FREQ = load_json(_MODULE_DEFAULT.RELATION_FREQ_FILE)
    
    # filter relations by domain
    relations = [x for x in candidate.relations if x.split('.')[0] not in _MODULE_DEFAULT.IGONORED_DOMAIN_LIST]

    def key_func(r):
        """get the ranking key of relation r"""
        r_tokens = _tokenize_relation(r)
        overlapping_val = len(set(proc_query_tokens) & set(r_tokens))
        return(
            _MODULE_DEFAULT.RELATION_FREQ.get(r,1),
            -overlapping_val
        )
    
    relations = sorted(relations, key=lambda x: key_func(x))

    relations_str = ' ; '.join(map(_normalize_relation, relations))

    return relations_str


def _extract_disamb_feature_from_problem(args, tokenizer, problem):
    """Extract feature from CWQEntityDisambProblems

    Args:
        args : model arguments
        tokenizer : model tokenizer
        problem (CWQEntityDsiambProblem) : a disambugation problem
    
    Returns:
        CWQEntityDisambFeature

    """
    pid = problem.pid
    query = problem.query
    query_tokens = word_tokenize(query.lower())

    candidate_input_ids = []
    candidate_token_type_ids = []

    if args.do_lower_case:
        query = query.lower()
    
    for c in problem.candidates:
        # one-hop relations of entity c
        relation_info = _construct_disamb_context(args, tokenizer, c, query_tokens)
        # label of entity c
        label_info = c.label
        
        if label_info is None:
            label_info = ''
            print('WANING INVALID LABEL', c.id, '|', c.label, '|', c.facc_label)    
        
        if args.do_lower_case:
            relation_info = relation_info.lower()
            label_info = label_info.lower()
        
        context_info = '{} {} {}'.format(label_info, tokenizer.sep_token, relation_info)

        # encoded entity
        c_encoded = tokenizer(query, context_info, truncation=True, max_length=args.max_seq_length, return_token_type_ids=True)
        candidate_input_ids.append(c_encoded['input_ids'])
        candidate_token_type_ids.append(c_encoded['token_type_ids'])

    target_idx = next((i for (i,x) in enumerate(problem.candidates) if x.id == problem.target_id), 0) 
    return CWQEntityDisambFeature(pid, candidate_input_ids, candidate_token_type_ids, target_idx)


def extract_disamb_features_from_examples(args, tokenizer, instances, do_predict=False):
    """Extract disambugation features from CWQDisambInstances, only retain valid disamb problems
    """
    valid_disamb_problems = []
    baseline_acc = 0
    for inst in instances:
        for p in inst.disamb_problems:
            if not do_predict:
                # do train, only append problems with target_ids
                if (len(p.candidates) > 1) and p.target_id is not None:
                    # if (len(p.candidates) > 0) and p.target_id is not None:
                        valid_disamb_problems.append(p)
                        if p.candidates[0].id == p.target_id:
                            baseline_acc += 1
                else:
                    # print(f'WARNING, Question {p.pid} has no target_id or has no more than one candidates')
                    pass
                    
            else:
                # predict or eval, append all the valid disamb problems
                if (len(p.candidates) >1):
                    valid_disamb_problems.append(p)
                else:
                    # print(f'WARNING, Question {p.pid} has no more than one candidates')
                    pass
                    
    
    hints = 'VALID : {}, ACC: {:.1f}'.format(len(valid_disamb_problems), baseline_acc / len(valid_disamb_problems))
    features = []
    for p in tqdm(valid_disamb_problems, total=len(valid_disamb_problems), desc=hints):
        feat = _extract_disamb_feature_from_problem(args, tokenizer, p)
        features.append(feat)
    return features



def load_and_cache_disamb_examples(args, tokenizer, evaluate=False, output_examples=False):
    # TODO
    logger = args.logger

    if args.local_rank not in [-1,0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    split_file = args.predict_file if evaluate else args.train_file
    dataset_id = os.path.basename(split_file).split('_')[0]
    split_id = os.path.basename(split_file).split('_')[1]
    # split_file = '_'.(join(os.path.basename(split_file).split('_')[:2])
    cached_features_file = os.path.join(
        'feature_cache',"entity_disamb_{}_{}_{}_{}".format(dataset_id, split_id,args.model_type,args.max_seq_length)
    )


    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        # cache exists, load it
        logger.info("Loading features from cached file %s", cached_features_file)
        data = torch.load(cached_features_file)
        examples = data['examples']
        features = data['features']
    else:
        # cache not exists, create it
        logger.info("Creating features from dataset file at %s", input_dir)
        candidate_file = args.predict_file if evaluate else args.train_file
        
        if not os.path.exists('feature_cache'):
            os.makedirs('feature_cache')

        example_cache = os.path.join('feature_cache', f'{dataset_id}_{split_id}_entity_disamb_example.bin')

        if os.path.exists(example_cache) and not args.overwrite_cache:
            examples = torch.load(example_cache)
        else:
            orig_split = split_id
            if dataset_id.lower()=='cwq':
                dataset_file = f'data/CWQ/sexpr/CWQ.{orig_split}.expr.json'
                examples = read_disamb_instances_from_entity_candidates(dataset_file, candidate_file)
            else:

                dataset_file = f'data/WebQSP/sexpr/WebQSP.{orig_split}.expr.json'
                examples = read_disamb_instances_from_entity_candidates(dataset_file, candidate_file)

            torch.save(examples, example_cache)
        
        features = extract_disamb_features_from_examples(args, tokenizer, examples, do_predict=args.do_predict)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({'examples': examples, 'features': features}, cached_features_file)

    
    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    if output_examples:
        return ListDataset(features), examples
    else:
        return ListDataset(features)
