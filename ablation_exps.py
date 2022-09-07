"""
Ablation Experiments
"""
import json
import os
import re
from tqdm import tqdm
import argparse

def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('action',type=str,help='Action to operate')
    parser.add_argument('--dataset', default='CWQ', help='CWQ or WebQSP')
    parser.add_argument('--eval_beam_size', default=50, type=int)
    parser.add_argument('--model_type', default='full', type=str, help='full or base')

    return parser.parse_args()

"""
Evaluation functions
"""
def evaluate_question_with_unseen_relation_entity(dataset='CWQ', model_type='full'):
    if dataset.lower() == 'cwq':
        if model_type.lower() == 'full':
            predictions_path = 'exps/CWQ_GMT_KBQA/beam_50_test_4_top_k_predictions.json_gen_sexpr_results.json_new.json'
        elif model_type.lower() == 'base':
            predictions_path = 'exps/CWQ_t5_base/beam_50_test_4_top_k_predictions.json_gen_sexpr_results.json_new.json'
        else:
            return
        predictions = load_json(predictions_path)
        unseen_qids = load_json('data/CWQ/generation/ablation/test_unseen_entity_or_relation_qids.json')
    elif dataset.lower() == 'webqsp':
        if model_type.lower() == 'full':
            predictions_path = 'exps/WebQSP_GMT_KBQA/beam_50_test_2_top_k_predictions.json_gen_sexpr_results_official_format.json_new.json'
        elif model_type.lower() == 'base':
            predictions_path = 'exps/WebQSP_t5_base/beam_50_test_2_top_k_predictions.json_gen_sexpr_results_official_format.json_new.json'
        else:
            return
        predictions = load_json(predictions_path)
        unseen_qids = load_json('data/WebQSP/generation/ablation/test_unseen_entity_or_relation_qids.json')
    else:
        unseen_qids = None
        return
    prediction_map = {pred["qid"]: pred for pred in predictions}
    
    p_list = []
    r_list = []
    f_list = []
    acc_num = 0

    for qid in tqdm(unseen_qids, total=len(unseen_qids), desc='Evaluating QA performance on question with unseen KB relation/entity'):
        if qid not in prediction_map:
            p = 0.0
            r = 0.0
            f = 0.0
        else:
            p = prediction_map[qid]["precision"] 
            r = prediction_map[qid]["recall"]
            f = prediction_map[qid]["f1"]
            if f == 1.0:
                acc_num += 1
        
        p_list.append(p)
        r_list.append(r)
        f_list.append(f)
    
    p_average = sum(p_list)/len(p_list)
    r_average = sum(r_list)/len(r_list)
    f_average = sum(f_list)/len(f_list)

    res = f'Total: {len(p_list)}, ACC:{acc_num/len(p_list)}, AVGP: {p_average}, AVGR: {r_average}, AVGF: {f_average}'

    print(res)

    dirname = os.path.dirname(predictions_path)
    with open (os.path.join(dirname,'eval_results_unseen_relation_or_entity.txt'),'w') as f:
        f.write(res)
        f.flush()

def entity_relation_linking_evaluation(dataset='CWQ', beam_size=50):
    """
    Evaluation of entity linking and relation linking

    Evaluation of entity linking
        - Before multi-task: disamb_entities/merged_{dataset}_test_linking_results.json
        - After multi-task: {dataset}_test_{test_batch_size}_beam_{beam_size}_candidate_entity_map.json

    Evaluation of relation linking:
        - Before multi-task: merged/{dataset}_test.json, in "cand_relation_list" with prediciton logits > 0.0
        - After multi-task: 
            prediction results: beam_{beam_size}_test_{batch_size}_top_k_predictions.json_gen_sexpr_results.json + beam_{beam_size}_test_{batch_size}_top_k_predictions.json_gen_failed_results.json
            relation with prediction logit > 0.5
    """
    if dataset.lower() == 'cwq':
        dirname = 'exps/CWQ_GMT_KBQA'
        gen_failed_predictions = load_json(os.path.join(dirname, f'beam_{beam_size}_test_4_top_k_predictions.json_gen_failed_results.json'))
        gen_succeed_predictions = load_json(os.path.join(dirname, f'beam_{beam_size}_test_4_top_k_predictions.json_gen_sexpr_results.json'))
        predictions = gen_failed_predictions + gen_succeed_predictions

        dataset_content = load_json('data/CWQ/generation/merged/CWQ_test.json')
        label_maps  = load_json('data/CWQ/generation/label_maps/CWQ_test_label_maps.json')
        after_entity_linking_res = load_json(os.path.join(dirname, f'CWQ_test_4_beam_{beam_size}_candidate_entity_map.json'))
        before_entity_linking_res = load_json('data/CWQ/entity_retrieval/disamb_entities/CWQ_merged_test_disamb_entities.json')

    elif dataset.lower() == 'webqsp':
        dirname = 'exps/WebQSP_GMT_KBQA'
        gen_failed_predictions = load_json(os.path.join(dirname, f'beam_{beam_size}_test_2_top_k_predictions.json_gen_failed_results.json'))
        gen_succeed_predictions = load_json(os.path.join(dirname, f'beam_{beam_size}_test_2_top_k_predictions.json_gen_sexpr_results.json'))
        predictions = gen_failed_predictions + gen_succeed_predictions

        dataset_content = load_json('data/WebQSP/generation/merged/WebQSP_test.json')
        label_maps = load_json('data/WebQSP/generation/label_maps/WebQSP_test_label_maps.json')
        after_entity_linking_res = load_json(os.path.join(dirname, f'WebQSP_test_2_beam_{beam_size}_candidate_entity_map.json'))
        before_entity_linking_res = load_json('data/WebQSP/entity_retrieval/disamb_entities/WebQSP_merged_test_disamb_entities.json')

    else:
        return
    
    assert len(predictions) == len(label_maps), print(len(predictions), len(dataset))
    assert len(predictions) == len(dataset_content), print(len(predictions), len(dataset_content))
    golden_entities = []
    golden_relations = []
    after_entity_predictions = []
    after_relation_predictions = []
    before_entity_predictions = []
    before_relation_predictions = []

    predictions_map = {pred["qid"]: pred for pred in predictions}
    dataset_content = {example["ID"]: example for example in dataset_content}

    for qid in tqdm(label_maps, total=len(label_maps)):
        assert qid in predictions_map, print(qid)
        golden_entities.append(list(label_maps[qid]["entity_label_map"].keys()))
        golden_relations.append(list(label_maps[qid]["rel_label_map"].keys()))
        after_relation_pred_indexes = [idx for (idx, score) in enumerate(predictions_map[qid]["pred"]["pred_relation_clf_labels"]) if float(score) > 0.5]
        before_relation_predictions.append([item[0] for item in dataset_content[qid]["cand_relation_list"] if float(item[1]) > 0.0])
        after_relation_predictions.append([dataset_content[qid]["cand_relation_list"][idx][0] for idx in after_relation_pred_indexes])
        if qid not in before_entity_linking_res:
            before_entity_predictions.append([])
        else:
            before_entity_predictions.append([item['id'] for item in before_entity_linking_res[qid]])
        if qid not in after_entity_linking_res:
            after_entity_predictions.append([])
        else:
            after_entity_predictions.append([item['id'] for item in after_entity_linking_res[qid].values()])
    
    after_relation_linking_res = general_PRF1(after_relation_predictions, golden_relations)
    after_entity_linking_res = general_PRF1(after_entity_predictions, golden_entities)
    before_relation_linking_res = general_PRF1(before_relation_predictions, golden_relations)
    before_entity_linking_res = general_PRF1(before_entity_predictions, golden_entities)

    with open(os.path.join(dirname, f'beam_{beam_size}_entity_relation_linking_evaluation.txt'), 'w') as f:
        f.write(f'After multi-task, Relation linking: {after_relation_linking_res}\n')
        f.write(f'After multi-task, Entity linking: {after_entity_linking_res}\n')
        f.write(f'Before multi-task, Relation linking: {before_relation_linking_res}\n')
        f.write(f'Before multi-task, Entity linking: {before_entity_linking_res}\n')
        print(f'After multi-task, Relation linking: {after_relation_linking_res}\n')
        print(f'After multi-task, Entity linking: {after_entity_linking_res}\n')
        print(f'Before multi-task, Relation linking: {before_relation_linking_res}\n')
        print(f'Before multi-task, Entity linking: {before_entity_linking_res}\n')


"""
Utility functions
"""
def load_json(fname, mode="r", encoding="utf8"):
    if "b" in mode:
        encoding = None
    with open(fname, mode=mode, encoding=encoding) as f:
        return json.load(f)


def dump_json(obj, fname, indent=4, mode='w' ,encoding="utf8", ensure_ascii=False):
    if "b" in mode:
        encoding = None
    with open(fname, "w", encoding=encoding) as f:
        return json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)


def general_PRF1(predictions, goldens):
    assert len(predictions) == len(goldens), print(len(predictions), len(goldens))
    p_list = []
    r_list = []
    f_list = []
    acc_num = 0
    for (pred, golden) in zip(predictions, goldens):
        pred = set(pred)
        golden = set(golden)
        if pred == golden:
            acc_num+=1
        if len(pred)== 0:
            if len(golden)==0:
                p=1
                r=1
                f=1
            else:
                p=0
                r=0
                f=0
        elif len(golden) == 0:
            p=0
            r=0
            f=0
        else:
            p = len(pred & golden)/ len(pred)
            r = len(pred & golden)/ len(golden)
            f = 2*(p*r)/(p+r) if p+r>0 else 0
        
        p_list.append(p)
        r_list.append(r)
        f_list.append(f)
    
    p_average = sum(p_list)/len(p_list)
    r_average = sum(r_list)/len(r_list)
    f_average = sum(f_list)/len(f_list)
    res = f'Total: {len(p_list)}, ACC:{acc_num/len(p_list)}, AVGP: {p_average}, AVGR: {r_average}, AVGF: {f_average}'
    return res


"""
Data preparation
"""
def get_train_unique_relations_entities(dataset="CWQ"):
    """
    Using files under label_maps/ folder
    """
    if dataset.lower() == 'cwq':
        if os.path.exists('data/CWQ/generation/ablation/train_unique_relation_entity.json'):
            return
    elif dataset.lower() == 'webqsp':
        if os.path.exists('data/WebQSP/generation/ablation/train_unique_relation_entity.json'):
            return
    else:
        return

    if dataset.lower() == 'cwq':
        train_label_maps = load_json('data/CWQ/generation/label_maps/CWQ_train_label_maps.json')
    elif dataset.lower() == 'webqsp':
        train_label_maps = load_json('data/WebQSP/generation/label_maps/WebQSP_train_label_maps.json')

    unique_relations = set()
    unique_entities = set()

    for qid in tqdm(train_label_maps, total=len(train_label_maps)):
        data = train_label_maps[qid]
        rels = data["rel_label_map"].keys()
        for rel in rels:
            unique_relations.add(rel)
        
        entities = data["entity_label_map"].keys()
        for ent in entities:
            unique_entities.add(ent)
    
    if dataset.lower() == 'cwq':
        dump_json({
            'entities': list(unique_entities),
            'relations': list(unique_relations)
        }, 'data/CWQ/generation/ablation/train_unique_relation_entity.json')
    elif dataset.lower() == 'webqsp':
        dump_json({
            'entities': list(unique_entities),
            'relations': list(unique_relations)
        }, 'data/WebQSP/generation/ablation/train_unique_relation_entity.json')
    


def get_test_unseen_questions(dataset='CWQ'):
    if dataset.lower() == 'cwq':
        if os.path.exists('data/CWQ/generation/ablation/test_unseen_entity_or_relation_qids.json'):
            return 
    elif dataset.lower() == 'webqsp':
        if os.path.exists('data/WebQSP/generation/ablation/test_unseen_entity_or_relation_qids.json'):
            return
    else:
        return
    if dataset.lower() == 'cwq':
        test_label_maps = load_json('data/CWQ/generation/label_maps/CWQ_test_label_maps.json')
        train_relation_entity_list = load_json('data/CWQ/generation/ablation/train_unique_relation_entity.json')
    elif dataset.lower() == 'webqsp':
        test_label_maps = load_json('data/WebQSP/generation/label_maps/WebQSP_test_label_maps.json')
        train_relation_entity_list = load_json('data/WebQSP/generation/ablation/train_unique_relation_entity.json')

    entities_list = train_relation_entity_list["entities"]
    relations_list = train_relation_entity_list["relations"]
    unseen_qids = set()

    for qid in tqdm(test_label_maps, total=len(test_label_maps)):
        data = test_label_maps[qid]
        relations = data["rel_label_map"].keys()
        for rel in relations:
            if rel not in relations_list:
                unseen_qids.add(qid)
                
        entities = data["entity_label_map"].keys()
        for ent in entities:
            if ent not in entities_list:
                unseen_qids.add(qid)

    unseen_qids = list(unseen_qids)
    if dataset.lower() == 'cwq':
        dump_json(unseen_qids, 'data/CWQ/generation/ablation/test_unseen_entity_or_relation_qids.json')
    elif dataset.lower() == 'webqsp':
        dump_json(unseen_qids, 'data/WebQSP/generation/ablation/test_unseen_entity_or_relation_qids.json')

if __name__=='__main__':
    args = _parse_args()
    action = args.action

    if action.lower() == 'linking_evaluation':
        entity_relation_linking_evaluation(
            args.dataset,
            args.eval_beam_size
        )
    elif action.lower() == 'unseen_evaluation':
        # Data preparation
        get_train_unique_relations_entities(args.dataset)
        get_test_unseen_questions(args.dataset)
        evaluate_question_with_unseen_relation_entity(args.dataset, args.model_type)