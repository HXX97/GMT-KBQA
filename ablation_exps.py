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



def get_unseen_question_QA_metrics(q_type="schema"):
    """
    type: relation | entity | schema
    """
    predictions_path = '../Data/WEBQSP/generation/exps/WebQSP_t5_generation_20epochs_bs2/beam_50_top_k_predictions.json_gen_sexpr_results.json_new.json'
    predictions = load_json(predictions_path)
    prediction_map = {pred["qid"]: pred for pred in predictions}
    if q_type == "schema":
        unseen_qids = load_json('../Data/WEBQSP/generation/ablation/test_unseen_entity_or_relation_qids.json')
    elif q_type == "relation":
        unseen_qids = load_json('../Data/WEBQSP/generation/ablation/test_unseen_relation_qids.json')
    elif q_type == "entity":
        unseen_qids = load_json('../Data/WEBQSP/generation/ablation/test_unseen_entity_qids.json')
    p_list = []
    r_list = []
    f_list = []
    acc_num = 0
    for qid in unseen_qids:
        if qid not in prediction_map:
            p = 0.0
            r = 0.0
            f = 0.0
        else:
            if prediction_map[qid]["answer_acc"]:
                acc_num += 1
            p = prediction_map[qid]["precision"] if prediction_map[qid]["precision"] else 0.0
            r = prediction_map[qid]["recall"] if prediction_map[qid]["recall"] else 0.0
            f = prediction_map[qid]["f1"] if prediction_map[qid]["f1"] else 0.0
        
        p_list.append(p)
        r_list.append(r)
        f_list.append(f)
    
    p_average = sum(p_list)/len(p_list)
    r_average = sum(r_list)/len(r_list)
    f_average = sum(f_list)/len(f_list)

    res = f'Total: {len(p_list)}, ACC:{acc_num/len(p_list)}, AVGP: {p_average}, AVGR: {r_average}, AVGF: {f_average}'
    print(res)

    dirname = os.path.dirname(predictions_path)
    with open (os.path.join(dirname,'eval_results_unseen_{}_compare.txt'.format(q_type)),'w') as f:
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

        dataset  = load_json('data/CWQ/generation/merged/CWQ_test.json')
        after_entity_linking_res = load_json(os.path.join(dirname, f'CWQ_test_4_beam_{beam_size}_candidate_entity_map.json'))
        before_entity_linking_res = load_json('data/CWQ/entity_retrieval/disamb_entities/CWQ_merged_test_disamb_entities.json')

    elif dataset.lower() == 'webqsp':
        dirname = 'exps/WebQSP_GMT_KBQA'
        gen_failed_predictions = load_json(os.path.join(dirname, f'beam_{beam_size}_test_2_top_k_predictions.json_gen_failed_results.json'))
        gen_succeed_predictions = load_json(os.path.join(dirname, f'beam_{beam_size}_test_2_top_k_predictions.json_gen_sexpr_results.json'))
        predictions = gen_failed_predictions + gen_succeed_predictions

        dataset  = load_json('data/WebQSP/generation/merged/WebQSP_test.json')
        after_entity_linking_res = load_json(os.path.join(dirname, f'WebQSP_test_2_beam_{beam_size}_candidate_entity_map.json'))
        before_entity_linking_res = load_json('data/WebQSP/entity_retrieval/disamb_entities/WebQSP_merged_test_disamb_entities.json')

    else:
        return
    
    assert len(predictions) == len(dataset), print(len(predictions), len(dataset))
    golden_entities = []
    golden_relations = []
    after_entity_predictions = []
    after_relation_predictions = []
    before_entity_predictions = []
    before_relation_predictions = []

    predictions_map = {pred["qid"]: pred for pred in predictions}
    dataset_map = {data["ID"]: data for data in dataset}

    for qid in tqdm(dataset_map, total=len(dataset_map)):
        assert qid in predictions_map, print(qid)
        golden_entities.append(list(dataset_map[qid]["gold_entity_map"].keys()))
        golden_relations.append(list(dataset_map[qid]["gold_relation_map"].keys()))
        after_relation_pred_indexes = [idx for (idx, score) in enumerate(predictions_map[qid]["pred"]["pred_relation_clf_labels"]) if float(score) > 0.5]
        before_relation_predictions.append([item[0] for item in dataset_map[qid]["cand_relation_list"] if float(item[1]) > 0.0])
        after_relation_predictions.append([dataset_map[qid]["cand_relation_list"][idx][0] for idx in after_relation_pred_indexes])
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


def modular_evaluation():
    """
    Evaluation of entity linking and relation linking
    Evaluation of entity linking
        - gen_sexpr_result.json_new.json, compare entity with logit > 0.5 with golden entities
        - WebQSP_candidate_entity_map.json, which is after disamiguation
    Evaluation of relation linking:
        - gen_sexpr_result.json_new.json, compare entity with logit > 0.5 with golden entities
    Entity linking and Relation linking performance before training
        - Relation: "cand_relation_list" from json file, logit > 0
        - entity: /home2/xxhu/QDT2SExpr/CWQ/data/linking_results/merged_WebQSP_test_linking_results.json
    """
    predictions_file = '../Data/WEBQSP/generation/exps/WebQSP_t5_generation_20epochs_bs2/beam_50_top_k_predictions.json_gen_sexpr_results.json_new.json'
    dirname = os.path.dirname(predictions_file)
    predictions = load_json(predictions_file)
    predictions_failed = load_json('../Data/WEBQSP/generation/exps/WebQSP_t5_generation_20epochs_bs2/beam_50_top_k_predictions.json_gen_failed_results.json')
    predictions = predictions + predictions_failed
    dataset = load_json('../Data/WEBQSP/generation/merged/WebQSP_test.json')
    disambiguated_entity_res = load_json('../Data/WEBQSP/generation/exps/WebQSP_t5_generation_20epochs_bs2/WebQSP_candidate_entity_map.json')
    facc1_elq_disambiguated_entity_res = load_json('../Data/WEBQSP/entity_retrieval/linking_results/merged_WebQSP_test_linking_results.json')

    assert len(predictions) == len(dataset), print(len(predictions), len(dataset))
    golden_entities = []
    golden_relations = []
    entity_predictions = []
    relation_predictions = []
    disambiguated_entity_predictions = []
    before_relation_predictions = [] # logits predicted by cross-encoder
    facc1_elq_disambiguated_entity_predictions = [] # Candidate entity, after disambiguation

    predictions_map = {pred["qid"]: pred for pred in predictions}
    dataset_map = {data["ID"]: data for data in dataset}
    for qid in dataset_map:
        assert qid in predictions_map, print(qid)
        golden_entities.append(list(dataset_map[qid]["gold_entity_map"].keys()))
        golden_relations.append(list(dataset_map[qid]["gold_relation_map"].keys()))
        entity_pred_indexes = [idx for (idx, score) in enumerate(predictions_map[qid]["pred"]["pred_entity_clf_labels"]) if float(score) > 0.5]
        entity_predictions.append([dataset_map[qid]["cand_entity_list"][idx]["id"] for idx in entity_pred_indexes])
        relation_pred_indexes = [idx for (idx, score) in enumerate(predictions_map[qid]["pred"]["pred_relation_clf_labels"]) if float(score) > 0.5]
        before_relation_predictions.append([item[0] for item in dataset_map[qid]["cand_relation_list"] if float(item[1]) > 0.0])
        relation_predictions.append([dataset_map[qid]["cand_relation_list"][idx][0] for idx in relation_pred_indexes])
        if qid not in disambiguated_entity_res:
            disambiguated_entity_predictions.append([])
        else:
            disambiguated_entity_predictions.append([item['id'] for item in list(disambiguated_entity_res[qid].values())])
        if qid not in facc1_elq_disambiguated_entity_res:
            facc1_elq_disambiguated_entity_predictions.append([])
        else:
            facc1_elq_disambiguated_entity_predictions.append([item for item in list(facc1_elq_disambiguated_entity_res[qid].keys())])
    
    relation_linking_res = general_PRF1(relation_predictions, golden_relations)
    entity_linking_res = general_PRF1(entity_predictions, golden_entities)
    disambiguated_entity_linking_res = general_PRF1(disambiguated_entity_predictions, golden_entities)
    before_relation_linking_res = general_PRF1(before_relation_predictions, golden_relations)
    facc1_elq_disambiguated_entity_linking_res = general_PRF1(facc1_elq_disambiguated_entity_predictions, golden_entities)
    with open(os.path.join(dirname, 'modular_evaluation_results_compare.txt'), 'w') as f:
        f.write('relation_linking_res: {}\n'.format(relation_linking_res))
        f.write('entity_linking_res: {}\n'.format(entity_linking_res))
        f.write('disambiguated_entity_linking_res: {}\n'.format(disambiguated_entity_linking_res))
        f.write('before_relation_linking_res: {}\n'.format(before_relation_linking_res))
        f.write('facc1_elq_disambiguated_entity_linking_res: {}\n'.format(facc1_elq_disambiguated_entity_linking_res))

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


def type_checker(token:str):
    pattern_year = r"^\d{4}$"
    pattern_year_month = r"^\d{4}-\d{2}$"
    pattern_year_month_date = r"^\d{4}-\d{2}-\d{2}$"
    if re.match(pattern_year, token):
        return True
    elif re.match(pattern_year_month, token):
        return True
    elif re.match(pattern_year_month_date, token):
        return True
    else:
        return False


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
    if dataset.lower() == 'cwq':
        if os.path.exists('data/CWQ/generation/ablation/train_unique_relation_entity.json'):
            print('file exists')
            return
    elif dataset.lower() == 'webqsp':
        if os.path.exists('data/WebQSP/generation/ablation/train_unique_relation_entity.json'):
            print('file exists')
            return
    else:
        return

    if dataset.lower() == 'cwq':
        train_dataset = load_json('data/CWQ/generation/merged/CWQ_train.json')
    elif dataset.lower() == 'webqsp':
        train_dataset = load_json('data/WebQSP/generation/merged/WebQSP_train.json')

    unique_relations = set()
    unique_entities = set()

    for data in tqdm(train_dataset, total=len(train_dataset)):
        rels = list(data["gold_relation_map"].keys())
        for rel in rels:
            unique_relations.add(rel)
        
        entities = list(data["gold_entity_map"].keys())
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
            print('file exists')
            return 
    elif dataset.lower() == 'webqsp':
        if os.path.exists('data/WebQSP/generation/ablation/test_unseen_entity_or_relation_qids.json'):
            print('file exists')
            return
    else:
        return
    if dataset.lower() == 'cwq':
        test_dataset = load_json('data/CWQ/generation/merged/CWQ_test.json')
        train_relation_entity_list = load_json('data/CWQ/generation/ablation/train_unique_relation_entity.json')
    elif dataset.lower() == 'webqsp':
        test_dataset = load_json('data/WebQSP/generation/merged/WebQSP_test.json')
        train_relation_entity_list = load_json('data/WebQSP/generation/ablation/train_unique_relation_entity.json')

    entities_list = train_relation_entity_list["entities"]
    relations_list = train_relation_entity_list["relations"]
    unseen_qids = set()

    for data in tqdm(test_dataset, total=len(test_dataset)):
        qid = data["ID"]
        relations = list(data["gold_relation_map"].keys())
        for rel in relations:
            if rel not in relations_list:
                unseen_qids.add(qid)
                
        entities = list(data["gold_entity_map"].keys())
        for ent in entities:
            if ent not in entities_list:
                unseen_qids.add(qid)

    unseen_qids = list(unseen_qids)
    if dataset.lower() == 'cwq':
        dump_json(unseen_qids, 'data/CWQ/generation/ablation/test_unseen_entity_or_relation_qids.json')
    elif dataset.lower() == 'webqsp':
        dump_json(unseen_qids, 'data/WebQSP/generation/ablation/test_unseen_entity_or_relation_qids.json')


def get_test_relation_unseen_questions():
    test_dataset = load_json('data/CWQ/final/merged/CWQ_test.json')
    train_relation_entity_list = load_json('data/CWQ/final/ablation/train_unique_relation_entity.json')
    relations_list = train_relation_entity_list["relations"]
    unseen_qids = []

    for data in test_dataset:
        qid = data["ID"]
        rels = list(data["gold_relation_map"].keys())
        for rel in rels:
            if rel not in relations_list:
                unseen_qids.append(qid)
                continue
    dump_json(unseen_qids, 'data/CWQ/final/ablation/test_unseen_relation_qids.json')


def get_test_entity_unseen_questions():
    test_dataset = load_json('data/WebQSP/final/merged/WebQSP_test.json')
    train_relation_entity_list = load_json('data/WebQSP/final/ablation/train_unique_relation_entity.json')
    entities_list = train_relation_entity_list["entities"]
    unseen_qids = []

    for data in test_dataset:
        qid = data["ID"]
        ents = list(data["gold_entity_map"].keys())
        for ent in ents:
            if ent not in entities_list:
                unseen_qids.append(qid)
                continue
    dump_json(unseen_qids, 'data/WebQSP/final/ablation/test_unseen_entity_qids.json')


def check_unseen_qids():
    all_unseen_qids = load_json('data/WebQSP/final/ablation/test_unseen_entity_or_relation_qids.json')
    relation_unseen_qids = load_json('data/WebQSP/final/ablation/test_unseen_relation_qids.json')
    entity_unseen_qids = load_json('data/WebQSP/final/ablation/test_unseen_entity_qids.json')
    assert set(all_unseen_qids) == set(relation_unseen_qids) | set(entity_unseen_qids)


"""
Error Analysis
"""

def compare_linking_results():
    classified_linking_results = load_json('exps/final/WebQSP_relation_entity_concat_add_prefix_warmup_epochs_5_20epochs_bs2/WebQSP_candidate_entity_map.json') # 多任务模型输出的结果
    prev_linking_results = load_json('/home2/xxhu/QDT2SExpr/CWQ/data/linking_results/merged_WebQSP_test_linking_results.json')
    dataset = load_json('data/WebQSP/final/merged/WebQSP_test.json')
    a_diff_b = [] # a True, b False
    b_diff_a = [] # b True, a False
    for data in dataset:
        qid = data["ID"]
        if qid not in classified_linking_results:
            if qid in prev_linking_results:
                b_diff_a.append(qid)
            continue
        if qid not in prev_linking_results:
            if qid in classified_linking_results:
                a_diff_b.append(qid)
            continue
        classified_res = set([v['id'] for (k,v) in classified_linking_results[qid].items()])
        prev_res = set(prev_linking_results[qid].keys())
        golden_res = set(data["gold_entity_map"].keys())

        if classified_res == golden_res:
            if prev_res != golden_res:
                a_diff_b.append(qid)
        if prev_res == golden_res:
            if classified_res != golden_res:
                b_diff_a.append(qid)
        
    print('Classified true, prev wrong: {} {}'.format(len(a_diff_b), a_diff_b))
    print('Prev true, Classified wrong: {} {}'.format(len(b_diff_a), b_diff_a))


def get_date_time_error():
    denormalization_errors = ['WebQTest-1000_a9122e6c3ec58aea7483e18c7074128a', 'WebQTest-212_d5309c79dc99d1d829805d88dd254553', 'WebQTest-1000_27638ce6ed6f359e62d04c2c4e322eee', 'WebQTest-1000_7457bb008a1da743b19ff9ce3a5cac63', 'WebQTest-1000_e1d76e5ac038c985924d768c2027643f', 'WebQTest-1000_3f03848605c6758ff2230a955cd92d65']
    dataset = load_json('data/CWQ/final/merged/CWQ_test.json')
    dataset = {data["ID"]: data for data in dataset}
    ids = []
    questions = []
    for qid in denormalization_errors:
        question = dataset[qid]["question"]
        questions.append(question)
        tokens = question.split(' ')
        for tok in tokens:
            if type_checker(tok):
                ids.append(qid)
    print(ids)


def main():
    # QA results on set questions with unseen KB relations/entities/schemas
    # get_unseen_question_QA_metrics(q_type="relation")
    # get_unseen_question_QA_metrics(q_type="entity")
    # get_unseen_question_QA_metrics(q_type="schema")

    # modular evaluation, including entity and relation linking evaluation
    modular_evaluation()

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