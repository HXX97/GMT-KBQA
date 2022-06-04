import os
from structure_filling_data_process import get_all_masked, get_entity_literal_masked, get_relation_literal_masked
import json

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


def error_analysis_on_generation_failed_cases():
    """
    For questions failed to generate executable logical forms, analyse top 1 generated S-expression
    Error types 
        - golden S-expression is "null"
        - denormaliztion error
        - S-expression conversion error (when both normed and denormed S-expression is consistent with golden, but failed to generate executable)
        - structure error
        - entity error
        - relation error
    """
    prediction_file = 'exps/final/WebQSP_relation_entity_concat_add_prefix_warmup_epochs_5_20epochs_bs2/beam_50_top_k_predictions.json_gen_failed_results.json'
    dirname = os.path.dirname(prediction_file)
    predictions = load_json(prediction_file)
    null_sexpr = []
    sparql_convertion_error = []
    denormalization_error = []
    structure_error = []
    entity_error = []
    relation_error = []
    others = []

    for pred in predictions:
        qid = pred["qid"]
        if pred["gt_sexpr"] == "null":
            null_sexpr.append(qid)
            continue
        top1_normed = pred["pred"]["predictions"][0]
        top1_denormed = pred["denormed_pred"][0]
        if top1_normed == pred["gt_normed_sexpr"] and top1_denormed != pred["gt_sexpr"]:
            denormalization_error.append(qid)
            continue
        if top1_normed == pred["gt_normed_sexpr"] and top1_denormed == pred["gt_sexpr"]:
            sparql_convertion_error.append(qid)
            continue

        golden_structure = get_all_masked(pred["gt_sexpr"])
        top1_denormed_structure = get_all_masked(top1_denormed)
        if golden_structure != top1_denormed_structure:
            structure_error.append(qid)
            continue

        golden_relation_maksed = get_relation_literal_masked(pred["gt_sexpr"])
        top1_denormed_relation_masked = get_relation_literal_masked(top1_denormed)
        if golden_relation_maksed != top1_denormed_relation_masked:
            entity_error.append(qid)
            continue

        golden_entity_masked = get_entity_literal_masked(pred["gt_sexpr"])
        top1_denormed_entity_masked = get_entity_literal_masked(top1_denormed)
        if golden_entity_masked != top1_denormed_entity_masked:
            relation_error.append(qid)
            continue
        others.append(qid)
    
    with open(os.path.join(dirname, 'error_analyis_generation_failed_cases.txt'), 'w') as f:
        f.write('Total length: {}\n'.format(len(predictions)))
        f.write('golden S-expression is null: {} {}\n'.format(len(null_sexpr), null_sexpr))
        f.write('sparql convertion error: {} {}\n'.format(len(sparql_convertion_error), sparql_convertion_error))
        f.write('denormalization error: {} {}\n'.format(len(denormalization_error), denormalization_error))
        f.write('structure error: {} {}\n'.format(len(structure_error), structure_error))
        f.write('entity error: {} {}\n'.format(len(entity_error), entity_error))
        f.write('relation error: {} {}\n'.format(len(relation_error), relation_error))
        f.write('others: {} {}\n'.format(len(others), others))


def error_analysis_on_generation_success_cases():
    """
    For questions with executable logical forms, but failed to answer question correctly, analyse the S-expression being executed
    Error types 
        - golden S-expression is "null"
        - denormaliztion error
        - S-expression conversion error (when both normed and denormed S-expression is consistent with golden, but failed to generate executable)
        - structure error
        - entity error
        - relation error
    """
    prediction_file = 'exps/final/WebQSP_relation_entity_concat_add_prefix_warmup_epochs_5_20epochs_bs2/beam_50_top_k_predictions.json_gen_sexpr_results.json_new.json'
    dirname = os.path.dirname(prediction_file)
    predictions = load_json(prediction_file)
    null_sexpr = []
    sparql_convertion_error = []
    denormalization_error = []
    structure_error = []
    entity_error = []
    relation_error = []
    others = []

    for pred in predictions:
        qid = pred["qid"]
        # 筛选 acc != 1 的问题
        if pred["answer_acc"]:
            continue
        if pred["gt_sexpr"] == "null":
            null_sexpr.append(qid)
            continue
        execute_index = pred["execute_index"]
        pred_normed = pred["pred"]["predictions"][execute_index]
        pred_denormed = pred["denormed_pred"][execute_index]
        if pred_normed == pred["gt_normed_sexpr"] and pred_denormed != pred["gt_sexpr"]:
            denormalization_error.append(qid)
            continue
        if pred_normed == pred["gt_normed_sexpr"] and pred_denormed == pred["gt_sexpr"]:
            sparql_convertion_error.append(qid)
            continue

        golden_structure = get_all_masked(pred["gt_sexpr"])
        pred_denormed_structure = get_all_masked(pred_denormed)
        if golden_structure != pred_denormed_structure:
            structure_error.append(qid)
            continue

        golden_relation_maksed = get_relation_literal_masked(pred["gt_sexpr"])
        pred_denormed_relation_masked = get_relation_literal_masked(pred_denormed)
        if golden_relation_maksed != pred_denormed_relation_masked:
            entity_error.append(qid)
            continue

        golden_entity_masked = get_entity_literal_masked(pred["gt_sexpr"])
        pred_denormed_entity_masked = get_entity_literal_masked(pred_denormed)
        if golden_entity_masked != pred_denormed_entity_masked:
            relation_error.append(qid)
            continue
        others.append(qid)
    
    total_len = len([pred for pred in predictions if not pred["answer_acc"]])

    with open(os.path.join(dirname, 'error_analyis_generation_success_cases.txt'), 'w') as f:
        f.write('Total length: {}\n'.format(total_len))
        f.write('golden S-expression is null: {} {}\n'.format(len(null_sexpr), null_sexpr))
        f.write('sparql convertion error: {} {}\n'.format(len(sparql_convertion_error), sparql_convertion_error))
        f.write('denormalization error: {} {}\n'.format(len(denormalization_error), denormalization_error))
        f.write('structure error: {} {}\n'.format(len(structure_error), structure_error))
        f.write('entity error: {} {}\n'.format(len(entity_error), entity_error))
        f.write('relation error: {} {}\n'.format(len(relation_error), relation_error))
        f.write('others: {} {}\n'.format(len(others), others))


def get_date_time_error():
    predictions = load_json('exps/final/WebQSP_relation_entity_concat_add_prefix_warmup_epochs_5_20epochs_bs2/beam_50_top_k_predictions.json_gen_failed_results.json')
    ids = []
    for pred in predictions:
        if '^^http://www.w3.org/2001/XMLSchema#dateTime' not in pred["gt_sexpr"] and pred["gt_sexpr"] != "null":
            if any(['^^http://www.w3.org/2001/XMLSchema#dateTime' in item for item in pred["denormed_pred"]]):
                ids.append(pred["qid"])
    print(ids)


def check_candidate_relations(split):
    dataset = load_json('data/WebQSP/final/merged/WebQSP_{}.json'.format(split))
    dataset = {data["ID"]: data for data in dataset}
    sorted_logits = load_json('/home3/xwu/workspace/QDT2SExpr/CWQ/data/WebQSP/rel_match/sorted_results/richRelation_2hopValidation_richEntity_crossEntropyLoss_top100_1parse/ep6/WebQSP_{}_cand_rel_logits_sorted.json'.format(split))
    for qid in dataset:
        cand_relation_list = [item[0:2] for item in dataset[qid]["cand_relation_list"]]
        logits_list = sorted_logits[qid][:10]
        assert cand_relation_list == logits_list, print(qid)


if __name__=='__main__':
    # error_analysis_on_generation_failed_cases()
    # error_analysis_on_generation_success_cases()
    # a = '( ARGMAX ( AND ( JOIN [ common, topic, notable types ] [ College/University ] ) ( JOIN ( R [ education, education, institution ] ) ( JOIN ( R [ people, person, education ] ) [ Charles R. Drew ] ) ) ) ( JOIN [ education, university, number of undergraduates ] [ measurement unit, dated integer, number ] ) )'
    # b = '( ARGMAX ( AND ( JOIN [ common, topic, notable types ] [ College/University ] ) ( JOIN ( R [ education, education, institution ] ) ( JOIN ( R [ people, person, education ] ) [ Charles Drew ] ) ) ) ( JOIN [ education, university, number of undergraduates ] [ measurement unit, dated integer, number ] ) )'
    # print(a == b)
    # get_date_time_error()
    for split in ['train', 'test']:
        print(split)
        check_candidate_relations(split)