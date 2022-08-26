import math
import os
import json
import re

# relations that regular expression cannot cover
extra_relations_prefix = ['topic_server.schemastaging_corresponding_entities_type', 'topic_server.webref_cluster_members_type', 'topic_server.population_number']
literal_mask = '[LIT]'
entity_mask = '[ENT]'
relation_mask = '[REL]'


def mask_relations(sexpr):
    sexpr = sexpr.replace('(',' ( ').replace(')',' ) ')
    toks = sexpr.split(' ')
    toks = [x for x in toks if len(x)]
    for idx in range(len(toks)):
        t = toks[idx].strip()
        if t in extra_relations_prefix:
            toks[idx] = relation_mask
        elif re.match(r"[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*",t) or re.match(r"[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*",t):
            toks[idx] = relation_mask
    return " ".join(toks)


def mask_entities(sexpr):
    sexpr = sexpr.replace('(',' ( ').replace(')',' ) ')
    toks = sexpr.split(' ')
    toks = [x.replace('\t.','') for x in toks if len(x)]
    for idx in range(len(toks)):
        t = toks[idx].strip()
        if t.startswith('m.') or t.startswith('g.'):
            toks[idx] = entity_mask
    return " ".join(toks)


def mask_literals(sexpr):
    operators_list = ['(', ')', 'AND', 'COUNT', 'R', 'JOIN', 'ARGMAX', 'ARGMIN', 'lt', 'gt', 'le', 'ge', 'TC', literal_mask, entity_mask, relation_mask]
    sexpr = sexpr.replace('(',' ( ').replace(')',' ) ')
    toks = sexpr.split(' ')
    toks = [x for x in toks if len(x)]
    for idx in range(len(toks)):
        t = toks[idx].strip()
        if (
            t not in operators_list 
            and not (t.startswith('m.') or t.startswith('g.')) 
            and not t in extra_relations_prefix 
            and not re.match(r"[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*",t) or re.match(r"[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*",t)
        ):
            toks[idx] = literal_mask

    return " ".join(toks)


def mask_special_literals(sexpr):
    """
    Special literals like \"as Robby Ray Stewart\"@en
    Has whitespace inside literal, which may influence tokenization
    """
    pattern = "\".*?\"(@en)?"
    sexpr = re.sub(pattern, literal_mask, sexpr)
    return sexpr


def get_all_masked(sexpr):
    sexpr = mask_special_literals(sexpr)
    sexpr = mask_literals(mask_entities(mask_relations(sexpr)))
    return sexpr

def get_relation_literal_masked(sexpr):
    sexpr = mask_special_literals(sexpr)
    sexpr = mask_literals(mask_relations(sexpr))
    return sexpr

def get_entity_literal_masked(sexpr):
    sexpr = mask_special_literals(sexpr)
    sexpr = mask_literals(mask_entities(sexpr))
    return sexpr

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


def error_analysis_on_generation_success_cases(prediction_path, original_data_path, evaluation_path, dataset_type='CWQ'):
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
    predictions = load_json(prediction_path)
    original_data = load_json(original_data_path)
    evaluation_results = load_json(evaluation_path)
    if dataset_type.lower() == 'cwq':
        original_data = {example["ID"]: example["question"] for example in original_data}
    elif dataset_type.lower() == 'webqsp':
        original_data = original_data["Questions"]
        original_data = {example["QuestionId"]: example["ProcessedQuestion"] for example in original_data}
    null_sexpr = []
    sparql_convertion_error = []
    denormalization_error = []
    structure_error = []
    entity_error = []
    relation_error = []
    others = []

    for (pred, eva) in zip(predictions, evaluation_results):
        assert pred['qid'] == eva["qid"], print(pred['qid'])
        qid = pred["qid"]
        if math.isclose(eva["f1"], 1.0): # answer correctly
            continue
        # compare final executed S-Expression
        execute_index = pred["execute_index"]
        pred_normed = pred["pred"]["predictions"][execute_index]
        pred_denormed = pred["denormed_pred"][execute_index]
        golden_structure = get_all_masked(pred["gt_sexpr"])
        pred_denormed_structure = get_all_masked(pred_denormed)
        if pred["gt_sexpr"] == "null":
            null_sexpr.append({
                'qid': qid,
                'question': original_data[qid],
                'golden_sexpr': pred["gt_sexpr"],
                'golden_sexpr_masked': golden_structure,
                'pred_sexpr': pred_denormed,
                'pred_sexpr_masked': pred_denormed_structure
            })
            continue
        if pred_normed == pred["gt_normed_sexpr"] and pred_denormed != pred["gt_sexpr"]:
            denormalization_error.append({
                'qid': qid,
                'question': original_data[qid],
                'golden_sexpr': pred["gt_sexpr"],
                'golden_sexpr_masked': golden_structure,
                'pred_sexpr': pred_denormed,
                'pred_sexpr_masked': pred_denormed_structure
            })
            continue
        if pred_normed == pred["gt_normed_sexpr"] and pred_denormed == pred["gt_sexpr"]:
            sparql_convertion_error.append({
                'qid': qid,
                'question': original_data[qid],
                'golden_sexpr': pred["gt_sexpr"],
                'golden_sexpr_masked': golden_structure,
                'pred_sexpr': pred_denormed,
                'pred_sexpr_masked': pred_denormed_structure
            })
            continue
        if golden_structure != pred_denormed_structure:
            structure_error.append({
                'qid': qid,
                'question': original_data[qid],
                'golden_sexpr': pred["gt_sexpr"],
                'golden_sexpr_masked': golden_structure,
                'pred_sexpr': pred_denormed,
                'pred_sexpr_masked': pred_denormed_structure
            })
            continue
        

        golden_relation_maksed = get_relation_literal_masked(pred["gt_sexpr"])
        pred_denormed_relation_masked = get_relation_literal_masked(pred_denormed)
        if golden_relation_maksed != pred_denormed_relation_masked:
            entity_error.append({
                'qid': qid,
                'question': original_data[qid],
                'golden_sexpr': pred["gt_sexpr"],
                'golden_relation_maksed': golden_relation_maksed,
                'pred_sexpr': pred_denormed,
                'pred_relation_masked': pred_denormed_relation_masked
            })
            continue

        golden_entity_masked = get_entity_literal_masked(pred["gt_sexpr"])
        pred_denormed_entity_masked = get_entity_literal_masked(pred_denormed)
        if golden_entity_masked != pred_denormed_entity_masked:
            relation_error.append({
                'qid': qid,
                'question': original_data[qid],
                'golden_sexpr': pred["gt_sexpr"],
                'golden_entity_masked': golden_entity_masked,
                'pred_sexpr': pred_denormed,
                'pred_entity_masked': pred_denormed_entity_masked
            })
            continue
        others.append({
            'qid': qid,
            'question': original_data[qid],
            'golden_sexpr': pred["gt_sexpr"],
            'pred_sexpr': pred_denormed,
        })
    
    total_len = len([eval for eval in evaluation_results if not math.isclose(eval["f1"], 1.0)])
    print(total_len)

    dump_json(
        {
            'total_length': total_len,
            'null_sexpr': {'length': len(null_sexpr), 'cases': null_sexpr},
            'sparql_convertion_error': {'length': len(sparql_convertion_error), 'cases': sparql_convertion_error},
            'denormalization_error': {'length': len(denormalization_error), 'cases': denormalization_error},
            'structure_error': {'length': len(structure_error), 'cases': structure_error},
            'entity_error': {'length': len(entity_error), 'cases': entity_error},
            'relation_error': {'length': len(relation_error), 'cases': relation_error},
            'others': {'length': len(others), 'cases': others},
        },
        f'{prediction_path}_error_analyis_generation_success_cases.json'
    )

def error_analysis_on_generation_failed_cases(prediction_path, original_data_path, dataset_type='CWQ'):
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
    predictions = load_json(prediction_path)
    original_data = load_json(original_data_path)
    if dataset_type.lower() == 'cwq':
        original_data = {example["ID"]: example["question"] for example in original_data}
    elif dataset_type.lower() == 'webqsp':
        original_data = original_data["Questions"]
        original_data = {example["QuestionId"]: example["ProcessedQuestion"] for example in original_data}

    null_sexpr = []
    sparql_convertion_error = []
    denormalization_error = []
    structure_error = []
    entity_error = []
    relation_error = []
    others = []

    for pred in predictions:
        qid = pred["qid"]
        top1_normed = pred["pred"]["predictions"][0]
        top1_denormed = pred["denormed_pred"][0]
        golden_structure = get_all_masked(pred["gt_sexpr"])
        top1_denormed_structure = get_all_masked(top1_denormed)
        if pred["gt_sexpr"] == "null":
            null_sexpr.append({
                'qid': qid,
                'question': original_data[qid],
                'golden_sexpr': pred["gt_sexpr"],
                'pred_sexpr': top1_denormed
            })
            continue
       
        if top1_normed == pred["gt_normed_sexpr"] and top1_denormed != pred["gt_sexpr"]:
            denormalization_error.append({
                'qid': qid,
                'question': original_data[qid],
                'golden_sexpr': pred["gt_sexpr"],
                'golden_sexpr_masked': golden_structure,
                'pred_sexpr': top1_denormed,
                'pred_sexpr_masked': top1_denormed_structure
            })
            continue
        if top1_normed == pred["gt_normed_sexpr"] and top1_denormed == pred["gt_sexpr"]:
            sparql_convertion_error.append({
                'qid': qid,
                'question': original_data[qid],
                'golden_sexpr': pred["gt_sexpr"],
                'golden_sexpr_masked': golden_structure,
                'pred_sexpr': top1_denormed,
                'pred_sexpr_masked': top1_denormed_structure
            })
            continue

        if golden_structure != top1_denormed_structure:
            structure_error.append({
                'qid': qid,
                'question': original_data[qid],
                'golden_sexpr': pred["gt_sexpr"],
                'golden_sexpr_masked': golden_structure,
                'pred_sexpr': top1_denormed,
                'pred_sexpr_masked': top1_denormed_structure
            })
            continue

        golden_relation_maksed = get_relation_literal_masked(pred["gt_sexpr"])
        top1_denormed_relation_masked = get_relation_literal_masked(top1_denormed)
        if golden_relation_maksed != top1_denormed_relation_masked:
            entity_error.append({
                'qid': qid,
                'question': original_data[qid],
                'golden_sexpr': pred["gt_sexpr"],
                'golden_relation_maksed': golden_relation_maksed,
                'pred_sexpr': top1_denormed,
                'pred_relation_masked': top1_denormed_relation_masked
            })
            continue

        golden_entity_masked = get_entity_literal_masked(pred["gt_sexpr"])
        top1_denormed_entity_masked = get_entity_literal_masked(top1_denormed)
        if golden_entity_masked != top1_denormed_entity_masked:
            relation_error.append({
                'qid': qid,
                'question': original_data[qid],
                'golden_sexpr': pred["gt_sexpr"],
                'golden_entity_masked': golden_entity_masked,
                'pred_sexpr': top1_denormed,
                'pred_entity_masked': top1_denormed_entity_masked
            })
            continue
        others.append({
            'qid': qid,
            'question': original_data[qid],
            'golden_sexpr': pred["gt_sexpr"],
            'pred_sexpr': top1_denormed,
        })
    total_len = len(null_sexpr) + len(sparql_convertion_error) + len(denormalization_error) + len(structure_error) + len(entity_error) + len(relation_error) + len(others)
    print('total_len: {}'.format(total_len))
    assert len(predictions) == total_len, print(len(predictions), total_len)
    dump_json(
        {
            'total_length': total_len,
            'null_sexpr': {'length': len(null_sexpr), 'cases': null_sexpr},
            'sparql_convertion_error': {'length': len(sparql_convertion_error), 'cases': sparql_convertion_error},
            'denormalization_error': {'length': len(denormalization_error), 'cases': denormalization_error},
            'structure_error': {'length': len(structure_error), 'cases': structure_error},
            'entity_error': {'length': len(entity_error), 'cases': entity_error},
            'relation_error': {'length': len(relation_error), 'cases': relation_error},
            'others': {'length': len(others), 'cases': others},
        },
        f'{prediction_path}_error_analyis_generation_failed_cases.json'
    )

if __name__=='__main__':
    error_analysis_on_generation_success_cases(
        'exps/CWQ_GMT_KBQA/beam_50_test_4_top_k_predictions.json_gen_sexpr_results.json_new.json',
        'data/CWQ/origin/ComplexWebQuestions_test.json',
        'exps/CWQ_GMT_KBQA/beam_50_test_4_top_k_predictions.json_gen_sexpr_results.json_new.json',
        "CWQ"
    )
    error_analysis_on_generation_success_cases(
        'exps/WebQSP_GMT_KBQA/beam_50_test_2_top_k_predictions.json_gen_sexpr_results.json',
        'data/WebQSP/origin/WebQSP.test.json',
        'exps/WebQSP_GMT_KBQA/beam_50_test_2_top_k_predictions.json_gen_sexpr_results_official_format.json_new.json',
        "WebQSP"
    )
    error_analysis_on_generation_failed_cases(
        'exps/CWQ_GMT_KBQA/beam_50_test_4_top_k_predictions.json_gen_failed_results.json',
        'data/CWQ/origin/ComplexWebQuestions_test.json',
        "CWQ"
    )
    error_analysis_on_generation_failed_cases(
        'exps/WebQSP_GMT_KBQA/beam_50_test_2_top_k_predictions.json_gen_failed_results.json',
        'data/WebQSP/origin/WebQSP.test.json',
        "WebQSP"
    )