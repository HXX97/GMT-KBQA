from components.utils import dump_json, load_json
import numpy as np
"""
记录一下当前的进展
    - CWQ 上，使用学长生成的实体消岐结果，会导致 0.5 左右的提升；
    - 使用我写的规则（xwu_test_get_merged_disambiguated_entities），得到的实体消岐结果，和原本的差距不大（0.2 左右）
    
    - CWQ 上，使用学长生成的候选实体重新训练最终模型，导致了 1.7 的下降
    - 正在跑把 候选实体 的 label 更新之后的模型，看看效果如何
        - 效果好了很多，在正常波动范围内
        - 不一样的 label 来自于 ELQ, 对于label 相同的实体，加了一些补充说明，比如 Tom(singer), Tom(boxer)

    - 对于 CWQ 原来的候选实体（应该是我跑的 FACC1 消岐模型）；
        - 我跑通了从 _elq 文件和 _FACC1_disamb 文件开始的流程，生成完全一致的结果
        - _elq 和 _FACC1_disamb 文件的生成？
        - 可能可以考虑把这部分结果和生成的代码传上去
    
    - _elq 文件是否存在不同？
        - test 差了 45 条，原来的文件中这45条似乎都是空的
        - train 一条都不差
    - facc1_disamb 文件是否存在不同?
        - 不考虑顺序， train 差了 8 条， test 差 1 条
        - 考虑顺序的话, train 差了 13642， test 差了 1813
        - 比较 facc1_disamb 与新生成的 facc1_unranked 不考虑顺序是否相同 -- test 完全一致
        - facc1_disamb 与新生成的 facc1_unranked 结合之前的消岐结果 -- test 完全一致
    - 结论就是使用我训练的消岐模型，除了 ELQ 的 test 少了45条链接结果，其他都是没问题的
    - 复现的 merged_data 结果，test 有13条顺序不一样的（不考虑顺序是完全一样的），应该没什么问题，可能是 merge 两种结果时的微小不同
    - 确认了是没问题的，可以另外弄一个文件夹，生成我的实体链接的一系列文件

消岐后的实体链接结果的复现
    - 见`/home2/xxhu/QDT2SExpr/CWQ/data/linking_results`下的数据和 `/home2/xxhu/QDT2SExpr/CWQ/get_final_entity_linking.py`文件
    - 对于 facc1, 只取 logits > 0 的部分
    - 对于 elq, 只取 logits > -1.5 的部分

WebQSP:
    - ELQ 的结果是完全一样的
    - FACC1 的排序结果不一样(train 1256, test 587), 召回结果基本一样（训练集测试集各差一个）
    - 先跑一下新 merged data 的训练吧，居然报错 shape '[2, 10, -1]' is invalid for input of size 10， 有数据的关系不足10个？
"""


def xwu_test_get_merged_disambiguated_entities(dataset, split):
    cand_ent_dir = f"data/{dataset}/entity_retrieval/candidate_entities"
    elq_cand_ent_file = f"{cand_ent_dir}/{dataset}_{split}_cand_entities_elq.json"
    facc1_cand_ent_file = f"{cand_ent_dir}/{dataset}_{split}_cand_entities_facc1.json"
    disamb_ent_file = f"data/{dataset}/entity_retrieval/xwu_test/{dataset}_merged_{split}_disamb_entities.json"

    elq_cand_ents = load_json(elq_cand_ent_file)
    facc1_cand_ents = load_json(facc1_cand_ent_file)

    # entities linked and ranked by facc1
    facc1_disamb_ents = {}
    for qid,cand_ents in facc1_cand_ents.items():
        mention_cand_map = {}
        for ent in cand_ents:
            if ent['mention'] not in mention_cand_map:
                mention_cand_map[ent['mention']]=ent            
        facc1_disamb_ents[qid] = [ent for (_,ent) in mention_cand_map.items()]

    # entities linked and ranked by elq
    elq_disamb_ents = {}        
    for qid,cand_ents in elq_cand_ents.items():
        mention_cand_map = {}
        for ent in cand_ents:
            if ent['mention'] not in mention_cand_map:
                mention_cand_map[ent['mention']]=ent
        
        elq_disamb_ents[qid] = [ent for (_,ent) in mention_cand_map.items()]

    disamb_ent_map = {}

    # 为了公平对比(和模型生成的消岐结果对比)
    # 优先选择 facc1 的实体
    # 当 facc1 一个实体都没链接到的时候，才考虑 elq 里头的实体；
    # 或者是 elq 里头出现了 label 与 FACC1 相同，但 mid 不同的情况
    for qid in elq_disamb_ents:
        disamb_entities = {}
        linked_entity_id_set = set()
        facc1_entities = facc1_disamb_ents[qid]
        elq_entities = elq_disamb_ents[qid]

        if len(facc1_entities) == 0:
            for ent in elq_entities:
                disamb_entities[ent['label']]={
                    "id":ent["id"],
                    "label":ent["label"],
                    "mention":ent["mention"],
                    "perfect_match":ent["perfect_match"]
                }
                linked_entity_id_set.add(ent['id'])
        else:
            for ent in facc1_entities:
                disamb_entities[ent['label']]={
                    "id":ent["id"],
                    "label":ent["label"],
                    "mention":ent["mention"],
                    "perfect_match":ent["perfect_match"]
                }
                linked_entity_id_set.add(ent['id'])
            
            for ent in elq_entities:
                if ent['id'] not in linked_entity_id_set: 
                    if ent['label'] in disamb_entities: # same label, different mid
                        disamb_entities[ent['label']]={
                            "id":ent["id"],
                            "label":ent["label"],
                            "mention":ent["mention"],
                            "perfect_match":ent["perfect_match"]
                        }
                        linked_entity_id_set.add(ent['id'])
        
        disamb_entities = [ent for (_,ent) in disamb_entities.items()]
        disamb_ent_map[qid] = disamb_entities

    dump_json(disamb_ent_map, disamb_ent_file, indent=4)


def check_disambiguated_cand_entity():
    merged_dataset = load_json('data/CWQ/generation/merged/CWQ_train.json')
    linking_res = load_json('data/CWQ/entity_retrieval/disamb_entities/CWQ_merged_train_disamb_entities.json')
    for item in merged_dataset:
        qid = item["ID"]
        ids = [ent['id'] for ent in item["disambiguated_cand_entity"]]
        disamb_ids = [ent['id'] for ent in linking_res[qid]]
        assert set(ids) == set(disamb_ids), print(qid)


def error_analysis():
    new_gen_success_results = load_json('exps/CWQ_relation_entity_concat_add_prefix_warmup_epochs_5_15epochs/beam_50_top_k_predictions.json_gen_sexpr_results.json_new.json')
    new_gen_failed_results = load_json('exps/CWQ_relation_entity_concat_add_prefix_warmup_epochs_5_15epochs/beam_50_top_k_predictions.json_gen_failed_results.json')
    prev_gen_success_results = load_json('/home3/xwu/workspace/QDT2SExpr/CWQ/exps/final/CWQ_relation_entity_concat_add_prefix_warmup_epochs_5_20epochs/15epoch/beam_50_top_k_predictions.json_gen_sexpr_results.json_new.json')
    prev_gen_failed_results = load_json('/home3/xwu/workspace/QDT2SExpr/CWQ/exps/final/CWQ_relation_entity_concat_add_prefix_warmup_epochs_5_20epochs/15epoch/beam_50_top_k_predictions.json_gen_failed_results.json')
    new_gen_success_results = {item["qid"]: item for item in new_gen_success_results}
    prev_gen_success_results = {item["qid"]: item for item in prev_gen_success_results}
    new_gen_failed_results = {item["qid"]: item for item in new_gen_failed_results}
    prev_gen_failed_results = {item["qid"]: item for item in prev_gen_failed_results}
    diff_qids = set()

    for qid in prev_gen_success_results:
        if qid not in new_gen_success_results:
            diff_qids.add(qid)
        else:
            if new_gen_success_results[qid]["f1"] < prev_gen_success_results[qid]["f1"]:
                diff_qids.add(qid)
    for qid in new_gen_failed_results:
        if qid not in prev_gen_failed_results:
            diff_qids.add(qid)
    print(len(diff_qids), list(diff_qids))


def compared_merged_data(split):
    prev_merged_data = load_json(f'data/CWQ/generation/merged_old/CWQ_{split}.json')
    new_merged_data = load_json(f'data/CWQ/generation/merged/CWQ_{split}.json')
    assert len(prev_merged_data) == len(new_merged_data), print(len(prev_merged_data), len(new_merged_data))
    diff_qids = []
    for (prev, new) in zip(prev_merged_data, new_merged_data):
        prev_relations = [item[0] for item in prev["cand_relation_list"]]
        prev_entities = set(item["id"] for item in prev["cand_entity_list"][:5])

        new_relations = [item[0] for item in new["cand_relation_list"]]
        new_entities = set(item["id"] for item in new["cand_entity_list"][:5])

        if prev_entities != new_entities:
            diff_qids.append(prev["ID"])
    print(len(diff_qids))

def get_entity_label_diff(split):
    prev_merged_data = load_json(f'data/CWQ/generation/merged/CWQ_{split}.json')
    diff = dict()
    for item in prev_merged_data:
        gold_entity_map = item["gold_entity_map"]
        ents = item["cand_entity_list"]
        for ent in ents:
            if ent['id'] in gold_entity_map and gold_entity_map[ent['id']] != ent['label']:
                diff[ent['id']] = [ent['label'], gold_entity_map[ent['id']]]
    dump_json(diff, f'{split}_label_diff.json')


def compare_elq_result(split):
    xwu_results = load_json(f'data/WebQSP/entity_retrieval/candidate_entities_xwu/WebQSP_{split}_cand_entities_elq.json')
    xxhu_results = load_json(f'data/WebQSP/entity_retrieval/candidate_entities/WebQSP_{split}_cand_entities_elq.json')
    assert len(xwu_results) == len(xxhu_results), print(len(xwu_results), len(xxhu_results))
    diff_qids = set()
    for qid in xwu_results:
        assert qid in xxhu_results, print(qid)
        if xwu_results[qid] != xxhu_results[qid]:
            diff_qids.add(qid)
    print(len(diff_qids), list(diff_qids))


def compare_facc1_ranked_result(split):
    xwu_results = load_json(f'data/CWQ/entity_retrieval/candidate_entities_xwu/CWQ_{split}_entities_facc1_unranked.json')
    xxhu_results = load_json(f'/home2/xxhu/QDT2SExpr/CWQ/data/CWQ_{split}_entities.json')
    assert len(xwu_results) == len(xxhu_results), print(len(xwu_results), len(xxhu_results))
    sequence_diff_qids = set()
    content_diff_qids = set()
    for qid in xwu_results:
        assert qid in xxhu_results, print(qid)
        xwu_ents = [item for mention_res in xwu_results[qid] for item in mention_res]
        xwu_ents = sorted(xwu_ents, key=lambda d: d.get('disamb_logits', 1.0), reverse=True) # FACC1 中没有 disamb_logits 往往是因为只召回了一个实体，所以给个比较高的分数
        xwu_ids = [item['id'] for item in xwu_ents]
        xxhu_ents = [item for mention_res in xxhu_results[qid] for item in mention_res]
        xxhu_ids = [item['id'] for item in xxhu_ents]
        if xwu_ids != xxhu_ids:
            sequence_diff_qids.add(qid)
        if set(xwu_ids) != set(xxhu_ids):
            content_diff_qids.add(qid)
    # print(len(sequence_diff_qids))
    print(len(content_diff_qids))


def compare_merged_el_result(split):
    prev_result = load_json(f'data/CWQ/entity_retrieval/candidate_entities_xwu/CWQ_{split}_merged_cand_entities_elq_facc1.json')
    new_result = load_json(f'data/CWQ/entity_retrieval/candidate_entities/CWQ_{split}_merged_cand_entities_elq_facc1.json')
    assert len(prev_result) == len(new_result), print(len(prev_result), len(new_result))
    sequence_diff_qids = set()
    content_diff_qids = set()
    for qid in prev_result:
        assert qid in new_result, print(qid)
        prev_ids = [item['id'] for item in prev_result[qid] if 'mention' in item]
        new_ids = [item['id'] for item in new_result[qid] if 'mention' in item]
        if prev_ids != new_ids:
            sequence_diff_qids.add(qid)
        if set(prev_ids) != set(new_ids):
            print(set(prev_ids))
            print(set(new_ids))
            content_diff_qids.add(qid)
        
    # print(len(sequence_diff_qids))
    # print(len(content_diff_qids))

def compare_merged_el_result_including_label(split):
    # prev_result = load_json(f'/home3/xwu/workspace/QDT2SExpr/CWQ/all_data_bk/CWQ/entity_linking_0414/CWQ_{split}_merged_elq_FACC1.json')
    prev_result = load_json(f'data/WebQSP/entity_retrieval/candidate_entities_xwu_bak/WebQSP_{split}_merged_cand_entities_elq_facc1.json')
    new_result = load_json(f'data/WebQSP/entity_retrieval/candidate_entities_xwu/WebQSP_{split}_merged_cand_entities_elq_facc1.json')
    print(len(new_result))
    assert len(prev_result) == len(new_result), print(len(prev_result), len(new_result))
    sequence_diff_qids = set()
    content_diff_qids = set()
    for qid in prev_result:
        assert qid in new_result, print(qid)
        prev_ids = [(item['id'], item['label']) for item in prev_result[qid] if 'mention' in item]
        new_ids = [(item['id'], item['label']) for item in new_result[qid] if 'mention' in item]
        if prev_ids != new_ids:
            sequence_diff_qids.add(qid)
        if set(prev_ids) != set(new_ids):
            content_diff_qids.add(qid)
        
    # print(len(sequence_diff_qids), list(sequence_diff_qids))
    print(len(content_diff_qids), list(content_diff_qids))


def combine_FACC1_and_disamb(split):
    """结合 FACC1 的链接结果，和消岐后的结果"""
    facc1_res = load_json(f'data/CWQ/generation/xwu_merged_new/entity_linking_res_from_service/CWQ_{split}_entities_facc1_unranked.json')
    disamb_res = load_json(f'/home3/xwu/workspace/QDT2SExpr/CWQ/results/disamb/CWQ_{split}/prediction.json')
    new_res = dict()
    missed_count = 0

    for id in facc1_res:
        value = facc1_res[id]
        for idx in range(len(value)): # different mentions
            disamb_id = id + '#' + str(idx)
            if disamb_id not in disamb_res:
                missed_count += 1 
            else:
                for linked_idx in range(len(value[idx])):  #linked entities
                    value[idx][linked_idx]['disamb_logits'] = disamb_res[disamb_id]["logits"][linked_idx]
                # sort by disamb_logits
                value[idx].sort(key=lambda item: float(item['disamb_logits']), reverse=True)
        new_res[id] = value
    
    dump_json(new_res, 'data/CWQ/generation/xwu_merged_new/entity_linking_res_from_service/CWQ_{}_entities_FACC1_disamb.json'.format(split))
    print('missed: {}'.format(missed_count))


def compare_facc1_disamb(split):
    prev_result = load_json(f'data/WebQSP/entity_retrieval/candidate_entities_xwu/WebQSP_{split}_entities_facc1_unranked.json')
    new_result = load_json(f'/home2/xxhu/QDT2SExpr/CWQ/data/WEBQSP_{split}_entities.json')
    assert len(prev_result) == len(new_result), print(len(prev_result), len(new_result))
    diff_qids = set()
    for qid in prev_result:
        prev_ids = [item['id'] for mention_res in prev_result[qid] for item in mention_res]
        new_ids = [item['id'] for mention_res in new_result[qid] for item in mention_res]
        if prev_ids != new_ids:
            diff_qids.add(qid)
    print(len(diff_qids), list(diff_qids))


def compare_WebQSP_merged_linking_results(split):
    xwu_results = load_json(f'/home3/xwu/workspace/QDT2SExpr/CWQ/all_data_bk/WebQSP/entity_linking_0423/WEBQSP_{split}_cand_entities_facc1.json')
    xxhu_results = load_json(f'data/WebQSP/entity_retrieval/candidate_entities/WebQSP_{split}_cand_entities_facc1.json')
    assert len(xwu_results) == len(xxhu_results), print(len(xwu_results), len(xxhu_results))
    diff_qids = set()
    for qid in xwu_results:
        assert qid in xxhu_results
        xwu_ent_ids = [item['id'] for item in xwu_results[qid]] # 有 mention 就不是随机得到的
        xxhu_ent_ids = [item['id'] for item in xxhu_results[qid]]
        if set(xwu_ent_ids) != set(xxhu_ent_ids):
            diff_qids.add(qid)
    print(len(diff_qids))


def check_webqsp_merged_cand_relation_length(split):
    merged_data = load_json(f'data/WebQSP/generation/merged/WebQSP_{split}.json')
    qids = set()
    for item in merged_data:
        if item["cand_relation_list"] is None or len(item["cand_relation_list"]) < 10:
            qids.add(item["ID"])
    print(len(qids), list(qids))

def remove_item_webqsp():
    merged_data = load_json('data/WebQSP/generation/merged_corrected/WebQSP_train.json')
    print('prev_len: {}'.format(len(merged_data)))
    new_merged_data = [item for item in merged_data if len(item["cand_relation_list"]) == 10]
    print('after_len: {}'.format(len(new_merged_data)))
    dump_json(new_merged_data, 'data/WebQSP/generation/merged_corrected/WebQSP_train.json')

def compare_disamb_logits(split):
    prev_logits = load_json(f'data/CWQ/entity_retrieval/candidate_entities_xwu/disamb_results/CWQ_{split}/predict_logits.json')
    new_logits = load_json(f'data/CWQ/entity_retrieval/candidate_entities_xwu_compared/disamb_results/CWQ_{split}/predict_logits.json')
    
    key_diff = set(new_logits.keys()) - set(prev_logits.keys())
    print(key_diff)
    assert len(prev_logits) == len(new_logits), print(len(prev_logits), len(new_logits))
    diff_keys = set()
    for key in prev_logits:
        assert key in new_logits, print(key)
        prev = np.array(prev_logits[key])
        prev_indexes = np.argsort(prev)
        new = np.array(new_logits[key])
        new_indexes = np.argsort(new)
        # print(prev_indexes.tolist(), new_indexes.tolist())
        if prev_indexes.tolist() != new_indexes.tolist():
            diff_keys.add(key)
    print(len(diff_keys), list(diff_keys))


def disamb_logits_generation(split):
    prev_logits = load_json(f'data/CWQ/entity_retrieval/candidate_entities_xwu/disamb_results/CWQ_{split}/predictions.json')
    new_logits = dict()
    new_predictions = dict()
    for key in prev_logits:
        new_logits[key] = prev_logits[key]['logits']
        new_predictions[key] = prev_logits[key]['index']
    dump_json(new_logits, f'data/CWQ/entity_retrieval/candidate_entities_xwu/disamb_results/CWQ_{split}/predict_logits.json')
    dump_json(new_predictions, f'data/CWQ/entity_retrieval/candidate_entities_xwu/disamb_results/CWQ_{split}/predictions.json')


def test_merged_length():
    prev_merged = load_json('data/WebQSP/generation/merged_old/WebQSP_test.json')
    print(len(prev_merged))


def check_disambed_entities(split, filter_empty_id=False):
    prev_disambed = load_json(f'data/CWQ/entity_retrieval/merged_linking_results/merged_CWQ_{split}_linking_results.json')
    new_disambed = load_json(f'data/CWQ/entity_retrieval/disamb_entities_xwu_bak/merged_CWQ_{split}_linking_results.json')
    diff_qids = set()
    special_qids = set()
    assert len(prev_disambed) == len(new_disambed), print(len(prev_disambed), len(new_disambed))
    for qid in prev_disambed:
        assert qid in new_disambed, print(qid)
        if filter_empty_id:
            prev_keys = [key for key in prev_disambed[qid].keys() if key != '']
            new_keys = [key for key in new_disambed[qid].keys() if key != '']
        else:
            prev_keys = prev_disambed[qid].keys()
            new_keys = new_disambed[qid].keys()
        if set(prev_keys) != set(new_keys):
            diff_qids.add(qid)
        else:
            for ent_id in prev_keys:
                if prev_disambed[qid][ent_id] != new_disambed[qid][ent_id]:
                    diff_qids.add(qid)
        if len(prev_keys) == 1 and len(new_keys) == 0:
            special_qids.add(qid)
    # print(len(diff_qids), list(diff_qids))
    print(len(special_qids), list(special_qids))


def compare_facc1_ignore_sequence(split):
    """不考虑顺序"""
    new_unranked = load_json(f'data/WebQSP/entity_retrieval/candidate_entities_xwu/WebQSP_{split}_entities_facc1_unranked.json')
    prev_ranked = load_json(f'/home3/xwu/workspace/QDT2SExpr/CWQ/all_data_bk/WebQSP/entity_linking_0423/WEBQSP_{split}_cand_entities_facc1.json')
    assert len(new_unranked) == len(prev_ranked), print(len(new_unranked), len(prev_ranked))
    diff_qids = set()
    for qid in new_unranked:
        assert qid in prev_ranked, print(qid)
        new_ids = set([item['id'] for mention_res in new_unranked[qid] for item in mention_res])
        prev_ids = set([item['id'] for item in prev_ranked[qid]])
        if prev_ids != new_ids:
            diff_qids.add(qid)
    print(len(diff_qids), list(diff_qids))


def compare_facc1(split):
    """考虑顺序"""
    new_ranked = load_json(f'data/CWQ/entity_retrieval/candidate_entities_xwu/CWQ_{split}_cand_entities_facc1.json')
    prev_ranked = load_json(f'data/CWQ/entity_retrieval/candidate_entities/CWQ_{split}_cand_entities_facc1.json')
    assert len(new_ranked) == len(prev_ranked), print(len(new_ranked), len(prev_ranked))
    diff_qids = set()
    sequence_diff_qids = set()
    for qid in new_ranked:
        assert qid in prev_ranked, print(qid)
        prev_ids = [(item['id'], item['label']) for item in prev_ranked[qid]]
        new_ids = [(item['id'], item['label']) for item in new_ranked[qid]]
        if prev_ids != new_ids:
            diff_qids.add(qid)
        if set(prev_ids) != set(new_ids):
            sequence_diff_qids.add(qid)
    print(len(diff_qids))
    print(len(sequence_diff_qids), list(sequence_diff_qids))


if __name__=='__main__':
    # xwu_test_get_merged_disambiguated_entities('CWQ', 'test')
    # check_disambiguated_cand_entity()
    # error_analysis()
    # compared_merged_data('train')
    # get_entity_label_diff('train')
    # for split in ['train', 'test']:
    #     print(split)
    #     compare_elq_result(split)
    # compare_facc1_ranked_result('test')
    # combine_FACC1_and_disamb('test')
    # compare_facc1_disamb('test')
    # compare_WebQSP_merged_linking_results('test')
    # check_webqsp_merged_cand_relation_length('train')
    # remove_item_webqsp()
    compare_disamb_logits('dev')
    # compare_merged_el_result('train')
    # compare_merged_el_result_including_label('test')
    # disamb_logits_generation('dev')
    # test_merged_length()
    # check_disambed_entities('test')
    # compare_facc1_ignore_sequence('train')
    # compare_facc1('test')