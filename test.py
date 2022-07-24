from collections import defaultdict
import collections
from components.utils import dump_json, load_json
import numpy as np
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
import torch
import logging
import faiss
logger = logging.getLogger()
class DenseIndexer(object):
    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []
        self.index = None

    def index_data(self, data: np.array):
        raise NotImplementedError

    def search_knn(self, query_vectors: np.array, top_docs: int):
        raise NotImplementedError

    def serialize(self, index_file: str):
        logger.info("Serializing index to %s", index_file)
        faiss.write_index(self.index, index_file)

    def deserialize_from(self, index_file: str):
        logger.info("Loading index from %s", index_file)
        self.index = faiss.read_index(index_file)
        logger.info(
            "Loaded index of type %s and size %d", type(self.index), self.index.ntotal
        )


# DenseFlatIndexer does exact search
class DenseFlatIndexer(DenseIndexer):
    def __init__(self, vector_sz: int = 1, buffer_size: int = 50000):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)
        self.index = faiss.IndexFlatIP(vector_sz)

    def index_data(self, data: np.array):
        n = len(data)
        # indexing in batches is beneficial for many faiss index types
        logger.info("Indexing data, this may take a while.")
        cnt = 0
        for i in range(0, n, self.buffer_size):
            vectors = [np.reshape(t, (1, -1)) for t in data[i : i + self.buffer_size]]
            vectors = np.concatenate(vectors, axis=0)
            self.index.add(vectors)
            cnt += self.buffer_size

        logger.info("Total data indexed %d", n)

    def search_knn(self, query_vectors, top_k):
        scores, indexes = self.index.search(query_vectors, top_k)
        return scores, indexes
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
    xwu_results = load_json(f'data/WebQSP/entity_retrieval/candidate_entities/WebQSP_{split}_cand_entities_elq.json')
    xxhu_results = load_json(f'data/WebQSP/entity_retrieval/candidate_entities_xxhu/WebQSP_{split}_cand_entities_elq.json')
    assert len(xwu_results) == len(xxhu_results), print(len(xwu_results), len(xxhu_results))
    diff_qids = set()
    for qid in xwu_results:
        assert qid in xxhu_results, print(qid)
        if xwu_results[qid] != xxhu_results[qid]:
            diff_qids.add(qid)
    print(len(diff_qids), list(diff_qids))


def compare_facc1_ranked_result(split):
    new_results = load_json(f'data/WebQSP/entity_retrieval/candidate_entities/WebQSP_{split}_cand_entities_facc1.json')
    prev_results = load_json(f'data/WebQSP/entity_retrieval/candidate_entities_xwu/WebQSP_{split}_cand_entities_facc1.json')
    assert len(new_results) == len(prev_results), print(len(new_results), len(prev_results))
    sequence_diff_qids = set()
    content_diff_qids = set()
    for qid in new_results:
        assert qid in prev_results, print(qid)
        xwu_ids = [item['id'] for item in new_results[qid]]
        xxhu_ids = [item['id'] for item in prev_results[qid]]
        if xwu_ids != xxhu_ids:
            sequence_diff_qids.add(qid)
        if set(xwu_ids) != set(xxhu_ids):
            content_diff_qids.add(qid)
    print(len(sequence_diff_qids))
    print(len(content_diff_qids))


def compare_merged_el_result(split):
    prev_result = load_json(f'/home3/xwu/workspace/QDT2SExpr/CWQ/all_data_bk/CWQ/entity_linking_0414/CWQ_{split}_merged_elq_FACC1.json')
    new_result = load_json(f'data/CWQ/entity_retrieval/candidate_entities/CWQ_{split}_merged_cand_entities_elq_facc1.json')
    assert len(prev_result) == len(new_result), print(len(prev_result), len(new_result))
    sequence_diff_qids = set()
    content_diff_qids = set()
    for qid in prev_result:
        assert qid in new_result, print(qid)
        prev_ids = [item['id'] for item in prev_result[qid] if 'mention' in item] # without 'mention' --> randomly sampled
        new_ids = [item['id'] for item in new_result[qid] if 'mention' in item]
        if prev_ids != new_ids:
            sequence_diff_qids.add(qid)
        if set(prev_ids) != set(new_ids):
            print(set(prev_ids))
            print(set(new_ids))
            content_diff_qids.add(qid)
        
    print(len(sequence_diff_qids), list(sequence_diff_qids))
    print(len(content_diff_qids), list(content_diff_qids))

def compare_merged_el_result_including_label(split):
    prev_result = load_json(f'/home3/xwu/workspace/QDT2SExpr/CWQ/all_data_bk/CWQ/entity_linking_0414/CWQ_{split}_merged_elq_FACC1.json')
    new_result = load_json(f'data/CWQ/entity_retrieval/candidate_entities/CWQ_{split}_merged_cand_entities_elq_facc1.json')
    print(len(new_result))
    assert len(prev_result) == len(new_result), print(len(prev_result), len(new_result))
    sequence_diff_qids = set()
    content_diff_qids = set()
    for qid in prev_result:
        assert qid in new_result, print(qid)
        prev_ids = [(item['id'], item['label']) for item in prev_result[qid] if 'mention' in item] # without 'mention' --> randomly sampled
        new_ids = [(item['id'], item['label']) for item in new_result[qid] if 'mention' in item]
        if prev_ids != new_ids:
            sequence_diff_qids.add(qid)
        if set(prev_ids) != set(new_ids):
            content_diff_qids.add(qid)
        
    print(len(sequence_diff_qids), list(sequence_diff_qids))
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
    prev_logits = load_json(f'data/CWQ/entity_retrieval/candidate_entities/disamb_results/CWQ_{split}/predict_logits.json')
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

def compared_generated_sexpr(split):
    prev_data = load_json('data/CWQ/sexpr/CWQ.{}.sexpr.json'.format(split))
    generated_data = load_json('data/CWQ/sexpr/xwu_test/CWQ.{}.sexpr.json'.format(split))
    prev_data = {item["ID"]: item for item in prev_data}
    generated_data = {item["ID"]: item for item in generated_data}
    diff = set()
    for idx in tqdm(generated_data, total=len(generated_data)):
        if idx not in prev_data:
            if generated_data[idx]["sexpr"] != "null":
                diff.add(idx)
        elif prev_data[idx]['sexpr'] != generated_data[idx]["sexpr"]:
            diff.add(idx)

        # if prev_data[idx]["sexpr"] != generated_data[idx]["sexpr"]:
        #     diff.add(idx)
    print('diff: {} {}'.format(len(diff), list(diff)))

"""
Bi-encoder 相关的
"""
def compare_training_data(split):
    new_training_data_path = 'data/WebQSP/relation_retrieval/bi-encoder/xwu_test_0713/WebQSP.{}.sampled.tsv'.format(split)
    prev_training_data_path = 'data/WebQSP/relation_retrieval/bi-encoder/xwu_test_0713/prev/{}.sampled.richRelation.1parse.tsv'.format(split)
    new_df = pd.read_csv(new_training_data_path, sep='\t', error_bad_lines=False).dropna()
    prev_df = pd.read_csv(prev_training_data_path, sep='\t', error_bad_lines=False).dropna()
    new_questions = new_df["question"].unique().tolist()
    prev_questions = prev_df["question"].unique().tolist()
    print(len(new_questions), len(prev_questions))
    print(new_questions == prev_questions)
    print(set(prev_questions) - set(new_questions))
    print(set(new_questions) - set(prev_questions))
    # for i in range(100):
    #     print(new_questions[i], prev_questions[i])


def calculate_topk_relation_recall(
    sorted_file, 
    dataset_file,
    topk=10
):
    r_list = []
    p_list = []
    dataset = load_json(dataset_file)
    # dataset = {example["ID"]: example for example in dataset}
    dataset = {example["QuestionId"]: example for example in dataset}
    sorted_relations = load_json(sorted_file)
    sorted_relations = {item["ID"]: item for item in sorted_relations}
    # assert len(sorted_relations) == len(dataset), print(len(dataset), len(sorted_relations))
    for qid in dataset:
        if qid not in sorted_relations:
            # print(qid)
            continue
        # pred_rels = sorted_relations[qid][:topk]
        # pred_rels = sorted_relations[qid]['relations'][:topk]
        # pred_rels = list(sorted_relations[qid]['cand_relation_list'].keys())
        pred_rels = [item[0] for item in sorted_relations[qid]['cand_relation_list']][:topk]
        golden_rels = dataset[qid]['gold_relation_map'].keys()
        if len(pred_rels)== 0:
            if len(golden_rels)==0:
                r=1
                p=1
            else:
                r=0
                p=0
        elif len(golden_rels)==0:
            r=0
            p=0
        else:
            r = len(pred_rels & golden_rels)/ len(golden_rels)
            p = len(pred_rels & golden_rels)/ len(pred_rels)
        r_list.append(r)
        p_list.append(p)

    print('topk: {}'.format(topk))
    print('Recall: {}'.format(sum(r_list)/len(r_list)))
    print('Precision: {}'.format(sum(p_list)/len(p_list)))

def substitude_relations_in_merged_file_with_addition(
    prev_merged_path, 
    output_path, 
    sorted_relations_path,
    addition_relations_path,
    topk=10
):
    """
    逻辑: 如果 sorted_relations_path 里头这个问题的候选关系为空，就选择addition_relations_path中这个问题候选关系的 topk
    如果是在二跳关系上进行预测，很可能候选关系为空
    """
    prev_merged = load_json(prev_merged_path)
    sorted_relations = load_json(sorted_relations_path)
    additional_relation = load_json(addition_relations_path)
    new_merged = []
    for example in tqdm(prev_merged, total=len(prev_merged)):
        qid = example["ID"]
        if qid not in sorted_relations or len(sorted_relations[qid]) < topk: # 我们需要恰好 10 个关系
            print(qid)
            cand_relations = additional_relation[qid][:topk]
        else:
            # cand_relations = [
            #     [item, 1.0, None]
            #     for item in sorted_relations[qid]["relations"][:topk]
            # ]
            cand_relations = sorted_relations[qid][:topk]
        example["cand_relation_list"] = cand_relations
        new_merged.append(example)
    assert len(prev_merged) == len(new_merged)
    dump_json(new_merged, output_path)


def substitude_relations_in_merged_file(
    prev_merged_path, 
    output_path, 
    sorted_logits_path,
    topk=10,
    two_hop_relation_path=None
):
    prev_merged = load_json(prev_merged_path)
    sorted_logits = load_json(sorted_logits_path)
    new_merged = []
    if two_hop_relation_path is not None:
        all_two_hop_relation = load_json(two_hop_relation_path)
        all_two_hop_relation = {item["id"]: item for item in all_two_hop_relation}
    else:
        all_two_hop_relation = None
    for example in tqdm(prev_merged, total=len(prev_merged)):
        qid = example["ID"]
        if qid not in sorted_logits:
            print(qid)
        if all_two_hop_relation is not None:
            two_hop_rels = all_two_hop_relation[qid]["two_hop_relations"]
            new_sorted_logits = [item for item in sorted_logits[qid] if item[0] in two_hop_rels][:topk]
            if len(new_sorted_logits) < topk:
                cur_len = len(new_sorted_logits)
                current_rels = [item[0] for item in new_sorted_logits]
                sorted_rels = [item[0] for item in sorted_logits[qid]]
                diff_rels = list(set(sorted_rels) - set(current_rels))
                for idx in range(topk-cur_len):
                    new_sorted_logits.append([
                        diff_rels[idx],
                        1.0,
                        None
                    ])
            
            if len(new_sorted_logits) != topk:
                print(qid)
            
            example["cand_relation_list"] = new_sorted_logits
        else:
            example["cand_relation_list"] = sorted_logits[qid][:topk]
        new_merged.append(example)
    dump_json(new_merged, output_path)


def validation_merged_file(prev_file, new_file):
    prev_data = load_json(prev_file)
    new_data = load_json(new_file)
    assert len(prev_data) == len(new_data), print(len(prev_data), len(new_data))
    for (prev, new) in tqdm(zip(prev_data, new_data), total=len(prev_data)):
        for key in prev.keys():
            if key != 'cand_relation_list':
                assert prev[key] == new[key]
            else:
                assert len(prev[key]) == 10
                assert len(new[key]) == 10, print(len(new[key]))


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
    
def calculate_disambiguated_el_res(src_path, golden_path):
    src_data = load_json(src_path)
    golden_data = load_json(golden_path)
    predictions = []
    golden = []
    for example in golden_data:
        qid = example["ID"]
        golden_entities = example["gold_entity_map"].keys()
        golden.append(golden_entities)
        if qid not in src_data or len(src_data[qid]) == 0:
            predictions.append([])
        else:
            predictions.append(src_data[qid].keys())
    
    print(general_PRF1(predictions, golden))

def calculate_disambiguated_el_res_rng(src_path, golden_path):
    src_data = load_json(src_path)
    src_data = {item["id"]: item for item in src_data}
    golden_data = load_json(golden_path)
    predictions = []
    golden = []
    for example in golden_data:
        qid = example["ID"]
        golden_entities = example["gold_entity_map"].keys()
        golden.append(golden_entities)
        if qid not in src_data:
            predictions.append([])
        else:
            predictions.append(src_data[qid]["freebase_ids"])
    
    print(general_PRF1(predictions, golden))

def validate_answer(prev_path, new_path):
    prev_data = load_json(prev_path)
    new_data = load_json(new_path)
    diff = set()
    assert len(prev_data) == len(new_data)
    for (prev, new) in zip(prev_data, new_data):
        assert prev["ID"] == new["ID"]
        if set(prev["answer"]) != set(new["answer"]):
            diff.add(prev["ID"])
    print(len(diff), list(diff))

def error_analysis_new(
    prev_success_path,
    prev_failed_path,
    new_success_path,
    new_failed_path
):
    prev_success = load_json(prev_success_path)
    prev_failed = load_json(prev_failed_path)
    new_success = load_json(new_success_path)
    new_failed = load_json(new_failed_path)

    prev_success = {item["qid"]: item for item in prev_success}
    prev_failed = {item["qid"]: item for item in prev_failed}
    new_success = {item["qid"]: item for item in new_success}
    new_failed = {item["qid"]: item for item in new_failed}

    diff = set()

    for qid in new_success:
        if qid in prev_success:
            if new_success[qid]["f1"] < prev_success[qid]["f1"]:
                diff.add(qid)
    
    for qid in new_failed:
        if qid not in prev_failed:
            diff.add(qid)
    print(len(diff), list(diff))


def main(split):
    compared_generated_sexpr(split)

def get_cross_encoder_tsv_max_len(tsv_file):
    tsv_df = pd.read_csv(tsv_file, sep='\t', error_bad_lines=False).dropna()
    tokenizer = AutoTokenizer.from_pretrained('hfcache/bert-base-uncased')
    length_dict = defaultdict(int)
    for idx in tqdm(range(len(tsv_df)), total=len(tsv_df)):
        question = tsv_df.loc[idx, 'question']
        relation = tsv_df.loc[idx, 'relation']
        tokenized = tokenizer.tokenize(question, relation)
        length_dict[len(tokenized)] += 1
    print(collections.OrderedDict(sorted(length_dict.items())))

def get_bi_encoder_tsv_max_len(tsv_file):
    tsv_df = pd.read_csv(tsv_file, sep='\t', error_bad_lines=False).dropna()
    tokenizer = AutoTokenizer.from_pretrained('hfcache/bert-base-uncased')
    length_dict = defaultdict(int)
    for idx in tqdm(range(len(tsv_df)), total=len(tsv_df)):
        question = tsv_df.loc[idx, 'question']
        relation = tsv_df.loc[idx, 'relation']
        tokenized_question = tokenizer.tokenize(question)
        length_dict[len(tokenized_question)] += 1
        tokenized_relation = tokenizer.tokenize(relation)
        length_dict[len(tokenized_relation)] += 1
    print(collections.OrderedDict(sorted(length_dict.items())))

def calculate_rng_2hop_recall(
    rng_data_path,
    golden_data_path,
):
    rng_data = load_json(rng_data_path)
    golden_data = load_json(golden_data_path)
    rng_data = {item["id"]: item for item in rng_data}
    golden_data = {item["QuestionId"]: item for item in golden_data}
    assert len(rng_data) == len(golden_data)
    r_list = []
    candidate_numbers = []
    for qid in rng_data:
        assert qid in golden_data
        pred_rels = rng_data[qid]["two_hop_relations"]
        golden_rels = golden_data[qid]['gold_relation_map'].keys()
        if len(pred_rels)== 0:
            if len(golden_rels)==0:
                r=1
            else:
                r=0
        elif len(golden_rels)==0:
            r=0
        else:
            r = len(pred_rels & golden_rels)/ len(golden_rels)
        r_list.append(r)
        candidate_numbers.append(len(pred_rels))

    print('Recall: {}'.format(sum(r_list)/len(r_list)))
    print('Average candidate number: {}'.format(sum(candidate_numbers)/len(candidate_numbers)))

def calculate_bi_encoder_recall(
    all_relations_path, 
    golden_data_path, 
    question_vectors_path, 
    index_file, 
    vector_size=768, 
    index_buffer=50000, 
    top_k=150, 
):
    all_relations = load_json(all_relations_path)
    golden_data = load_json(golden_data_path)
    
    index = DenseFlatIndexer(vector_size, index_buffer)
    index.deserialize_from(index_file) 
    question_vectors = torch.load(question_vectors_path).cpu().detach().numpy()
    _, pred_relation_indexes = index.search_knn(question_vectors, top_k=top_k)
    pred_relations = pred_relation_indexes.tolist()
    pred_relations = [[all_relations[index] for index in indexes ] for indexes in pred_relation_indexes ]
    
    assert len(golden_data) == len(pred_relations)
    
    r_list = []
    for i in range(len(golden_data)):
        pred_rels = pred_relations[i]
        # pred_rels = [rel.split('|')[0] for rel in pred_rels]
        golden_rels = golden_data[i]["gold_relation_map"].keys()
        if len(pred_rels)== 0:
            if len(golden_rels)==0:
                r=1
            else:
                r=0
        elif len(golden_rels)==0:
            r=0
        else:
            r = len(pred_rels & golden_rels)/ len(golden_rels)
        r_list.append(r)

    print('Recall: {}'.format(sum(r_list)/len(r_list)))
    return sum(r_list)/len(r_list)

def validate_tsv_positive_samples(
    tsv_file_1,
    tsv_file_2
):
    tsv1_df = pd.read_csv(tsv_file_1, sep='\t', error_bad_lines=False).dropna()
    tsv2_df = pd.read_csv(tsv_file_2, sep='\t', error_bad_lines=False).dropna()
    print(len(tsv1_df), len(tsv2_df), len(tsv1_df) == len(tsv2_df))
    tsv1_positive = tsv1_df.loc[tsv1_df['label'] == 1] 
    tsv2_positive = tsv2_df.loc[tsv2_df['label'] == 1]
    print(len(tsv1_positive), len(tsv2_positive), len(tsv1_positive) == len(tsv2_positive))


def validate_ptrain_pdev_vectors(
    ptrain_question_vector_path,
    pdev_question_vector_path,
    train_question_vector_path,
    index_file,
    ptrain_data_path,
    pdev_data_path,
    train_data_path,
    vector_size=768, 
    index_buffer=50000, 
    top_k=100
):
    ptrain_data = load_json(ptrain_data_path)
    pdev_data = load_json(pdev_data_path)
    train_data = load_json(train_data_path)
    index = DenseFlatIndexer(vector_size, index_buffer)
    index.deserialize_from(index_file) 

    ptrain_question_vectors = torch.load(ptrain_question_vector_path).cpu().detach().numpy()
    pdev_question_vectors = torch.load(pdev_question_vector_path).cpu().detach().numpy()
    train_question_vectors = torch.load(train_question_vector_path).cpu().detach().numpy()

    _, ptrain_pred_relation_indexes = index.search_knn(ptrain_question_vectors, top_k=top_k)
    _, pdev_pred_relation_indexes = index.search_knn(pdev_question_vectors, top_k=top_k)
    _, train_pred_relation_indexes = index.search_knn(train_question_vectors, top_k=top_k)
    ptrain_id_relation_map = {}
    pdev_id_relation_map = {}
    train_id_relation_map = {}

    for idx in tqdm(range(len(ptrain_data)), total = len(ptrain_data)):
        example = ptrain_data[idx]
        qid = example["QuestionId"]
        pred_relations = ptrain_pred_relation_indexes[idx]
        ptrain_id_relation_map[qid] = pred_relations
    
    for idx in tqdm(range(len(pdev_data)), total = len(pdev_data)):
        example = pdev_data[idx]
        qid = example["QuestionId"]
        pred_relations = pdev_pred_relation_indexes[idx]
        pdev_id_relation_map[qid] = pred_relations
    
    for idx in tqdm(range(len(train_data)), total = len(train_data)):
        example = train_data[idx]
        qid = example["QuestionId"]
        pred_relations = train_pred_relation_indexes[idx]
        train_id_relation_map[qid] = pred_relations
    
    for qid in tqdm(train_id_relation_map, total=len(train_id_relation_map)):
        if qid in pdev_id_relation_map:
            # print(train_id_relation_map[qid])
            # print(pdev_id_relation_map[qid])
            if list(train_id_relation_map[qid]) != list(pdev_id_relation_map[qid]):
                print(qid)
        elif qid in ptrain_id_relation_map:
            # print(set(train_id_relation_map[qid]) - set(ptrain_id_relation_map[qid]))
            # print(ptrain_id_relation_map[qid])
            if list(train_id_relation_map[qid]) != list(ptrain_id_relation_map[qid]):
                print(qid)


def compare_generation_results(
    generation_failed_a_path,
    generation_succeed_a_path,
    generation_failed_b_path,
    generation_succeed_b_path
):
    generation_failed_a = load_json(generation_failed_a_path)
    generation_failed_b = load_json(generation_failed_b_path)
    generation_failed_a_qids = [item["qid"] for item in generation_failed_a]
    generation_failed_b_qids = [item["qid"] for item in generation_failed_b]
    diff = set(generation_failed_a_qids) - set(generation_failed_b_qids)
    print('A failed but b succeed: {}'.format(list(diff)))

    generation_succeed_a = load_json(generation_succeed_a_path)
    generation_succeed_b = load_json(generation_succeed_b_path)
    generation_succeed_a = {item["qid"]: item for item in generation_succeed_a}
    generation_succeed_b = {item["qid"]: item for item in generation_succeed_b}
    diff2 = set()
    for qid in generation_succeed_b:
        if qid in generation_succeed_a:
            if generation_succeed_a[qid]["f1"] < generation_succeed_b[qid]["f1"]:
                diff2.add(qid)
    print('A has less f1: {}'.format(list(diff2)))


def calc_relation_num_influence(
    dataset_file_path,
    gen_failed_results_path,
    gen_succeed_results_path,
    topk=5,
):
    """
    如果考虑 top5 之外的候选关系
    1. 是 golden 关系: 如果出现在 f1 > 0.1 的问题中，并且最终执行的 S-Expr 包含该关系，正例 + 1
    2. 不是 golden 关系: 可执行，f1<0.1, 且执行的那一条出现该关系
        不可执行: 50条中有出现这个关系的， 负例加1
    仍然要记录 id, 人工再看看
    """
    dataset = load_json(dataset_file_path)
    gen_failed = load_json(gen_failed_results_path)
    gen_failed = {item["qid"]: item for item in gen_failed}
    gen_succeed = load_json(gen_succeed_results_path)
    gen_succeed =  {item["qid"]: item for item in gen_succeed}
    positive_samples = set()
    negative_samples = set()
    for example in dataset:
        golden_relations = example["gold_relation_map"].keys()
        cand_relations = example["cand_relation_list"][topk:]
        qid = example["ID"]
        for cand_rel in cand_relations:
            if cand_rel[0] in golden_relations:
                if qid not in gen_succeed:
                    continue
                else:
                    executed_sexpr = gen_succeed[qid]["logical_form"]
                    if cand_rel[0] in executed_sexpr:
                        positive_samples.add(qid)
            else:
                if qid in gen_failed:
                    predictions = gen_failed[qid]["pred"]["predictions"]
                    normalized_rel = cand_rel[2].split('|')[0].strip()
                    if any([normalized_rel in sexpr for sexpr in predictions]):
                        negative_samples.add(qid)
                elif qid in gen_succeed:
                    executed_sexpr = gen_succeed[qid]["logical_form"]
                    if cand_rel[0] in executed_sexpr:
                        negative_samples.add(qid)
    print('positive: {} {}'.format(len(positive_samples), list(positive_samples)))
    print('negative: {} {}'.format(len(negative_samples), list(negative_samples)))

def compare_answers(json_path_1, json_path_2):
    json_1 = load_json(json_path_1)
    json_1 = {example["ID"]: example["answer"] for example in json_1}
    json_2 = load_json(json_path_2)
    json_2 = {example["ID"]: example["answer"] for example in json_2}
    for qid in json_1:
        assert qid in json_2
        assert set(json_1[qid]) == set(json_2[qid]), print(qid)


if __name__=='__main__':
    # compare_answers(
    #     'data/CWQ/origin/ComplexWebQuestions_test.json',
    #     'data/CWQ/generation/merged/CWQ_test.json'
    # )
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
    # compare_disamb_logits('dev')
    # compare_merged_el_result('train')
    # compare_merged_el_result_including_label('train')
    # disamb_logits_generation('dev')
    # test_merged_length()
    # check_disambed_entities('test')
    # compare_facc1_ignore_sequence('train')
    # compare_facc1('test')

    # for split in ['test', 'train', 'dev']:
    #     compared_generated_sexpr(split)
    # compare_training_data('train')

    # calculate_top10_relation_recall(
    #     # 'data/WebQSP/relation_retrieval/candidate_relations_0714_xwu/WebQSP_test_cand_rels_sorted.json',
    #     # '/home3/xwu/new_workspace/GMT-KBQA/data/WebQSP/generation/merged/WebQSP_test.json'
    #     'data/CWQ/relation_retrieval/candidate_relations/CWQ_test_cand_rels_sorted.json',
    #     '/home3/xwu/new_workspace/GMT-KBQA/data/CWQ/generation/merged_old/CWQ_test.json'
    # )
    # CWQ 
    # for split in ['test', 'dev', 'train']:
    #     # substitude_relations_in_merged_file(
    #     #     f'data/CWQ/generation/merged/CWQ_{split}.json',
    #     #     f'data/CWQ/generation/merged_0715_retrain_new_data/CWQ_{split}.json',
    #     #     f'data/CWQ/relation_retrieval/0715_retrain/CWQ_{split}_cand_rel_logits.json',
    #     # )
        # calculate_topk_relation_recall(
        #     f'data/CWQ/generation/merged_0724_ep1/CWQ_{split}.json',
        #     f'data/CWQ/generation/merged_old/CWQ_{split}.json',
        # )
        # validation_merged_file(
        #     f'data/CWQ/generation/merged/CWQ_{split}.json',
        #     f'data/CWQ/generation/merged_0715_retrain_new_data/CWQ_{split}.json',
        # )
        # validate_answer(
        #     f'data/CWQ/origin/ComplexWebQuestions_{split}.json',
        #     f'data/CWQ/generation/merged_old/CWQ_{split}.json',
        # )
    
    # WebQSP
    # for split in ['train', 'ptrain', 'pdev', 'test']:
    for split in ['train', 'test']:
        # substitude_relations_in_merged_file_with_addition(
        #     f'data/WebQSP/generation/merged_old/WebQSP_{split}.json',
        #     f'data/WebQSP/generation/merged_question_relation_ep3_2hop/WebQSP_{split}.json',
        #     f'data/WebQSP/relation_retrieval_0717/candidate_relations/rich_relation_3epochs_question_relation_maxlen_34_ep3_2hop/WebQSP_{split}_cand_rel_logits.json',
        #     f'data/WebQSP/relation_retrieval_0717/candidate_relations/rich_relation_3epochs_question_relation_maxlen_34_ep3/WebQSP_{split}_cand_rel_logits.json',
        #     topk=10,
        # )
        # substitude_relations_in_merged_file(
        #     f'data/WebQSP/generation/merged_old/WebQSP_{split}.json',
        #     f'data/WebQSP/generation/merged_question_relation_ep3_2hop/WebQSP_{split}.json',
        #     f'data/WebQSP/relation_retrieval_0717/candidate_relations/rich_relation_3epochs_question_relation_maxlen_34_ep3_2hop/WebQSP_{split}_cand_rel_logits.json',
        #     topk=10,
        # )
        calculate_topk_relation_recall(
            # f'data/WebQSP/relation_retrieval/candidate_relations_yhshu/WebQSP_{split}_cand_rels_sorted.json',
            # f'data/WebQSP/relation_retrieval_0717/candidate_relations/rich_relation_3epochs_question_relation_maxlen_34_ep3_2hop/WebQSP_{split}_cand_rels_sorted.json',
            f'data/WebQSP/generation/merged_corrected_relation_final/WebQSP_{split}.json',
            # f'data/WebQSP/generation/merged_old/WebQSP_{split}.json',
            f'data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.{split}.goldenRelation.json',
            topk=10
        )
        # validation_merged_file(
        #     f'data/WebQSP/generation/merged_old/WebQSP_{split}.json',
        #     f'data/WebQSP/generation/merged_question_relation_ep3_2hop/WebQSP_{split}.json',
        # )
        # calculate_disambiguated_el_res(
        #     f'data/WebQSP/entity_retrieval/linking_results/merged_WebQSP_{split}_linking_results.json',
        #     f'data/WebQSP/generation/merged_old/WebQSP_{split}.json'
        # )
        # calculate_disambiguated_el_res_rng(
        #     f'data/WebQSP/relation_retrieval/cross-encoder/rng_linking_results/{split}_rng_elq.json',
        #     f'data/WebQSP/generation/merged_old/WebQSP_{split}.json'
        # )
        # get_cross_encoder_tsv_max_len(f'data/WebQSP/relation_retrieval_0717/cross-encoder/rich_relation_3epochs_rich_entity_rich_relation_1parse/WebQSP.{split}.tsv')

        # calculate_rng_2hop_recall(
        #     f'data/WebQSP/relation_retrieval_0722/cross-encoder/rng_kbqa_linking_results/webqsp_{split}_rng_el_two_hop_relations.json',
        #     f'data/WebQSP/relation_retrieval_0722/bi-encoder/WebQSP.{split}.goldenRelation.json'
        # )

        # calculate_bi_encoder_recall(
        #     'data/common_data/freebase_relations_filtered.json',
        #     f'data/WebQSP/relation_retrieval_0722/bi-encoder/WebQSP.{split}.goldenRelation.json',
        #     'data/WebQSP/relation_retrieval_0722/bi-encoder/vectors/relation_3epochs/WebQSP_{}_ep3_questions.pt'.format(split),
        #     'data/WebQSP/relation_retrieval_0722/bi-encoder/index/relation_3epochs/ep_3_flat.index',
        #     top_k=200
        # )
    # error_analysis_new(
    #     'exps/WebQSP_relation_entity_concat_add_prefix_warmup_epochs_5_20epochs_bs2/beam_50_top_k_predictions.json_gen_sexpr_results.json_new.json',
    #     'exps/WebQSP_relation_entity_concat_add_prefix_warmup_epochs_5_20epochs_bs2/beam_50_top_k_predictions.json_gen_failed_results.json',
    #     'exps/WebQSP_0715_retrain/beam_50_top_k_predictions.json_gen_sexpr_results.json_new.json',
    #     'exps/WebQSP_0715_retrain/beam_50_top_k_predictions.json_gen_failed_results.json',
    # )

    # validate_tsv_positive_samples(
    #     'data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.train.sampled.tsv',
    #     'data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.train.sampled.normal.tsv',
    # )

    # get_bi_encoder_tsv_max_len(
    #     'data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.train.sampled.normal.tsv'
    # )

    # validate_ptrain_pdev_vectors(
    #     'data/WebQSP/relation_retrieval_0717/bi-encoder/vectors/rich_relation_3epochs/WebQSP_ptrain_ep3_questions.pt',
    #     'data/WebQSP/relation_retrieval_0717/bi-encoder/vectors/rich_relation_3epochs/WebQSP_pdev_ep3_questions.pt',
    #     'data/WebQSP/relation_retrieval_0717/bi-encoder/vectors/rich_relation_3epochs/WebQSP_train_ep3_questions.pt',
    #     'data/WebQSP/relation_retrieval_0717/bi-encoder/index/rich_relation_3epochs/ep_3_flat.index',
    #     'data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.ptrain.goldenRelation.json',
    #     'data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.pdev.goldenRelation.json',
    #     'data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.train.goldenRelation.json',
    # )

    # compare_generation_results(
    #     'exps/WebQSP_0715_retrain/beam_50_top_k_predictions.json_gen_failed_results.json',
    #     'exps/WebQSP_0715_retrain/beam_50_top_k_predictions.json_gen_sexpr_results_official_format.json_new.json',
    #     'exps/WebQSP_relation_entity_concat_add_prefix_warmup_epochs_5_20epochs_bs2/beam_50_top_k_predictions.json_gen_failed_results.json',
    #     'exps/WebQSP_relation_entity_concat_add_prefix_warmup_epochs_5_20epochs_bs2/beam_50_top_k_predictions.json_gen_sexpr_results.json_new.json',
    # )

    # calc_relation_num_influence(
    #     'data/WebQSP/generation/merged_0719_9291/WebQSP_test.json',
    #     'exps/WebQSP_question_relation_maxlen34/beam_50_top_k_predictions.json_gen_failed_results.json',
    #     'exps/WebQSP_question_relation_maxlen34/beam_50_top_k_predictions.json_gen_sexpr_results.json'
    # )

