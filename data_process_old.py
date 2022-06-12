from functools import reduce
import json
import random

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

def combine_FACC1_and_disamb(split):
    """结合 FACC1 的链接结果，和消岐后的结果"""
    facc1_res = load_json('CWQ_{}_entities_FACC1.json'.format(split))
    disamb_res = load_json('/home3/xwu/workspace/QDT2SExpr/CWQ/results/disamb/CWQ_{}/predictions.json'.format(split))
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
    
    dump_json(new_res, 'CWQ_{}_entities_FACC1_disamb.json'.format(split))
    print('missed: {}'.format(missed_count))


def get_all_train_entities_FACC1():
    """得到 FACC1 在训练集上的链接结果中的所有链接到的实体（去重）"""
    train_facc1_res = load_json('CWQ_train_entities_FACC1.json')
    res = []
    for id in train_facc1_res:
        for mention_res in train_facc1_res[id]:
            for linked_res in mention_res:
                res.append({
                    "id": linked_res["id"],
                    "label": linked_res["label"],
                    "pop_score": linked_res["pop_score"]
                })
    
    # 按照 id 去重
    res = list(dict((v['id'],v) for v in res).values())
    dump_json(res, 'CWQ_unique_train_entities_FACC1.json')


def get_all_train_entities_ELQ():
    """得到 ELQ 在训练集上的链接结果中的所有链接到的实体（根据 id 去重）"""
    train_elq_res = load_json('CWQ_train_cand_entities_elq.json')
    res = []
    for id in train_elq_res:
        for linked_res in train_elq_res[id]:
            res.append({
                "id": linked_res["id"],
                "label": linked_res["label"],
            })       
    
    # 按照 id 去重
    res = list(dict((v['id'],v) for v in res).values())
    dump_json(res, 'CWQ_unique_train_entities_elq.json')

def add_candidate_entities_list_to_merged_data(split):
    prev_merged_data = load_json('../merged_all_data/2hopValidation_maskMention_richRelation_CrossEntropyLoss_top100/CWQ_{}_all_data.json'.format(split))
    facc1_disamb_res = load_json('CWQ_{}_entities_FACC1_disamb.json'.format(split))
    train_entities_facc1 = load_json('CWQ_unique_train_entities_FACC1.json')
    new_merged_data = []


    for item in prev_merged_data:
        qid = item["ID"]
        print(qid)
        cand_entity_list = []
        if qid not in facc1_disamb_res or len(facc1_disamb_res[qid]) == 0:
            # 没有任何链接结果：暂时 golden + 任意选
            for mid in item["gold_entity_map"]:
                cand_entity_list.append([
                    item["gold_entity_map"][mid], # label
                    str(0.0) # 这类伪数据， logits 赋0
                ])
            diff_entities = list(filter(lambda v: v["id"] not in item["gold_entity_map"], train_entities_facc1))
            random_entities = random.sample(diff_entities, 10 - len(cand_entity_list))

            for ent in random_entities:
                cand_entity_list.append([
                    ent["label"],
                    str(0.0) # 这类伪数据，logits 赋0
                ])
        else: # 有链接结果，需要检查是否 > 10
            total_len = reduce(
                lambda x,y: x + len(y),
                facc1_disamb_res[qid],
                0
            )
            if total_len > 10:
                random_choice_list = []
                for mention_res in facc1_disamb_res[qid]:
                    if len(mention_res) == 0:
                        continue
                    cand_entity_list.append([
                        mention_res[0]["label"],
                        str(mention_res[0]["disamb_logits"]) if "disamb_logits" in mention_res[0] else str(0.0)# 这类伪数据， logits 赋0
                    ])
                    random_choice_list.extend(mention_res[1:])
                # 从random_choice_list中随机选择，凑满10个
                random_entities = random.sample(random_choice_list, 10 - len(cand_entity_list))

                for ent in random_entities:
                    cand_entity_list.append([
                        ent["label"],
                        str(ent["disamb_logits"]) if "disamb_logits" in ent else str(0.0) # 这类伪数据，logits 赋0
                    ])
            else: # 少于10，则全部加入candidate, 并任意选凑满10个
                for mention_res in facc1_disamb_res[qid]:
                    for v in mention_res:
                        cand_entity_list.append([
                            v["label"],
                            str(v["disamb_logits"]) if "disamb_logits" in v else str(0.0) # 这类伪数据，logits 赋0
                        ])
            
                random_entities = random.sample(train_entities_facc1, 10 - len(cand_entity_list))

                for ent in random_entities:
                    cand_entity_list.append([
                        ent["label"],
                        str(0.0) # 这类伪数据，logits 赋0
                    ])
        assert len(cand_entity_list) == 10, print(qid)
        item["cand_entity_list"] = cand_entity_list
        new_merged_data.append(item)
    
    dump_json(new_merged_data, '../merged_all_data/2hopValidation_maskMention_richRelation_CrossEntropyLoss_top100_candidate_entities_FACC1_disamb/CWQ_{}_all_data.json'.format(split))


def combine_FACC1_and_elq(split, sample_size=10):

    entity_dir = 'data/CWQ/entity_retrieval/candidate_entities'

    facc1_disamb_res = load_json(f'{entity_dir}/CWQ_{split}_cand_entities_facc1.json')
    elq_res = load_json(f'{entity_dir}/CWQ_{split}_cand_entities_elq.json')

    print('lens: {}'.format(len(elq_res.keys())))

    combined_res = dict()
    # train_entities_elq = load_json('CWQ_unique_train_entities_elq.json')

    train_entities_elq = {}
    elq_res_train = load_json(f'{entity_dir}/CWQ_train_cand_entities_elq.json')
    for qid,cand_ents in elq_res_train.items():
        for ent in cand_ents:
            train_entities_elq[ent['id']] = ent['label']
        
    
    train_entities_elq = [{"id":mid,"label":label} for mid,label in train_entities_elq.items()]


    for qid in elq_res:
        cur = dict() # unique by mid

        elq_result = elq_res[qid]
        facc1_result = facc1_disamb_res.get(qid,[])
        # facc1_result = [item for mention_res in facc1_disamb_res[qid] for item in mention_res] if qid in facc1_disamb_res else []
        
        # sort by score
        elq_result = sorted(elq_result, key=lambda d: d.get('score', -20.0), reverse=True)
        facc1_result = sorted(facc1_result, key=lambda d: d.get('logit', 0.0), reverse=True)

        # merge the linking results of ELQ and FACC1 one by one
        idx = 0
        while len(cur.keys()) < sample_size:
            if idx < len(elq_result):
                cur[elq_result[idx]["id"]] = elq_result[idx]
            if len(cur.keys()) < sample_size and idx < len(facc1_result):
                cur[facc1_result[idx]["id"]] = facc1_result[idx]
            if idx >= len(elq_result) and idx >= len(facc1_result):
                break
            idx += 1

        if len(cur.keys()) < sample_size:
            # sample some entities to reach the sample size
            diff_entities = list(filter(lambda v: v["id"] not in cur.keys(), train_entities_elq))
            random_entities = random.sample(diff_entities, 10 - len(cur.keys()))
            for ent in random_entities:
                cur[ent["id"]] = ent

        assert len(cur.keys()) == sample_size, print(qid)
        combined_res[qid] = list(cur.values())

    
    merged_file_path = f'{entity_dir}/CWQ_{split}_merged_elq_FACC1.json'
    print(f'Writing merged candidate entities to {merged_file_path}')
    dump_json(combined_res, merged_file_path, indent=4)

    

def calculate_el_recall(split):
    # merged_dataset = load_json('../merged_all_data/2hopValidation_maskMention_richRelation_CrossEntropyLoss_top100/CWQ_{}_all_data.json'.format(split))
    merged_dataset = load_json('/home3/xwu/workspace/QDT2SExpr/CWQ/data/CWQ/final/merged/CWQ_{}.json'.format(split))
    el_res = load_json('0506/CWQ_{}_FACC1_update_label.json'.format(split))

    r_list = []
    for data in merged_dataset:
        qid = data["ID"]
        if qid not in el_res:
            r = 0
        else:
            golden_entities = set(data["gold_entity_map"].keys())
            pred_entities = set(map(lambda item: item["id"], el_res[qid]))
            if len(pred_entities) == 0:
                if len(golden_entities) == 0:
                    r=1
                else:
                    r=0
            elif len(golden_entities) == 0:
                r = 0
            else:
                r = len(pred_entities & golden_entities) / len(golden_entities)
        
        r_list.append(r)
    
    r_average = sum(r_list)/len(merged_dataset)
    print('Average Recall: {}'.format(r_average))


def update_entity_label():
    """
    FACC1 里头的 label 有些问题（会出现外文字符）
    统一改用 label_maps 文件夹下的 label
    """
    missed_entity = set()
    for split in ['train', 'dev', 'test']:
        el_res = load_json('0506/CWQ_{}_entities.json'.format(split))
        label_map = load_json('../final/label_maps/CWQ_{}_entity_label_map.json'.format(split))
        extra_label_map = load_json('0506/CWQ_FACC1_extra_label_map.json')
        
        updated_res = dict()
        
        for qid in el_res:
            values = el_res[qid]
            values = [list_item for list in values for list_item in list]
            for item in values:
                if item["id"] in label_map:
                    item["label"] = label_map[item["id"]]
                elif item["id"] in extra_label_map:
                    item['label'] = extra_label_map[item['id']]
                else:
                    missed_entity.add(item['id'])
                    
            updated_res[qid] = values
        
        dump_json(updated_res, '0506/CWQ_{}_FACC1_update_label.json'.format(split))
    
    # dump_json(list(missed_entity), '0506/CWQ_FACC1_missed_entities.json')
    print(len(list(missed_entity)))


def add_candidate_entities_to_merged_data(split):
    merged_data = load_json('../merged_all_data/2hopValidation_maskMention_richRelation_CrossEntropyLoss_top100/CWQ_{}_all_data.json'.format(split))
    candidate_entities = load_json('CWQ_{}_merged_elq_FACC1.json'.format(split))

    new_data = []
    for data in merged_data:
        qid = data["ID"]
        if qid not in candidate_entities:
            print(qid)
            continue
        else:
            entities = list(map(lambda item: {"id": item["id"], "label": item["label"]}, candidate_entities[qid]))
            assert len(entities) == 10, print(qid)
            data["cand_entity_list"] = entities
            new_data.append(data)
    
    dump_json(new_data, '../merged_all_data/2hopValidation_maskMention_richRelation_CrossEntropyLoss_top100_candidate_entities_merged_FACC1_elq/CWQ_{}_all_data.json'.format(split))


# def get_entity_map_from_FACC1_result(split):
#     """
    
#     """

def test():
    a = load_json('CWQ_test_cand_entities_elq.json')
    b = load_json('CWQ_test_entities_FACC1_disamb.json')
    print(len(a))
    print(len(b))
    a_set = set()
    b_set = set()
    for qid in a:
        if len(a[qid]) == 0:
            a_set.add(qid)
    for qid in b:
        if len(b[qid]) == 0:
            b_set.add(qid)
    print(list(a_set))
    print(list(b_set))
    print(list(a_set & b_set))


if __name__ == "__main__":
    # for split in ['test', 'train', 'dev']:
    #     combine_FACC1_and_disamb(split)
    # get_all_train_entities_FACC1()

    # for split in ['train', 'dev', 'test']:
    #     print(split)
    #     add_candidate_entities_list_to_merged_data(split)

    combine_FACC1_and_elq('test')
    # for split in ['test','train', 'dev']:
    #     print(split)
    #     # combine_FACC1_and_elq(split)
    #     calculate_el_recall(split)
        # add_candidate_entities_to_merged_data(split)
    
    # update_entity_label()
    # test()