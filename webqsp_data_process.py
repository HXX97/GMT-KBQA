import os
from components.utils import dump_json, load_json
from tqdm import tqdm


def substitude_relations_in_merged_file(
    prev_merged_path, 
    output_path, 
    sorted_relations_path,
    addition_relations_path,
    topk=10
):
    """
    逻辑: 如果 sorted_relations_path 里头这个问题的候选关系为空，就选择addition_relations_path中这个问题候选关系的 topk
    """
    prev_merged = load_json(prev_merged_path)
    sorted_relations = load_json(sorted_relations_path)
    additional_relation = load_json(addition_relations_path)
    new_merged = []
    for example in tqdm(prev_merged, total=len(prev_merged)):
        qid = example["ID"]
        if qid not in sorted_relations or len(sorted_relations[qid]["relations"]) < topk: # 我们需要恰好 10 个关系
            print(qid)
            cand_relations = additional_relation[qid][:topk]
        else:
            cand_relations = [
                [item, 1.0, None]
                for item in sorted_relations[qid]["relations"][:topk]
            ]
        example["cand_relation_list"] = cand_relations
        new_merged.append(example)
    assert len(prev_merged) == len(new_merged)
    dump_json(new_merged, output_path)

def merge_splits(
    train_split_path,
    dev_split_path,
    merged_path
):
    """
    把 train 和 dev 的结果合并起来
    """
    train_data = load_json(train_split_path)
    dev_data = load_json(dev_split_path)
    merged_data = {**train_data, **dev_data}
    print('train: {}'.format(len(train_data)))
    print('dev: {}'.format(len(dev_data)))
    print('merged: {}'.format(len(merged_data)))
    assert len(merged_data) == len(train_data) + len(dev_data)
    dump_json(merged_data, merged_path)

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
                assert len(new[key]) == 10, print(new["ID"], len(new[key]))

def train_dev_split_as_per_rng(
    prev_train_path,
    rng_train_path,
    rng_dev_path,
    new_folder
):
    prev_train = load_json(prev_train_path)
    rng_train_keys = load_json(rng_train_path).keys()
    rng_dev_keys = load_json(rng_dev_path).keys()

    new_train_data = []
    new_dev_data = []
    for example in prev_train:
        qid = example["ID"]
        if qid not in rng_train_keys and qid not in rng_dev_keys:
            print(qid)
        if qid in rng_train_keys:
            new_train_data.append(example)
        elif qid in rng_dev_keys:
            new_dev_data.append(example)
    print('prev_train: {}'.format(len(prev_train)))
    print('new_train: {}'.format(len(new_train_data)))
    print('new_dev: {}'.format(len(new_dev_data)))
    dump_json(new_train_data, os.path.join(new_folder, 'WebQSP_train.json'))
    dump_json(new_dev_data, os.path.join(new_folder, 'WebQSP_dev.json'))

if __name__=='__main__':
    # merge_splits(
    #     'data/WebQSP/relation_retrieval/candidate_relations_yhshu/webqsp_train_relations.json',
    #     'data/WebQSP/relation_retrieval/candidate_relations_yhshu/webqsp_dev_relations.json',
    #     'data/WebQSP/relation_retrieval/candidate_relations_yhshu/webqsp_train_dev_merged_relations.json'
    # )
    # substitude_relations_in_merged_file(
    #     'data/WebQSP/generation/merged_old/WebQSP_train.json',
    #     'data/WebQSP/generation/merged_yhshu/WebQSP_train.json',
    #     'data/WebQSP/relation_retrieval/candidate_relations_yhshu/webqsp_train_dev_merged_relations.json',
    #     'data/WebQSP/relation_retrieval/0715_retrain/WebQSP_train_cand_rel_logits.json',
    #     topk=10
    # )
    # substitude_relations_in_merged_file(
    #     'data/WebQSP/generation/merged_old/WebQSP_test.json',
    #     'data/WebQSP/generation/merged_yhshu/WebQSP_test.json',
    #     'data/WebQSP/relation_retrieval/candidate_relations_yhshu/webqsp_test_relations.json',
    #     'data/WebQSP/relation_retrieval/0715_retrain/WebQSP_test_cand_rel_logits.json',
    #     topk=10
    # )
    # for split in ['train', 'test']:
    #     validation_merged_file(
    #         f'data/WebQSP/generation/merged_old/WebQSP_{split}.json',
    #         f'data/WebQSP/generation/merged_yhshu/WebQSP_{split}.json',
    #     )
    train_dev_split_as_per_rng(
        'data/WebQSP/generation/merged_yhshu/WebQSP_train.json',
        'data/WebQSP/relation_retrieval/candidate_relations_yhshu/webqsp_train_relations.json',
        'data/WebQSP/relation_retrieval/candidate_relations_yhshu/webqsp_dev_relations.json',
        'data/WebQSP/generation/merged_yhshu_with_dev_split'
    )