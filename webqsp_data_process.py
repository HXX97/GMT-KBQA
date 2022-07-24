import os
from components.utils import dump_json, load_json
from tqdm import tqdm


def substitude_relations_in_merged_file_with_cross_encoder_res(
    prev_merged_path, 
    output_path, 
    sorted_logits_folder,
    topk=5,
):
    prev_merged = load_json(prev_merged_path)
    if 'train' in prev_merged_path:
        sorted_logits = {
            **load_json(os.path.join(sorted_logits_folder, 'WebQSP_ptrain_cand_rel_logits.json')),
            **load_json(os.path.join(sorted_logits_folder, 'WebQSP_pdev_cand_rel_logits.json'))
        }
    elif 'test' in prev_merged_path:
        sorted_logits = load_json(os.path.join(sorted_logits_folder, 'WebQSP_test_cand_rel_logits.json'))
    new_merged = []
    debug=True
    for example in tqdm(prev_merged, total=len(prev_merged)):
        qid = example["ID"]
        if qid not in sorted_logits:
            print(qid)
        # top 5 + last 5
        example["cand_relation_list"] = []
        example["cand_relation_list"].extend(sorted_logits[qid][:topk])
        example["cand_relation_list"].extend(sorted_logits[qid][-topk:])
        # if debug:
        #     print(sorted_logits[qid])
        #     print(example["cand_relation_list"], len(example["cand_relation_list"]))
        new_merged.append(example)
        debug=False
    dump_json(new_merged, output_path)

def substitude_relations_in_merged_file_2hop(
    prev_merged_path, 
    output_path, 
    sorted_relations_path,
    addition_relations_folder,
    topk=10
):
    """
    逻辑: 如果 sorted_relations_path 里头这个问题的候选关系为空，就选择addition_relations_path中这个问题候选关系的 topk
    如果是在二跳关系上进行预测，很可能候选关系为空
    """
    prev_merged = load_json(prev_merged_path)
    sorted_relations = load_json(sorted_relations_path)
    if 'train' in prev_merged_path:
        additional_relation = {
            **load_json(os.path.join(addition_relations_folder, 'WebQSP_ptrain_cand_rel_logits.json')),
            **load_json(os.path.join(addition_relations_folder, 'WebQSP_pdev_cand_rel_logits.json'))
        }
    elif 'test' in prev_merged_path:
        additional_relation = load_json(os.path.join(addition_relations_folder, 'WebQSP_test_cand_rel_logits.json'))
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

def train_dev_split_as_per_files(
    prev_train_path,
    split_train_path,
    split_dev_path,
    new_folder
):
    prev_train = load_json(prev_train_path)
    split_train = load_json(split_train_path)
    split_train = {item["QuestionId"]: item for item in split_train}
    split_train_keys = split_train.keys()

    split_dev = load_json(split_dev_path)
    split_dev = {item["QuestionId"]: item for item in split_dev}
    split_dev_keys = split_dev.keys()

    new_train_data = []
    new_dev_data = []
    for example in prev_train:
        qid = example["ID"]
        if qid not in split_train_keys and qid not in split_dev_keys:
            print(qid)
        if qid in split_train_keys:
            new_train_data.append(example)
        elif qid in split_dev_keys:
            new_dev_data.append(example)
    print('prev_train: {}'.format(len(prev_train)))
    print('new_train: {}'.format(len(new_train_data)))
    print('new_dev: {}'.format(len(new_dev_data)))
    dump_json(new_train_data, os.path.join(new_folder, 'WebQSP_train.json'))
    dump_json(new_dev_data, os.path.join(new_folder, 'WebQSP_dev.json'))

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
    # train_dev_split_as_per_rng(
    #     'data/WebQSP/generation/merged_yhshu/WebQSP_train.json',
    #     'data/WebQSP/relation_retrieval/candidate_relations_yhshu/webqsp_train_relations.json',
    #     'data/WebQSP/relation_retrieval/candidate_relations_yhshu/webqsp_dev_relations.json',
    #     'data/WebQSP/generation/merged_yhshu_with_dev_split'
    # )
    # for split in ['train', 'test']:
    #     substitude_relations_in_merged_file_with_cross_encoder_res(
    #         f'data/WebQSP/generation/merged_old/WebQSP_{split}.json',
    #         f'data/WebQSP/generation/0722/merged_top5_last5/WebQSP_{split}.json',
    #         f'data/WebQSP/relation_retrieval_0722/candidate_relations/relation_3epochs_question_relation',
    #         topk=5
    #     )
        # calculate_topk_relation_recall(
        #     # f'data/WebQSP/relation_retrieval/candidate_relations_yhshu/WebQSP_{split}_cand_rels_sorted.json',
        #     # f'data/WebQSP/relation_retrieval_0717/candidate_relations/rich_relation_3epochs_question_relation_maxlen_34_ep3_2hop/WebQSP_{split}_cand_rels_sorted.json',
        #     f'data/WebQSP/generation/0722/merged_top5_last5/WebQSP_{split}.json',
        #     # f'data/WebQSP/generation/merged_old/WebQSP_{split}.json',
        #     f'data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.{split}.goldenRelation.json',
        #     topk=10
        # )
    #     validation_merged_file(
    #         f'data/WebQSP/generation/merged_old/WebQSP_{split}.json',
    #         f'data/WebQSP/generation/0722/merged_top5_last5/WebQSP_{split}.json',
    #     )
    # for split in ['train', 'test']:
        # substitude_relations_in_merged_file_2hop(
        #     f'data/WebQSP/generation/merged_old/WebQSP_{split}.json',
        #     f'data/WebQSP/generation/0722/merged_2hop/WebQSP_{split}.json',
        #     f'data/WebQSP/relation_retrieval_0722/candidate_relations/relation_3epochs_question_relation/WebQSP_{split}_2hop_cand_rel_logits.json',
        #     f'data/WebQSP/relation_retrieval_0722/candidate_relations/relation_3epochs_question_relation/',
        #     topk=10
        # )
        # calculate_topk_relation_recall(
        #     # f'data/WebQSP/relation_retrieval/candidate_relations_yhshu/WebQSP_{split}_cand_rels_sorted.json',
        #     # f'data/WebQSP/relation_retrieval_0717/candidate_relations/rich_relation_3epochs_question_relation_maxlen_34_ep3_2hop/WebQSP_{split}_cand_rels_sorted.json',
        #     f'data/WebQSP/generation/0722/merged_2hop/WebQSP_{split}.json',
        #     # f'data/WebQSP/generation/merged_old/WebQSP_{split}.json',
        #     f'data/WebQSP/relation_retrieval_0722/bi-encoder/WebQSP.{split}.goldenRelation.json',
        #     topk=10
        # )
        # validation_merged_file(
        #     f'data/WebQSP/generation/merged_old/WebQSP_{split}.json',
        #     f'data/WebQSP/generation/0722/merged_2hop/WebQSP_{split}.json',
        # )
    train_dev_split_as_per_files(
        'data/WebQSP/generation/0722/merged_question_relation_ep3_2hop/WebQSP_prev_train.json',
        'data/WebQSP/origin/WebQSP.ptrain.json',
        'data/WebQSP/origin/WebQSP.pdev.json',
        'data/WebQSP/generation/0722/merged_question_relation_ep3_2hop/'
    )