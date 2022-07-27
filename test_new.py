from components.utils import dump_json, load_json
import pandas as pd
from tqdm import tqdm


def compare_json_file(file_1_path, file_2_path):
    print(file_1_path)
    file_1 = load_json(file_1_path)
    file_2 = load_json(file_2_path)
    print(file_1 == file_2)
    assert file_1 == file_2

def compare_json_file_set(file_1_path, file_2_path):
    print(file_1_path)
    file_1 = set(load_json(file_1_path))
    file_2 = set(load_json(file_2_path))
    print(file_1 == file_2)
    assert file_1 == file_2

def compare_json_relations_sequence(file_1_path, file_2_path):
    """
    每次 Inference, 关系的 logits 是可能有微小不同的，我们考虑关系的顺序即可
    """
    prev_data = load_json(file_1_path)
    new_data = load_json(file_2_path)
    assert len(prev_data) == len(new_data), print(len(prev_data), len(new_data))
    for (prev, new) in tqdm(zip(prev_data, new_data), total=len(prev_data)):
        for key in prev.keys():
            if key != 'cand_relation_list':
                assert prev[key] == new[key]
            else:
                prev_relations = [rel[0] for rel in prev['cand_relation_list']]
                new_relations = [rel[0] for rel in new['cand_relation_list']]
                if set(prev_relations) != set(new_relations):
                    print(prev["ID"])

def compare_json_disamb_entities(file_1_path, file_2_path):
    print(file_1_path)
    file_1 = load_json(file_1_path)
    file_2 = load_json(file_2_path)
    assert len(file_1) == len(file_2), print(len(file_1), len(file_2))
    not_equal_num = 0
    for qid in file_1:
        assert qid in file_2, print(qid)
        if len(file_1[qid]) == 0:
            file_1_items = []
        else:
            file_1_items = [{
                'id': key,
                'label': value['label']
            } for (key, value) in file_1[qid].items()]

        file_2_items = [{
            'id': item["id"],
            'label': item["label"]
        } for item in file_2[qid]]
        if file_1_items != file_2_items:
           not_equal_num += 1
    print('not_equal_num: {}'.format(not_equal_num))


def compare_merged_entities(file_1_path, file_2_path):
    """
    1. 随机采样的实体不进行对比（特点，没有 label）
    2. 只考虑实体的 id
    """
    print(file_1_path)
    print(file_2_path)
    file_1 = load_json(file_1_path)
    file_2 = load_json(file_2_path)
    assert len(file_1) == len(file_2), print(len(file_1), len(file_2))
    not_equal_num = 0
    for qid in file_1:
        assert qid in file_2, print('qid not found: {}'.format(qid))
        file_1_entities = [(ent['id'], ent['label']) for ent in file_1[qid] if 'mention' in ent]
        file_2_entities = [(ent['id'], ent['label']) for ent in file_2[qid] if 'mention' in ent]
        if file_1_entities != file_2_entities:
            not_equal_num += 1
    print('not_equal: {}'.format(not_equal_num))

def compare_json_file_diy(file_1_path, file_2_path):
    print(file_1_path)
    file_1 = load_json(file_1_path)
    file_2 = load_json(file_2_path)
    file_1_keys = file_1.keys()
    file_2_keys = file_2.keys()
    print('file_1_keys: {}'.format(len(file_1_keys)))
    print('file_2_keys: {}'.format(len(file_2_keys)))
    for key in file_1_keys:
        assert key in file_2_keys
        file_1_relations = set(file_1[key])
        file_2_relations = set(file_2[key])
        # file1 是 file2 的子集
        diff = [rel for rel in file_1_relations if rel not in file_2_relations]
        if len(diff) > 0:
            print(len(diff))

def compare_two_hop_relations_map(file_1_path, file_2_path):
    print(file_1_path)
    file_1 = load_json(file_1_path)
    file_2 = load_json(file_2_path)
    assert len(file_1) == len(file_2)
    for eid in file_1:
        assert eid in file_2
        print(len(file_1[eid]), len(file_2[eid]))
        assert set(file_1[eid]) == set(file_2[eid]), print(eid)

def compare_two_hop_relations(file_1_path, file_2_path):
    print(file_1_path)
    file_1 = load_json(file_1_path)
    file_2 = load_json(file_2_path)
    assert len(file_1) == len(file_2)
    for (data_1, data_2) in zip(file_1, file_2):
        assert data_1["id"] == data_2["id"]
        print(len(set(data_1["two_hop_relations"])))
        print(len(set(data_2["two_hop_relations"])))
        assert set(data_1["two_hop_relations"]) == set(data_2["two_hop_relations"]), print(data_1["id"])

def compare_json_keys(file_1_path, file_2_path):
    file_1 = load_json(file_1_path)
    file_2 = load_json(file_2_path)
    print(file_1.keys() == file_2.keys())
    assert file_1.keys() == file_2.keys()

def compare_tsv_file(tsv_1_path, tsv_2_path):
    print(tsv_1_path)
    """希望 tsv 文件内容完全一致"""
    tsv_1_df = pd.read_csv(tsv_1_path, delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
    tsv_2_df = pd.read_csv(tsv_2_path, delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
    assert list(tsv_1_df['relation'].values) == list(tsv_2_df['relation'].values)
    print(list(tsv_1_df['relation'].values) == list(tsv_2_df['relation'].values))

def compare_tsv_file_ptrain_pdev(ptrain_path, pdev_path, train_path):
    """
    总量: train = ptrain + pdev
    golden 关系: train = ptrain + pdev
    """
    ptrain_df = pd.read_csv(ptrain_path, delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
    pdev_df = pd.read_csv(pdev_path, delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
    train_df = pd.read_csv(train_path, delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
    ptrain_golden = ptrain_df[ptrain_df['label'] == 1]
    pdev_golden = pdev_df[pdev_df['label'] == 1]
    train_golden = train_df[train_df['label'] == 1]
    assert len(train_golden) == len(ptrain_golden) + len(pdev_golden)
    assert set(list(train_golden['relation'].values)) == (set(list(ptrain_golden['relation'].values)) | set(list(pdev_golden['relation'].values)))

def compare_tsv_file_randomly_sampled(tsv_1_path, tsv_2_path):
    """
    有一些 tsv 文件是通过随机采样得到的，因此只能比较其中 golden 关系的集合是否相等
    """
    tsv_1_df = pd.read_csv(tsv_1_path, delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
    tsv_2_df = pd.read_csv(tsv_2_path, delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
    tsv_1_golden = tsv_1_df[tsv_1_df['label'] == 1]['relation'].unique()
    tsv_2_golden = tsv_2_df[tsv_2_df['label'] == 1]['relation'].unique()
    print(len(tsv_1_golden), len(tsv_2_golden))
    assert set(tsv_1_golden) == set(tsv_2_golden)

def compare_tsv_file_randomly_sampled_question(tsv_1_path, tsv_2_path):
    """
    有一些 tsv 文件是通过随机采样得到的，因此只能比较其中 golden 关系的集合是否相等
    还要比较 question 的集合是否相等
    """
    tsv_1_df = pd.read_csv(tsv_1_path, delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
    tsv_2_df = pd.read_csv(tsv_2_path, delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
    print(len(tsv_1_df) == len(tsv_2_df), print(len(tsv_1_df), len(tsv_2_df)))
    tsv_1_golden = tsv_1_df[tsv_1_df['label'] == 1]['relation'].unique()
    tsv_2_golden = tsv_2_df[tsv_2_df['label'] == 1]['relation'].unique()
    tsv_1_question = tsv_1_df['question'].unique()
    tsv_2_question = tsv_2_df['question'].unique()
    print(len(tsv_1_golden), len(tsv_2_golden))
    assert set(tsv_1_golden) == set(tsv_2_golden)
    print(len(tsv_1_question), len(tsv_2_question))
    assert set(tsv_1_question) == set(tsv_2_question)

def compare_sorted_relations(sorted_relations_1_path, sorted_relations_2_path):
    print(sorted_relations_1_path)
    sorted_relations_1 = load_json(sorted_relations_1_path)
    sorted_relations_2 = load_json(sorted_relations_2_path)
    # 我们只会用到 top-10 的关系，所以比较 top-10 即可
    assert len(sorted_relations_1) == len(sorted_relations_2)
    for qid in sorted_relations_1:
        assert qid in sorted_relations_2
        relations_1_top10 = sorted_relations_1[qid][:10]
        relations_2_top10 = sorted_relations_2[qid][:10]
        if relations_1_top10 != relations_2_top10:
            if set(relations_1_top10) != set(relations_2_top10):
                # 会出现一些不同，应该是 logits 过于接近的情况
                print(qid)

def compare_json_label_mention_id(json_path_1, json_path_2):
    print(json_path_1)
    file_1 = load_json(json_path_1)
    file_2 = load_json(json_path_2)
    for qid in file_1:
        if qid not in file_2:
            print('qid not found: {}'.format(qid))
        file_1_item = [{
            'id': item["id"],
            'label': item["label"],
            'mention': item["mention"]
        } for item in file_1[qid]]
        file_2_item = [{
            'id': item["id"],
            'label': item["label"],
            'mention': item["mention"]
        } for item in file_2[qid]]
        if file_1_item != file_2_item:
            print('not equal: {}'.format(qid))

def relation_data_process_unit_test():
    """
    CWQ
    """
    for split in ['train', 'dev', 'test']:
        compare_json_file(
            f'data/CWQ/relation_retrieval_0723/bi-encoder/CWQ.{split}.goldenRelation.json',
            f'data/CWQ/relation_retrieval/bi-encoder/CWQ.{split}.goldenRelation.json'
        )
    for split in ['train', 'dev']:
        # 得到的结果稍有不同，不纠结了, 直接把源目录的复制过来吧
        compare_tsv_file_randomly_sampled_question(
            f'data/CWQ/relation_retrieval_0723/bi-encoder/CWQ.{split}.sampled.tsv',
            f'data/CWQ/relation_retrieval/bi-encoder/CWQ.{split}.sampled.tsv'
        )


    """
    WebQSP
    """
    # print('Comparing golden relations')
    # for split in ['train', 'ptrain', 'pdev', 'test']:
    #     compare_json_file(
    #         f'data/WebQSP/relation_retrieval/bi-encoder/WebQSP.{split}.goldenRelation.json',
    #         f'data/WebQSP/relation_retrieval_final/bi-encoder/WebQSP.{split}.goldenRelation.json'
    #     )
    # print('Comparing tsv files')
    # compare_tsv_file_randomly_sampled(
    #     f'data/WebQSP/relation_retrieval_final/bi-encoder/WebQSP.train.sampled.tsv',
    #     f'data/WebQSP/relation_retrieval/bi-encoder/WebQSP.train.sampled.tsv',
    # )

    # for split in ['ptrain', 'pdev']:
    #     compare_json_file(
    #         f'data/WebQSP/origin/WebQSP.{split}_repeat.json',
    #         f'data/WebQSP/origin/WebQSP.{split}.json'
    #     )
    # for split in ['train', 'ptrain', 'pdev', 'test']:
    #     compare_json_file(
    #         f'data/WebQSP/relation_retrieval_0722/bi-encoder/WebQSP.{split}.goldenRelation.json',
    #         f'data/WebQSP/relation_retrieval_final/bi-encoder/WebQSP.{split}.goldenRelation.json'
    #     )
    # compare_tsv_file_ptrain_pdev(
    #     'data/WebQSP/relation_retrieval_0722/bi-encoder/WebQSP.ptrain.sampled.tsv',
    #     'data/WebQSP/relation_retrieval_0722/bi-encoder/WebQSP.pdev.sampled.tsv',
    #     'data/WebQSP/relation_retrieval_0722/bi-encoder/WebQSP.train.sampled.tsv',
    # )
    # compare_tsv_file_randomly_sampled(
    #     'data/WebQSP/relation_retrieval_0722/bi-encoder/WebQSP.train.sampled.tsv',
    #     'data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.train.sampled.normal.tsv',
    # )
    # 二跳关系查询的检查
    # print('Comparing unique entity ids')
    # compare_json_file_set(
    #     'data/WebQSP/relation_retrieval/cross-encoder/rng_kbqa_linking_results/unique_entity_ids.json',
    #     'data/WebQSP/relation_retrieval_final/cross-encoder/rng_kbqa_linking_results/unique_entity_ids.json',
    # )
    # compare_json_keys(
    #     'data/WebQSP/relation_retrieval_0722/cross-encoder/rng_kbqa_linking_results/entities_2hop_relations.json',
    #     'data/WebQSP/relation_retrieval_0717/cross-encoder/rng_kbqa_linking_results/entities_2hop_relations.json'
    # )
    # for split in ['train', 'test']:
    #     compare_two_hop_relations(
    #         f'data/WebQSP/relation_retrieval_0717/cross-encoder/rng_kbqa_linking_results/webqsp_{split}_rng_el_two_hop_relations.json',
    #         f'data/WebQSP/relation_retrieval_0722/cross-encoder/rng_kbqa_linking_results/webqsp_{split}_rng_el_two_hop_relations.json',
    #     )
        # compare_two_hop_relations_map(
        #     'data/WebQSP/relation_retrieval_0717/cross-encoder/rng_kbqa_linking_results/entities_2hop_relations.json',
        #     'data/WebQSP/relation_retrieval_0722/cross-encoder/rng_kbqa_linking_results/entities_2hop_relations.json',
        # )
        

def relation_retrieval_unit_test():
    """
    CWQ, 到cross-encoder 的训练数据这边，确认无误，可复现
    """
    # for split in ['test']:
    #     compare_tsv_file(
    #         f'data/CWQ/relation_retrieval_0723/cross-encoder/mask_mention_1epoch_question_relation/CWQ.{split}.tsv',
    #         f'data/CWQ/relation_retrieval/cross-encoder/mask_mention_1epoch_question_relation/CWQ.{split}.tsv',
    #     )

    """
    WebQSP
    """
    for split in ['train', 'ptrain', 'pdev', 'test', 'train_2hop', 'test_2hop']:
        compare_tsv_file(
            f'data/WebQSP/relation_retrieval_final/cross-encoder/rich_relation_3epochs_question_relation/WebQSP.{split}.tsv',
            f'data/WebQSP/relation_retrieval/cross-encoder/rich_relation_3epochs_question_relation/WebQSP.{split}.tsv',
        )
        # compare_tsv_file(
        #     f'data/WebQSP/relation_retrieval_final/cross-encoder/rich_relation_3epochs_question_relation/WebQSP.{split}_2hop.tsv',
        #     f'data/WebQSP/relation_retrieval_0717/cross-encoder/2hop_relations/WebQSP.{split}.tsv'
        # )
        # compare_sorted_relations(
        #     f'data/WebQSP/relation_retrieval_final/candidate_relations/rich_relation_3epochs_question_relation/WebQSP_{split}_cand_rels_sorted.json',
        #     f'data/WebQSP/relation_retrieval_0717/candidate_relations/rich_relation_3epochs_question_relation_maxlen_34_ep3/WebQSP_{split}_cand_rels_sorted.json'
        # )
        # compare_sorted_relations(
        #     f'data/WebQSP/relation_retrieval_final/candidate_relations/rich_relation_3epochs_question_relation/WebQSP_{split}_2hop_cand_rels_sorted.json',
        #     f'data/WebQSP/relation_retrieval_0717/candidate_relations/rich_relation_3epochs_question_relation_maxlen_34_ep3_2hop/WebQSP_{split}_cand_rels_sorted.json'
        # )
    # compare_tsv_file_ptrain_pdev(
    #     'data/WebQSP/relation_retrieval_final/cross-encoder/rich_relation_3epochs_question_relation/WebQSP.ptrain.tsv',
    #     'data/WebQSP/relation_retrieval_final/cross-encoder/rich_relation_3epochs_question_relation/WebQSP.pdev.tsv',
    #     'data/WebQSP/relation_retrieval_final/cross-encoder/rich_relation_3epochs_question_relation/WebQSP.train.tsv',
    # )
    # compare_json_file_diy(
    #     'data/WebQSP/relation_retrieval/cross-encoder/rng_linking_results/WebQSP.2hopRelations.rng.elq.candEntities.json',
    #     'data/WebQSP/relation_retrieval_final/cross-encoder/rng_kbqa_linking_results/entities_2hop_relations.json'
    # )

def merge_logits_unit_test():
    # for split in ['train', 'dev', 'test']:
    for split in ['train', 'test']:
        # 每次 predict 的时候，关系 logits 可能有轻微的变动，比较关系的顺序应该就可以？
        # compare_json_relations_sequence(
        #     f'data/CWQ/generation/merged_0724_ep1/CWQ_{split}.json',
        #     f'data/CWQ/generation/merged_test/CWQ_{split}.json',
        # )
        compare_json_relations_sequence(
            f'data/WebQSP/generation/merged_relation_final/WebQSP_{split}.json',
            f'data/WebQSP/generation/merged_test/WebQSP_{split}.json',
        )
    # compare_json_file(
    #     f'data/WebQSP/generation/merged_relation_final/WebQSP_train_all.json',
    #     f'data/WebQSP/generation/merged_question_relation_ep3_2hop/WebQSP_train.json',
    # )
    # compare_json_file(
    #     f'data/WebQSP/generation/merged_relation_final/WebQSP_test.json',
    #     f'data/WebQSP/generation/merged_question_relation_ep3_2hop/WebQSP_test.json',
    # )

def compare_2hop_relations():
    compare_json_file_diy(
        'data/WebQSP/relation_retrieval_final/cross-encoder/rng_kbqa_linking_results/WebQSP.2hopRelations.rng.elq.candEntities.json',
        'data/WebQSP/relation_retrieval_final/cross-encoder/rng_kbqa_linking_results/entities_2hop_relations.json'
    )

def compare_elq_cand_entities(dataset):
    if dataset.lower() == 'webqsp':
        for split in ['train', 'test']:
            compare_json_file(
                f'data/WebQSP/entity_retrieval/candidate_entities/WebQSP_{split}_cand_entities_elq.json',
                f'/home3/xwu/workspace/QDT2SExpr/CWQ/all_data_bk/WebQSP/entity_linking_0423/WEBQSP_{split}_cand_entities_elq.json'
            )
    else:
        for split in ['train', 'dev', 'test']:
            compare_json_file(
                f'data/CWQ/entity_retrieval_0724/candidate_entities/CWQ_{split}_cand_entities_elq.json',
                f'/home3/xwu/workspace/QDT2SExpr/CWQ/all_data_bk/CWQ/entity_linking_0414/CWQ_{split}_cand_entities_elq.json'
            )

def compare_facc1_ranked_entities(dataset):
    if dataset.lower() == 'webqsp':
        for split in ['train', 'test']:
            compare_json_label_mention_id(
                f'data/WebQSP/entity_retrieval/candidate_entities/WebQSP_{split}_cand_entities_facc1.json',
                f'/home3/xwu/workspace/QDT2SExpr/CWQ/all_data_bk/WebQSP/entity_linking_0423/WEBQSP_{split}_cand_entities_facc1.json'
            )
    else:
        for split in ['train', 'dev', 'test']:
            compare_json_label_mention_id(
                f'data/CWQ/entity_retrieval_0724/candidate_entities/CWQ_{split}_cand_entities_facc1.json',
                f'/home3/xwu/workspace/QDT2SExpr/CWQ/all_data_bk/CWQ/entity_linking_0414/CWQ_{split}_entities_FACC1_disamb.json'
            )
    

def compare_combined_entities(dataset):
    if dataset.lower() == 'webqsp':
        for split in ['train', 'test']:
            compare_merged_entities(
                f'data/WebQSP/entity_retrieval/candidate_entities/WebQSP_{split}_merged_cand_entities_elq_facc1.json',
                f'/home3/xwu/workspace/QDT2SExpr/CWQ/all_data_bk/WebQSP/entity_linking_0423/WEBQSP_{split}_cand_entities_merged_elq_FACC1.json'
            )
    else:
        for split in ['train', 'dev', 'test']:
            compare_merged_entities(
                f'data/CWQ/entity_retrieval/candidate_entities/CWQ_{split}_merged_cand_entities_elq_facc1.json',
                f'/home3/xwu/workspace/QDT2SExpr/CWQ/all_data_bk/CWQ/entity_linking_0414/CWQ_{split}_merged_elq_FACC1.json'
            )

def compare_disamb_entities(dataset):
    if dataset.lower() == 'webqsp':
        for split in ['train', 'test']:
            compare_json_disamb_entities(
                f'/home3/xwu/workspace/QDT2SExpr/CWQ/all_data_bk/WebQSP/final/entity_linking_results/merged_WebQSP_{split}_linking_results.json',
                f'data/WebQSP/entity_retrieval/disamb_entities/WebQSP_merged_{split}_disamb_entities.json'
            )
    else:
        for split in ['train', 'dev', 'test']:
            compare_json_disamb_entities(
                f'data/CWQ/entity_retrieval/merged_linking_results/merged_CWQ_{split}_linking_results.json',
                f'data/CWQ/entity_retrieval/disamb_entities/CWQ_merged_{split}_disamb_entities.json'
            )

def entity_retrieval_unit_test(dataset='WebQSP'):
    # compare_elq_cand_entities(dataset)
    # compare_facc1_ranked_entities(dataset)
    # compare_combined_entities(dataset)
    compare_disamb_entities(dataset)


        

def main():
    # 实体检索相关单元测试
    # entity_retrieval_unit_test('CWQ')

    # 关系检索相关的单元测试
    # relation_data_process_unit_test()
    # relation_retrieval_unit_test()
    merge_logits_unit_test()
    # compare_2hop_relations()
    # for split in ['train', 'test']:
    #     compare_json_file(
    #         f'data/CWQ/generation/merged_0724_ep1/CWQ_{split}.json',
    #         f'data/CWQ/generation/merged_old/CWQ_{split}.json'
    #     )

if __name__=='__main__':
    main()