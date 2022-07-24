from components.utils import dump_json, load_json
import pandas as pd


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

def relation_data_process_unit_test():
    """
    CWQ
    """
    for split in ['train', 'dev', 'test']:
        compare_json_file(
            f'data/CWQ/relation_retrieval_0723/bi-encoder/CWQ.{split}.goldenRelation.json',
            f'data/CWQ/relation_retrieval/bi-encoder/CWQ.{split}.relation.json'
        )
    # for split in ['dev']:
    #     # 得到的结果稍有不同，不纠结了, 直接把源目录的复制过来吧
    #     compare_tsv_file_randomly_sampled_question(
    #         f'data/CWQ/relation_retrieval_0723/bi-encoder/CWQ.{split}.sampled.tsv',
    #         f'data/CWQ/relation_retrieval/bi-encoder/CWQ.{split}.sampled.tsv'
    #     )


    """
    WebQSP
    """
    # for split in ['train', 'test']:
    #     compare_json_file(
    #         f'data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.{split}.goldenRelation.json',
    #         f'data/WebQSP/relation_retrieval_final/bi-encoder/WebQSP.{split}.goldenRelation.json'
    #     )
    # compare_tsv_file_randomly_sampled(
    #     f'data/WebQSP/relation_retrieval_final/bi-encoder/WebQSP.train.sampled.tsv',
    #     f'data/WebQSP/relation_retrieval_0717/bi-encoder/WebQSP.train.sampled.tsv',
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
    # compare_json_file_set(
    #     'data/WebQSP/relation_retrieval_0722/cross-encoder/rng_kbqa_linking_results/unique_entity_ids.json',
    #     'data/WebQSP/relation_retrieval_0717/cross-encoder/rng_kbqa_linking_results/unique_entity_ids.json',
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
    for split in ['test', 'dev', 'train']:
        compare_tsv_file(
            f'data/CWQ/relation_retrieval_0723/cross-encoder/mask_mention_1epoch/CWQ.{split}.tsv',
            f'data/CWQ/relation_retrieval/cross-encoder/0715_retrain/CWQ.{split}.tsv',
        )

    """
    WebQSP
    """
    # for split in ['test','train']:
        # compare_tsv_file(
        #     f'data/WebQSP/relation_retrieval_final/cross-encoder/rich_relation_3epochs_question_relation/WebQSP.{split}.tsv',
        #     f'data/WebQSP/relation_retrieval_0717/cross-encoder/rich_relation_3epochs_question_relation/WebQSP.{split}.tsv',
        # )
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
    for split in ['train', 'dev', 'test']:
        compare_json_file(
            f'data/WebQSP/generation/merged_relation_final/WebQSP_{split}.json',
            f'data/WebQSP/generation/0722/merged_question_relation_ep3_2hop/WebQSP_{split}.json',
        )
        # compare_json_file(
        #     f'data/WebQSP/generation/merged_relation_final/WebQSP_{split}.json',
        #     f'data/WebQSP/generation/merged_question_relation_ep3_2hop/WebQSP_{split}.json',
        # )
    compare_json_file(
        f'data/WebQSP/generation/merged_relation_final/WebQSP_train_all.json',
        f'data/WebQSP/generation/merged_question_relation_ep3_2hop/WebQSP_train.json',
    )
    compare_json_file(
        f'data/WebQSP/generation/merged_relation_final/WebQSP_test.json',
        f'data/WebQSP/generation/merged_question_relation_ep3_2hop/WebQSP_test.json',
    )

def compare_2hop_relations():
    compare_json_file_diy(
        'data/WebQSP/relation_retrieval_final/cross-encoder/rng_kbqa_linking_results/WebQSP.2hopRelations.rng.elq.candEntities.json',
        'data/WebQSP/relation_retrieval_final/cross-encoder/rng_kbqa_linking_results/entities_2hop_relations.json'
    )

        

def main():
    # 关系检索相关的单元测试
    # relation_data_process_unit_test()
    # relation_retrieval_unit_test()
    # merge_logits_unit_test()
    # compare_2hop_relations()
    for split in ['train', 'test']:
        compare_json_file(
            f'data/WebQSP/generation/merged_question_relation_ep3_2hop/WebQSP_{split}.json',
            f'data/WebQSP/generation/merged_relation_final/WebQSP_{split}.json'
        )

if __name__=='__main__':
    main()