GMT-KBQA
==============================

<!-- Project Organization
------------

    ├── LICENSE
    ├── README.md                         <- The top-level README for developers using this project.
    ├── cheatsheet_main_exps.txt          <- running cheatsheet of main experiments
    ├── cheatsheet_data_preparation.txt   <- running cheatsheet for data preparation, e.g., entity detection and disambiguation
    ├── common
    |   ├── components                    <- utility functions
        ├── entity_linker                 <- entity detection, linking and disambiguation
        ├── executor                      <- utility functions for executing SPARQLs
        ├── inputDataset                  <- dataset generation
        ├── models                        <- code for models of GMT-KBQA
        ├── ontology                      <- Freebase ontology information
        ├── qdt_generation                <- Deprecated
        ├── cache
        ├── feature cache
    ├── CWQ
    |   ├── data                          <- source data, ablation data and label maps for CWQ dataset
        ├── scripts                       <- shell scripts for running different setting of GMT-KBQA on CWQ dataset
        ├── exps                          <- saved models and experiment results of GMT-KBQA on CWQ dataset
        ├── saved_models                  <- saved models of relation_detection_and_linking on CWQ
    ├── old                               <- deprecated
    ├── relation_detection_and_linking
    |   ├── biEncoder                     <- code for bi-encoder: data preparation, training, vector generation, index
        ├── crossEncoder                  <- code for cross-encoder: model, training, evaluation, prediction
    ├── WebQSP
    |   ├── data                          <- source data, ablation data and label maps for CWQ dataset
        ├── scripts                       <- shell scripts for running different setting of GMT-KBQA on WebQSP dataset
        ├── saved_models                  <- saved models of relation_detection_and_linking on WebQSP
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p> -->



<!-- ### Logical Form Generation via Multi-task Learning for Complex Question Answering over Knowledge Bases -->

This is an original implementation of paper "Logical Form Generation via Multi-task Learning for Complex Question Answering over Knowledge Bases"
[[Paper PDF](TODO:)]

## Citation
---

```
TODO: cite
```

## Abstract
---
TODO: Brief introduction to GMT-KBQA

<!-- ## Model Description

TODO:


## Code description

TODO:

## Code

The code for question answering is mainly in  `common`.

The code for candidate relations retrieval is mainly in `relation_detection_and_linking`. -->


## Reproducing the Results on CWQ and WebQSP

(1) **Prepare dataset and pretrained checkpoints**

Download the CWQ dataset from [here](https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AACuu4v3YNkhirzBOeeaHYala) and put them under `data/CWQ/origin`. The dataset files should be named as `ComplexWebQuestions_test[train,dev].json`.

Download the WebQSP dataset from [here](https://www.microsoft.com/en-us/research/publication/the-value-of-semantic-parse-labeling-for-knowledge-base-question-answering-2/) and put them under `data/WebQSP/origin`. The dataset files should be named as `WebQSP.test[train].json`.

Setup Freebase: you need to modify variable `FREEBASE_SPARQL_WRAPPER_URL` and `FREEBASE_ODBC_PORT` in `config.py`.

(2) **Parse SPARQL queries to S-expressions** 

As stated in the paper, we generate S-expressions which are not provided by the original dataset.
Here we provide the scripts to parse SPARQL queries to S-expressions. 

Run `python parse_sparql_cwq.py`, and it will augment the original dataset files with s-expressions. 
The augmented dataset files are saved as `data/CWQ/sexpr/CWQ.test[train,dev].json`.

Run `python parse_sparql_webqsp.py` and the augmented dataset files are saved as `data/WebQSP/sexpr/WebQSP.test[train,dev].json`. 

(3) **Retrieve Candidate Entities** 

This step can be ***skipped***, as we've provided the entity retrieval retuls in 
- CWQ: `data/CWQ/entity_retrieval/candidate_entities/CWQ_test[train,dev]_merged_cand_entities_elq_facc1.json`.
- WebQSP: `data/WebQSP/entity_retrieval/candidate_entities/WebQSP_test[train]_merged_cand_entities_elq_facc1.json`

If you want to retrieve the candidate entities from scratch, follow the steps below:

1. Obtain the linking results from ELQ. Firstly you should deploy our tailored [ELQ](https://github.com/WuXuan374/ELQ4GMTQA). Then you should modify variable `ELQ_SERVICE_URL` in `config.py` according to your own ELQ service url. Next run 
    - CWQ: `python detect_and_link_entity.py --dataset CWQ --split test[train,dev] --linker elq` to get candidate entities linked by ELQ. The results will be saved as `data/CWQ/entity_retrieval/candidate_entities/CWQ_test[train,dev]_cand_entities_elq.json`.
    - WebQSP: Run `python detect_and_link_entity.py --dataset WebQSP --split test[train] --linker elq`, and the results will be saved as `data/WebQSP/entity_retrieval/candidate_entities/WebQSP_test[train]_cand_entities_elq.json`

2. Retrieve candidate entities from FACC1. 
    - CWQ: Firstly run
    `python detect_and_link_entity.py --dataset CWQ --split test[train,dev] --linker facc1` to retrieve candidate entities.
    Then run `sh scripts/run_entity_disamb.sh CWQ predict test[train,dev]` to rank the candidates by a BertRanker. The ranked results will be saved as `data/CWQ/entity_retrieval/candidate_entities/CWQ_test[train,dev]_cand_entities_facc1.json`. 

    - WebQSP: Firstly run
    `python detect_and_link_entity.py --dataset WebQSP --split test[train] --linker facc1` to retrieve candidate entities. Then run `sh scripts/run_entity_disamb.sh WebQSP predict test[train]` to rank the candidates by a BertRanker. The ranked results will be saved as `data/WebQSP/entity_retrieval/candidate_entities/WebQSP_test[train]_cand_entities_facc1.json`

    To reproduce our disambiguation results, please download `feature_cache/` and place it under root folder from our checkpoint.

3. Finally, merge the linking results of ELQ and FACC1.  
    - CWQ: `python data_process.py merge_entity --dataset CWQ --split test[train,dev]`, and the final entity retrieval results are saved as `data/CWQ/entity_retrieval/candidate_entities/CWQ_test[train,dev]_merged_cand_entities_elq_facc1.json`. Note for CWQ, entity label will be standardized in final entity retrieval results.
    - WebQSP: `python data_process.py merge_entity --dataset WebQSP --split test[train]`, and the final entity retrieval results are saved as `data/WebQSP/entity_retrieval/candidate_entities/WebQSP_test[train]_merged_cand_entities_elq_facc1.json`.



(4) **Retrieve Candidate Relations** 

This step can also be ***skipped*** , as we've provided the candidate relations in `data/{DATASET}/relation_retrieval/`

If you want to retrive the candidate relations from scratch, follow the steps below:

1. Train the bi-encoder to encode questions and relations.
    - CWQ: Run `python run_relation_data_process.py sample_data --dataset CWQ --split train[dev]` to prepare training data. Then run `sh scripts/run_bi_encoder_CWQ.sh mask_mention` to train bi-encoder model. Trained model will be saved in `data/CWQ/relation_retrieval/bi-encoder/saved_models/mask_mention`.
    - WebQSP: Run `python run_relation_data_process.py sample_data --dataset WebQSP --split train` to prepare training data. Then run `sh scripts/run_bi_encoder_WebQSP.sh rich_relation_3epochs` to train bi-encoder model. Trained model will be saved in `data/WebQSP/relation_retrieval/bi-encoder/saved_models/rich_relation_3epochs`. For WebQSP, linking entities' two-hop relations will be queried and cached.

2. Build the index of encoded relations.
    - CWQ: To encode Freebase relations using trained bi-encoder, run `python relation_retrieval/bi-encoder/build_and_search_index.py encode_relation --dataset CWQ`. Then run `python relation_retrieval/bi-encoder/build_and_search_index.py build_index --dataset CWQ` to build the index of encoded relations. Index file will be saved as `data/CWQ/relation_retrieval/bi-encoder/index/mask_mention/ep_1_flat.index`.
    - WebQSP: To encode Freebase relations using trained bi-encoder, run `python relation_retrieval/bi-encoder/build_and_search_index.py encode_relation --dataset WebQSP`. Then run `python relation_retrieval/bi-encoder/build_and_search_index.py build_index --dataset WebQSP` to build the index of encoded relations. Index file will be saved as `data/WebQSP/relation_retrieval/bi-encoder/index/rich_relation_3epochs/ep_3_flat.index`.

3. Retrieve candidate relations using index.
    - CWQ: First encode questions into vector by running `python relation_retrieval/bi-encoder/build_and_search_index.py encode_question --dataset CWQ --split test[train, dev]`. Then candidate relations can be retrieved using index by running `python relation_retrieval/bi-encoder/build_and_search_index.py retrieve_relations --dataset CWQ --split train[dev, test]`. The retrieved relations will be saved as the training data of cross-encoder in `data/CWQ/relation_retrieval/cross-encoder/mask_mention_1epoch_question_relation/CWQ_test[train,dev].tsv`.
    - WebQSP: First encode questions into vector by running `python relation_retrieval/bi-encoder/build_and_search_index.py encode_question --dataset WebQSP --split train[ptrain, pdev, test]`. Then candidate relations can be retrieved using index by running `python relation_retrieval/bi-encoder/build_and_search_index.py retrieve_relations --dataset WebQSP --split test_2hop[train_2hop, train, test, ptrain, pdev]`. The retrieved relations will be saved as the training data of cross-encoder in `data/WebQSP/relation_retrieval/cross-encoder/rich_relation_3epochs_question_relation/WebQSP_train[ptrain, pdev, test, train_2hop, test_2hop].tsv`.

4. Train the cross-encoder to rank retrieved relations.
    - CWQ: To train, run `sh scripts/run_cross_encoder_CWQ_question_relation.sh train mask_mention_1epoch_question_relation`. Trained models will be saved as `data/CWQ/relation_retrieval/cross-encoder/saved_models/mask_mention_1epoch_question_relation/CWQ_ep_1.pt`. To get inference results, run `sh scripts/run_cross_encoder_CWQ_question_relation.sh predict mask_mention_1epoch_question_relation test[train/dev] CWQ_ep_1.pt`.  Inference result(logits) will be stored in `data/CWQ/relation_retrieval/cross-encoder/saved_models/mask_mention_1epoch_question_relation/CWQ_ep_1.pt_test[train/dev]]`.
    - WebQSP: To train, run `sh scripts/run_cross_encoder_WebQSP_question_relation.sh train rich_relation_3epochs_question_relation`. Trained models will be saved as `data/WebQSP/relation_retrieval/cross-encoder/saved_models/rich_relation_3epochs_question_relation/WebQSP_ep_3.pt`. To get inference results, run `sh scripts/run_cross_encoder_WebQSP_question_relation.sh predict rich_relation_3epochs_question_relation test/[train, train_2hop, test_2hop] WebQSP_ep_3.pt`.Inference result(logits) will be stored in `data/WebQSP/relation_retrieval/cross-encoder/saved_models/rich_relation_3epochs_question_relation/WebQSP_ep_3.pt_test/[train, train_2hop, test_2hop]`.


5. Merge the logits with relations to get sorted relations for each question.
    - CWQ: run `python data_process.py merge_relation --dataset CWQ --split test[train,dev]`. The sorted relations will be saved as `data/CWQ/relation_retrieval/candidate_relations/CWQ_test[train,dev]_cand_rels_sorted.json`
    - WebQSP: run `python data_process.py merge_relation --dataset WebQSP --split test[train, train_2hop, test_2hop]`. The sorted relations will be saved as `data/WebQSP/relation_retrieval/candidate_relations/WebQSP_test[train]_cand_rels_sorted.json`

6. (optional) To only substitude candidate relations in previous merged file, please refer to `substitude_relations_in_merged_file()` in `data_process.py`.

(5) **Generate Logical Forms through multi-task learning**

1.  Prepare all the input data for logical form generation and the two auxiliary tasks (entity disambiguation and relation classification). 
    - CWQ: Run `python data_process.py merge_all --dataset CWQ --split test[train,dev]` The merged data file will be saved as `data/CWQ/generation/merged/CWQ_test[train,dev].json`.

    - WebQSP: Run `python data_process.py merge_all --dataset WebQSP --split test[train]`. The merged data file will be saved as `data/WebQSP/generation/merged/WebQSP_test[train].json`.

2. Training logical form generation model.
    - CWQ: our full model can be trained by running `sh scripts/GMT_KBQA_CWQ.sh train {FOLDER_NAME}`, The trained model will be saved in `exps/CWQ_{FOLDER_NAME}`.
    - WebQSP: our full model can be trained by running `sh scripts/GMT_KBQA_WebQSP.sh train {FOLDER_NAME}`. The trained model will be saved in `exps/WebQSP_{FOLDER_NAME}`.
    - Command for training other model variants mentioned in our paper can be found in `generation_command.txt`.

3. Command for training model(as shown in 2.) will also do inference on `test` split. To inference on other split or inference alone:
    - CWQ: You can run `sh scripts/GMT_KBQA_CWQ.sh predict {FOLDER_NAME} False test 50 4` to do inference on `test` split with `beam_size=50` and `test_batch_size=4`. 
    - WebQSP: You can run `sh scripts/GMT_KBQA_WebQSP.sh predict {FOLDER_NAME} False test 50 2` to do inference on `test` split alone with `beam_size=50` and `test_batch_size = 2` . 
    - Command for inferencing on other model variants can be found in `generation_command.txt`.

4. To evaluate trained models:
    - CWQ: Run `python3 eval_topk_prediction_final.py --split test --pred_file exps/CWQ_GMT_KBQA/beam_50_test_4_top_k_predictions.json --test_batch_size 4 --dataset CWQ`
    - WebQSP: Run `python3 eval_topk_prediction_final.py --split test --pred_file exps/WebQSP_GMT_KBQA/beam_50_test_2_top_k_predictions.json --test_batch_size 2 --dataset WebQSP`

(6) **Ablation experiments**
1. Evaluate entity linking and relation linking result:
    - CWQ: Run `python ablation_exps.py linking_evaluation --dataset CWQ`
    - WebQSP: Run `python ablation_exps.py linking_evaluation --dataset WebQSP`
2. Evaluate QA performance on quesitions with unseen entity/relation
    - CWQ: Run `python ablation_exps.py unseen_evaluation --dataset CWQ --model_type full` to get evaluation result on our full model `GMT-KBQA`. Run `python ablation_exps.py unseen_evaluation --dataset CWQ --model_type base` to get evaluation result on `T5-base` model.
    - WebQSP: Run `python ablation_exps.py unseen_evaluation --dataset WebQSP --model_type full` to get evaluation result on our full model `GMT-KBQA`. Run `python ablation_exps.py unseen_evaluation --dataset WebQSP --model_type base` to get evaluation result on `T5-base` model.


## Candidate relation retrieval
all codes are under relation_detection_and_linking/ folder
### Bi-encoder

You can download trained bi-encoder models, question, relation vectors and index file from [TODO:](TODO:).

all codes are under Relation_retrieval/bi-encoder folder

#### Data preparation
You can refer to main() function in process_data.py.

#### Training
You can refer to cheatsheet_bi_encoder.txt for specific running commands. Path for saving models and logs can be configured by modifying scripts under scripts/ folder.

#### Caching and retrieval
- embedding relations to vectors for inference effieciency
- building index for relations using FAISS
- embedding questions to vectors
- retreival nearest relations using FAISS & construct source data for cross-encoder
Above functions can be found in main() function in build_faiss_index.py

### Cross-encoder

Source data can be found in (TODO:)[TODO]

all codes are under Relation_retrieval/cross-encoder folder

#### Training, evaluation and prediction
- Please refer to cheatsheet_cross_encoder.txt, with specific instructions

#### add retrieved relations to dataset of GMT-KBQA
- Get retrieved relations based on logits.
- Evaluation of P,R,F1 on retrieved relations under different settings
- Add retrieved relations to dataset of GMT-KBQA
Please refer to data_process.py

## Candidate entity retrieval
The source data of generation & question answering under Data/{CWQ}/{WebQSP}/generation/merged, has already integrated entity linking results.
If you want to retrieve entities by your own, please refer to Entity_retrieval/cheatsheet_data_preparation.txt.

### add retrieved entities to dataset of GMT-KBQA

## Logical form generation and question answering
- all codes are under Generation/ folder.
- cheatsheet_main_exps.txt contains running commands for all models (including variants) mentioned in our paper.
- You can refer to run_multitask_generator_final.py for details of training and prediction of GMT-KBQA
- You can refer to eval_topk_prediction_final.py for logical form grounding as well as evaluation of GMT-KBQA. Note: you should setup Freebase checkpoint as well as ODBC connection before grounding, as shown in executor/sparql_executor.py

### Ablation study
ablation_exps.py includes code for following ablation experiments:
- Entity and relation linking evaluation
- QA results on set questions with unseen KB items.

## Contact

For any question, please contact [TODO:](TODO)


## 测试进度说明
- 以本说明为准，不要重复测试了
### Parse SPARQL queries to S-expressions
- 测试过了
### Retrieve Candidate Entities
#### WebQSP
- 最后的结果在 `data/WebQSP/entity_retrieval`目录下; 这里头的结果和论文所使用的的结果基本上完全一致
- **tldr: 只有 disamb_entities，用代码生成的结果和论文中使用的结果差距比较大；其他文件的差距都很小，可忽略**
    - 需要做的: 可能要分别用 linking_results/目录下的消岐结果和 disamb_entities/下的消岐结果跑一下没有实体链接的模型；或者在 github 中说明这部分使用的文件与代码不同
- 其他:
    - _elq 文件和之前的完全一致
    - _facc1_unranked 之前没有生成，无法比较
    - _facc1 文件 train 和 test 各有10来处不同，但经过检查，主要都是 logit 的细微变化，导致实体排序的细微变化，没什么影响，不考虑了
    - _merged_elq_facc1.json: train 10 处, test 5处，同样基本上就是一些实体的顺序问题，无影响
    - disamb_entities: 对比 data/WebQSP/entity_retrieval_0724/disamb_entities 和 /home3/xwu/workspace/QDT2SExpr/CWQ/all_data_bk/WebQSP/final/entity_linking_results/
        - 这一项的差距很大，train 有 2380 个不同，test 有 462 个不同（其实我们只会用到 test）-- 只考虑 id 和 label 的情况下

#### CWQ
- 最后的结果在 `data/CWQ/entity_retrieval`目录下; 这里头的结果和论文所使用的的结果基本上完全一致
- **tldr: 只有 disamb_entities，用代码生成的结果和论文中使用的结果差距比较大；其他文件的差距都很小，可忽略**
    - 需要做的: 可能要分别用 merged_linking_results/目录下的消岐结果和 disamb_entities/下的消岐结果跑一下没有实体链接的模型；或者在 github 中说明这部分使用的文件与代码不同
- 其他
    - _elq 文件是直接从论文对应的数据那边复制过来的
    - _facc1_unranked 也直接从 candidate_entities 那边 复制过来
    - _facc1 文件的格式不一样，不直接比较
    - 直接比较 merged_ 文件: （考虑 id 和 label）
        - train: 78 处不同
        - dev： 15 处不同
        - test: 13 处不同
        - 这基本上就没关系了
    - disamb_entities
        - train 有 22135 处不同
        - dev 2837 处不同
        - test 2837 处不同
### Retrieve Candidate Relations
#### WebQSP
- 已全部测试过了，bug 也修复了，最后的结果都在`data/WebQSP/relation_retrieval_final`目录下(已更名为`data/WebQSP/relation_retrieval`)，使用的 merged 文件在`data/WebQSP/generation/merged_relation_final`下(已更名为`data/WebQSP/generation/merged`)
- 每次跑 Logits 的 inference 时，生成的 logits 会有细微不同，可能导致 top 10 关系发生变化，test 里头可能有20多条会变化；但应该是正常的，因为排序靠后的这些 logits 很接近
#### CWQ
- 已经全部测试过了（新跑了一遍生成），最后的结果在`data/CWQ/relation_retrieval_0723` 目录下(已更名为`data/CWQ/relation_retrieval`)，使用的 merged 文件在 `data/CWQ/generation/merged_0724_ep1`(已更名为`data/CWQ/generation/merged`)
- 每次跑 Logits 的 inference 时，生成的 logits 会有细微不同，可能导致 top 10 关系发生变化，test 里头可能有40 多条会变化；但应该是正常的，因为排序靠后的这些 logits 很接近

### Preparing Logical Form generation data
- 总体来说，代码生成的结果和原来差不多
- cand_entity_list 可能不同，原因是一方面新生成的实体有一些不一样，另一方面有一些实体是随机得到的，每次结果不一样
- gold_relation_map 可能不同，使用了新的正则表达式（WebQSP 使用所有的 golden relations）
- label_maps 的生成: 
    - WebQSP 改用了所有 parses 的 label_maps, **现在位于label_maps/** 目录下；经检查，代码没有问题
    - CWQ 使用代码生成的 label_maps 位于 label_maps_test 下，和原来的 label_maps 有: train 113, dev 10, test 9 处不同，问题不大，主要原因是关系抽取正则表达式的修改。**现在位于label_maps/ 目录下。**
#### CWQ
- CWQ: train 1329, dev 144, test 162, diff key: ['gold_relation_map', 'cand_entity_list']
#### WebQSP
- train 667, test: 370, key: ['cand_entity_list', 'sparql', 'gold_type_map', 'gold_relation_map', 'gold_entity_map']
- gold_*_map: 从 1parse 到所有 parse
- sparql 不同，但是 sexpr 都是相同的：应该是以前的数据，sparql 没有完全按照 sexpr 的选择规则
    - 不过考虑到生成模型中没有使用到 sparql (训练的时候是用 sexpr 作为生成目标)，测试的时候最终也是计算QA效果，这个 bug 没什么影响
## TODO
- * 检查二跳关系的函数
    - 函数能生成相同结果，entity_id 的来源也找到了
- 删掉没用的代码
- 所有的绝对路径等放到 config.py 里头
- common_data 下的一些数据，对应 relation_retrieval 里头的函数，确认一下

## 实体链接存在的问题
- /home2/xxhu/QDT2SExpr/CWQ/data/linking_results/WebQSP_train_elq_results.json 其实是测试集的链接结果，等于最后的 merged_data 里头只有 facc1 的结果
    - 只有 WebQSP 训练集有这个问题