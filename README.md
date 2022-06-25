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


## Reproducing the Results on CWQ
(1) **Prepare dataset and pretrained checkpoints**

Download the CWQ dataset from [here](https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AACuu4v3YNkhirzBOeeaHYala) and put them under `data/CWQ/origin`. The dataset files should be named as `ComplexWebQuestions_test[train,dev].json`.

(2) **Parse SPARQL queries to S-expressions**

As stated in the paper, we generate S-expressions which are not provided by the original dataset.
Here we provide the scripts to parse SPARQL queries 

Run `python parse_sparql_cwq.py`, and it will augment the original dataset files with s-expressions. 
The augmented dataset files are saved as `data/CWQ/sexpr/CWQ.test[train,dev].json`.

(3) **Retrieve Candidate Entities**

This step can be ***skipped***, as we've provided the entity retrieval retuls in `data/CWQ/entity_retrieval/candidate_entities/CWQ_test[train,dev]_merged_cand_entities_elq_FACC1.json`.
0623吴轩: `data/CWQ/entity_retrieval/candidate_entities_xwu/CWQ_test[train,dev]_merged_cand_entities_elq_FACC1.json`

If you want to retrieve the candidate entities from scratch, follow the steps below:

1. Obtain the linking results from ELQ. Firstly you should deploy our tailored [ELQ](). Then run `python detect_and_link_entity.py --dataset CWQ --split test[train,dev] --linker elq` to get candidate entities linked by ELQ. The results will be saved as `data/CWQ/entity_retrieval/candidate_entities/CWQ_test[train,dev]_cand_entities_elq.json`.
0623吴轩: 这一步我直接复制 /home3/xwu/workspace/QDT2SExpr/CWQ/all_data_bk/CWQ/entity_linking_0414 目录下的结果，经对比，和学长生成的结果差距在于测试集有45个问题的链接结果为空，但这只能让性能更差吧应该


2. Retrieve candidate entities from FACC1. Firstly run
`python detect_and_link_entity.py --dataset CWQ --split test[train,dev] --linker facc1` to retrieve candidate entities.
Then run `sh scripts/run_entity_disamb.sh CWQ predict test[train,dev]` to rank the candidates by a BertRanker. The ranked results will be saved as `data/CWQ/entity_retrieval/candidate_entities/CWQ_test[train,dev]_cand_entities_facc1.json`.
0623吴轩: 按照上述命令首先生成 unranked;
<!-- 生成 ranked 之后，应该和FACC1_disamb 对比一下，disamb_result 也可以对比
对比了 disamb_result，在排序结果上还是会有一些区别，直接看看 merge 之后的差别大不大吧 -->
感觉不对，应该用自己的 disamb_result, 结合学长的代码(get_candidate_entity_linking_with_logits)，生成 ranked;

3. Finally, merge the linking results of ELQ and FACC1 by running `python data_process.py merge_entity --dataset CWQ --split test[train,dev]`, and the final entity retrieval results are saved as `data/CWQ/entity_retrieval/candidate_entities/CWQ_test[train,dev]_merged_cand_entities_elq_facc1.json`.
0623吴轩: 直接跑这个脚本试试; 基本上和原来只有个位数左右的区别了
在原来的基础上，加一个 update_label?然后比较 id 和 label 的联合一致性, 结果发现也是个位数，大功告成！CWQ实体链接结果复现成功

(4) **Retrieve Candidate Relations**

This step can also be ***skipped*** , as we've provided t he candidate relations in `data/CWQ/relation_retrieval/`

If you want to retrive the candidate relations from scratch, follow the steps below:

1. Train the bi-encoder to encode questions and relations, and build the index of encoded relations. //TODO `complete the running commands`
2. Retrieve relations by dense search over the index. //TODO `complete the running commands`
3. Train the cross-encoder to rank retrieved relations. //TODO `complete the running commands`
4. Merge the logits with relations to get sorted relations for each question by running `python data_process.py merge_relation --dataset CWQ --split test[train,dev]`. The sorted relations will be saved as `data/CWQ/relation_retrieval/candidate_relations/CWQ_test[train,dev]_cand_rels_sorted.json`

(5) **Generate Logical Forms through multi-task learning**
1. Run `python data_process.py --merge_all --dataset CWQ --split test[train,dev]` prepare all the input data for logical form generation and the two auxiliary tasks (entity disambiguation and relation classification). The merged data file will be saved as `data/CWQ/generation/merged/CWQ_test[train,dev].json`
吴轩 0623: merge_all 直接原来的结果，就说明因为有一些候补实体是随机筛选得到的，为了保证与论文中的设置完全一致，直接将论文训练的数据集放上来
2. Run `sh scripts/run_t5_relation_entity_concat_add_prefix_warmup_epochs_5_15epochs_CWQ.sh train multitask0617` to train the logical form generation model. The trained model will be saved in `exps/CWQ_multitask0617`. Command for training other model variants can be found in `cheatsheet_generation.txt`.
3. Command for training model(as shown in 2.) will also do inference on `test` split. You can run `sh scripts/run_t5_relation_entity_concat_add_prefix_warmup_epochs_5_15epochs_CWQ.sh predict multitask0617 False test 50 4` to do inference on `test` split alone. Command for inferencing on other model variants can be found in `cheatsheet_generation.txt`.
4. Run `python3 eval_topk_prediction_final.py --split test --pred_file exps/multitask0617/beam_50_top_k_predictions.json` to evaluate trained model.



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
