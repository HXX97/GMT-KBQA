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
TODO: 发现一个问题，旧的 merged 文件里头吧 sexpr == 'null' 的问题都过滤掉了, 后续看看要不要把过滤的问题补上

(1) **Prepare dataset and pretrained checkpoints**

Download the CWQ dataset from [here](https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AACuu4v3YNkhirzBOeeaHYala) and put them under `data/CWQ/origin`. The dataset files should be named as `ComplexWebQuestions_test[train,dev].json`.

Download the WebQSP dataset from [here](https://www.microsoft.com/en-us/research/publication/the-value-of-semantic-parse-labeling-for-knowledge-base-question-answering-2/) and put them under `data/WebQSP/origin`. The dataset files should be named as `WebQSP.test[train].json`.

(2) **Parse SPARQL queries to S-expressions** 

As stated in the paper, we generate S-expressions which are not provided by the original dataset.
Here we provide the scripts to parse SPARQL queries to S-expressions. 

Run `python parse_sparql_cwq.py`, and it will augment the original dataset files with s-expressions. 
The augmented dataset files are saved as `data/CWQ/sexpr/CWQ.test[train,dev].json`.

Run `python parse_sparql_webqsp.py` and the augmented dataset files are saved as `data/WebQSP/sexpr/WebQSP.test[train,dev].json`. 

(3) **Retrieve Candidate Entities** (测试完毕，没有问题)

This step can be ***skipped***, as we've provided the entity retrieval retuls in 
- CWQ: `data/CWQ/entity_retrieval/candidate_entities/CWQ_test[train,dev]_merged_cand_entities_elq_facc1.json`.
- WebQSP: `data/WebQSP/entity_retrieval/candidate_entities/WebQSP_test[train]_merged_cand_entities_elq_facc1.json`

If you want to retrieve the candidate entities from scratch, follow the steps below:

1. Obtain the linking results from ELQ. Firstly you should deploy our tailored [ELQ](). Then run 
    - CWQ: `python detect_and_link_entity.py --dataset CWQ --split test[train,dev] --linker elq` to get candidate entities linked by ELQ. The results will be saved as `data/CWQ/entity_retrieval/candidate_entities/CWQ_test[train,dev]_cand_entities_elq.json`.
    - WebQSP: Run `python detect_and_link_entity.py --dataset WebQSP --split test[train] --linker elq`, and the results will be saved as `data/WebQSP/entity_retrieval/candidate_entities/WebQSP_test[train]_cand_entities_elq.json`

2. Retrieve candidate entities from FACC1. 
    - CWQ: Firstly run
    `python detect_and_link_entity.py --dataset CWQ --split test[train,dev] --linker facc1` to retrieve candidate entities.
    Then run `sh scripts/run_entity_disamb.sh CWQ predict test[train,dev]` to rank the candidates by a BertRanker. The ranked results will be saved as `data/CWQ/entity_retrieval/candidate_entities/CWQ_test[train,dev]_cand_entities_facc1.json`. 

    - WebQSP: Firstly run
    `python detect_and_link_entity.py --dataset WebQSP --split test[train] --linker facc1` to retrieve candidate entities. Then run `sh scripts/run_entity_disamb.sh WebQSP predict test[train]` to rank the candidates by a BertRanker. The ranked results will be saved as `data/WebQSP/entity_retrieval/candidate_entities/WebQSP_test[train]_cand_entities_facc1.json`

    To reproduce our disambiguation results, please download  `feature_cache/` and place it under root folder from our checkpoint.

3. Finally, merge the linking results of ELQ and FACC1.  
    - CWQ: `python data_process.py merge_entity --dataset CWQ --split test[train,dev]`, and the final entity retrieval results are saved as `data/CWQ/entity_retrieval/candidate_entities/CWQ_test[train,dev]_merged_cand_entities_elq_facc1.json`. Note for CWQ, entity label will be standardized in final entity retrieval results.
    - WebQSP: `python data_process.py merge_entity --dataset WebQSP --split test[train]`, and the final entity retrieval results are saved as `data/WebQSP/entity_retrieval/candidate_entities/WebQSP_test[train]_merged_cand_entities_elq_facc1.json`.



(4) **Retrieve Candidate Relations**

This step can also be ***skipped*** , as we've provided t he candidate relations in `data/CWQ/relation_retrieval/`

If you want to retrive the candidate relations from scratch, follow the steps below:

1. Train the bi-encoder to encode questions and relations.
    - CWQ: Run `python run_relation_retriever.py sample_data --dataset CWQ --split train[dev]` to prepare training data. Then run `sh scripts/run_bi_encoder_CWQ.sh {YOUR_FOLDER_NAME}` to train bi-encoder model. Trained model will be saved in `data/CWQ/relation_retrieval/bi-encoder/saved_models/{YOUR_FOLDER_NAME}`.
    - WebQSP: Run `python run_relation_retriever.py sample_data --dataset WebQSP --split train` to prepare training data. Then run `sh scripts/run_bi_encoder_WebQSP.sh {YOUR_FOLDER_NAME}` to train bi-encoder model. Trained model will be saved in `data/WebQSP/relation_retrieval/bi-encoder/saved_models/{YOUR_FOLDER_NAME}`. Besides, run `python run_relation_retriever.py prepare_2hop_relations --dataset WebQSP` to prepare linking entities' two-hop relations

2. Build the index of encoded relations.
    - CWQ: To encode Freebase relations using trained bi-encoder, run `python relation_retrieval/bi-encoder/build_and_search_index.py encode_relation --dataset CWQ`. Then run `python relation_retrieval/bi-encoder/build_and_search_index.py build_index --dataset CWQ` to build the index of encoded relations. Index file will be saved as `data/CWQ/relation_retrieval/bi-encoder/index/flat.index`.
    - WebQSP: To encode Freebase relations using trained bi-encoder, run `python relation_retrieval/bi-encoder/build_and_search_index.py encode_relation --dataset WebQSP`. Then run `python relation_retrieval/bi-encoder/build_and_search_index.py build_index --dataset WebQSP` to build the index of encoded relations. Index file will be saved as `data/WebQSP/relation_retrieval/bi-encoder/index/ep_{EPOCH}_flat.index`.

3. Retrieve candidate relations using index.
    - CWQ: First encode questions into vector by running `python relation_retrieval/bi-encoder/build_and_search_index.py encode_question --dataset CWQ --split test[train, dev]`. Then candidate relations can be retrieved using index by running `python relation_retrieval/bi-encoder/build_and_search_index.py retrieve_relations --dataset CWQ --split test[train, dev]`. The retrieved relations will be saved as the training data of cross-encoder in `data/CWQ/relation_retrieval/cross-encoder/CWQ_test[train,dev].tsv`.
    - WebQSP: First encode questions into vector by running `python relation_retrieval/bi-encoder/build_and_search_index.py encode_question --dataset WebQSP --split test[train]`. Then candidate relations can be retrieved using index by running `python relation_retrieval/bi-encoder/build_and_search_index.py retrieve_relations --dataset WebQSP --split test[train, ptrain, pdev](注意ptrain pdev 和 train test 调用不同的函数)`. The retrieved relations will be saved as the training data of cross-encoder in `data/WebQSP/relation_retrieval/cross-encoder/WebQSP_test[train,dev].tsv`.
    - TODO: `CWQ.2hopRelations.candEntities.json` 这个文件是如何获得的

4. Train the cross-encoder to rank retrieved relations.
    - CWQ: To train, run `sh scripts/run_cross_encoder_CWQ.sh train {FOLDER_NAME}`. Trained models will be saved as `data/CWQ/relation_retrieval/cross-encoder/saved_models/{FOLDER_NAME}/{MODEL_NAME}`. To get inference results, run `sh scripts/run_cross_encoder_CWQ.sh predict {FOLDER_NAME} test[train/dev] {MODEL_NAME}`.  Inference result(logits) will be stored in `data/CWQ/relation_retrieval/cross-encoder/saved_models/{FOLDER_NAME}/{MODEL_NAME}_test[train/dev]]`.
    - (TODO: 还需要确认在原模型上进行推理得到的结果是完全一致的)WebQSP: To train, run `sh scripts/run_cross_encoder_WebQSP.sh train {FOLDER_NAME}`. Trained models will be saved as `data /WebQSP/relation_retrieval/cross-encoder/saved_models/{FOLDER_NAME}/{MODEL_NAME}`. To get inference results, run `sh scripts/run_cross_encoder_WebQSP.sh predict {FOLDER_NAME} test/[train] {MODEL_NAME}`.Inference result(logits) will be stored in `data/WebQSP/relation_retrieval/cross-encoder/saved_models/{FOLDER_NAME}/{MODEL_NAME}_test/[train]`.


5. Merge the logits with relations to get sorted relations for each question.
    - CWQ: run `python data_process.py merge_relation --dataset CWQ --split test[train,dev]`. The sorted relations will be saved as `data/CWQ/relation_retrieval/candidate_relations/CWQ_test[train,dev]_cand_rels_sorted.json`
    - WebQSP: run `python data_process.py merge_relation --dataset WebQSP --split test[train]`. The sorted relations will be saved as `data/WebQSP/relation_retrieval/candidate_relations/WebQSP_test[train]_cand_rels_sorted.json`

(5) **Generate Logical Forms through multi-task learning**

0. 吴轩 0623: merge_all 直接原来的结果，就说明因为有一些候补实体是随机筛选得到的，为了保证与论文中的设置完全一致，直接将论文训练的数据集放上来
1.  Prepare all the input data for logical form generation and the two auxiliary tasks (entity disambiguation and relation classification). 
    - CWQ: Run `python data_process.py merge_all --dataset CWQ --split test[train,dev]` The merged data file will be saved as `data/CWQ/generation/merged/CWQ_test[train,dev].json`.

    - WebQSP: Run `python data_process.py merge_all --dataset WebQSP --split test[train]`. The merged data file will be saved as `data/WebQSP/generation/merged/WebQSP_test[train].json`.

2. Training logical form generation model.
    - CWQ: our full model can be trained by running `sh scripts/run_t5_relation_entity_concat_add_prefix_warmup_epochs_5_15epochs_CWQ.sh train {FOLDER_NAME}`, The trained model will be saved in `exps/CWQ_{FOLDER_NAME}`.
    - WebQSP: our full model can be trained by running `sh scripts/run_t5_entity_concat_add_prefix_warmup5_20epochs_WebQSP.sh train {FOLDER_NAME}`. The trained model will be saved in `exps/WebQSP_{FOLDER_NAME}`.
    - Command for training other model variants mentioned in our paper can be found in `cheatsheet_generation.txt`.

3. Command for training model(as shown in 2.) will also do inference on `test` split. To inference on other split or inference alone:
    - CWQ: You can run `sh scripts/run_t5_relation_entity_concat_add_prefix_warmup_epochs_5_15epochs_CWQ.sh predict {FOLDER_NAME} False test 50 4` to do inference on `test` split alone with `beam_size=50` and `test_batch_size=4`. 
    - WebQSP: You can run `sh scripts/run_t5_entity_concat_add_prefix_warmup5_20epochs_WebQSP.sh predict {FOLDER_NAME} False test 50 2` to do inference on `test` split alone with `beam_size=50` and `test_batch_size = 2` . 
    - Command for inferencing on other model variants can be found in `cheatsheet_generation.txt`.

4. To evaluate trained models:
    - CWQ: Run `python3 eval_topk_prediction_final.py --split test --pred_file exps/{FOLDER_NAME}/beam_50_top_k_predictions.json`
    - WebQSP: Run `python3 eval_topk_prediction_final.py --split test --pred_file exps/{FOLDER_NAME}/beam_50_top_k_predictions.json --dataset WebQSP`



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
