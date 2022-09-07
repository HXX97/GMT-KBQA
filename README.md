GMT-KBQA
==============================

## Abstract
<image src="figures/Overview-new.png">
---
Question answering over knowledge bases (KBQA) for complex questions is a challenging task in natural language processing.
Recently, generation-based methods that translate natural language questions to executable logical forms have achieved promising performance.
However, most of the existing methods struggle in handling questions with unseen KB items and novel combinations of relations. 
Some methods leverage auxiliary information to augment logical form generation, but the noise introduced can also lead to incorrect results.
To address these issues, we propose GMT-KBQA, a Generation-based KBQA method via Multi-Task learning.
GMT-KBQA first gets candidate entities and relations through dense retrieval, and then introduces a multi-task model which jointly learns entity disambiguation, relation classification, and logical form generation.
Experimental results show that GMT-KBQA achieves state-of-the-art results on both ComplexWebQuestions and WebQuestionsSP datasets. 
Furthermore, the detailed evaluation demonstrates that GMT-KBQA benefits from the auxiliary tasks and has a strong generalization capability.

## Repository description
------------

    ├── LICENSE
    ├── README.md                         <- The top-level README for developers using this project.
    ├── .gitignore
    ├── ablation_exps.py                  
    ├── config.py                         <- configuration file
    ├── data_process.py                   <- code for constructing generation task data
    ├── detect_and_link_entity.py         
    ├── error_analysis.py                 
    ├── eval_topk_prediction_final.py     <- code for executing generated logical forms and evaluation
    ├── generation_command.txt            <- command for training, prediction and evaluation of our model
    ├── parse_sparql_cwq.py               <- parse sparql to s-expression
    ├── parse_sparql_webqsp.py            <- parse sparql to s-expression
    ├── run_entity_disamb.py            
    ├── run_multitask_generator_final.py  <- code for training and prediction of our model
    ├── run_relation_data_process.py      <- data preprocess for relation linking
    |
    ├── components                        <- utility functions
    |
    ├── data                         
        ├── common_data                   <- Freebase meta data
            ├── facc1                     <- facc1 data for entity linking   
        ├── CWQ
            ├── entity_retrieval                     
            ├── generation                     
            ├── origin                    
            ├── relation_retrieval                     
            ├── sexpr
        ├── WebQSP
            ├── entity_retrieval                     
            ├── generation                     
            ├── origin                    
            ├── relation_retrieval                     
            ├── sexpr                                       
    ├── entity_retrieval                  <- code for entity detection, linking and disambiguation
    ├── executor                          <- utility functions for executing SPARQLs
    ├── exps                              <- saved model checkpoints and training/prediction/evaluation results
    ├── generation                        <- code for model and dataset evaluation scripts
    ├── inputDataset                      <- code fordataset generation
    ├── lib                               <- virtuoso library
    ├── ontology                          <- Freebase ontology information
    ├── relation_retrieval                <- code for relation retrieval
        ├── bi-encoder
        ├── cross-encoder
    ├── scripts                           <- scripts for entity/relation retrieval and LF generation
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p> 



<!-- ### Logical Form Generation via Multi-task Learning for Complex Question Answering over Knowledge Bases -->

This is an original implementation of paper "Logical Form Generation via Multi-task Learning for Complex Question Answering over Knowledge Bases"
[[Paper PDF](TODO:)]

## Citation
---

```
TODO: cite
```


## Reproducing the Results on CWQ and WebQSP

**Note that all data preparation steps can be skipped, as we've provided those results [here](https://drive.google.com/drive/folders/1QT4EG5wxtcLc8_XT5dTZ2WCi1jtByAyf?usp=sharing). That is, for a fast start, only step (1), (6), (7) is necessary.** 

**At the same time, we also provided detailed instruction for reproducing these results, marked with `(Optional)` below.**

(1) **General setup**

Download the CWQ dataset [here](https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AACuu4v3YNkhirzBOeeaHYala) and put them under `data/CWQ/origin`. The dataset files should be named as `ComplexWebQuestions_test[train,dev].json`.

Download the WebQSP dataset from [here](https://www.microsoft.com/en-us/research/publication/the-value-of-semantic-parse-labeling-for-knowledge-base-question-answering-2/) and put them under `data/WebQSP/origin`. The dataset files should be named as `WebQSP.test[train].json`.

Setup Freebase: Both datasets use Freebase as the knowledge source. You may refer to [Freebase Setup](https://github.com/dki-lab/Freebase-Setup) to set up a Virtuoso triplestore service. After starting your virtuoso service, please replace variable `FREEBASE_SPARQL_WRAPPER_URL` and `FREEBASE_ODBC_PORT` in `config.py` with your own.

**Other data, model checkpoints as well as evaluation results can be downloaded [here](https://drive.google.com/drive/folders/1QT4EG5wxtcLc8_XT5dTZ2WCi1jtByAyf?usp=sharing). Please refer to [README_download.md](https://drive.google.com/file/d/1_QKfo5lr0Ht9Fiu51w5tqwUivnxagaF7/view?usp=sharing) and download what you need. Besides, FACC1 mention information can be downloaded following [data/common_data/facc1/README.md](https://github.com/HXX97/GMT-KBQA/blob/main/data/common_data/facc1/README.md)**

You may create a conda environment according to configuration file `environment.yml`:
```
conda env create -f environment.yml 
```

And then activate this environment:
```
conda activate gmt
```

(2) **(Optional) Parse SPARQL queries to S-expressions** 

This step can be ***skipped***, as we've provided the entity retrieval retuls in 
- CWQ: `data/CWQ/sexpr/CWQ.test[train,dev].jso`.
- WebQSP: `data/WebQSP/sexpr/WebQSP.test[train].json`

As stated in the paper, we generate S-expressions which are not provided by the original dataset.
Here we provide the scripts to parse SPARQL queries to S-expressions. 

- CWQ: Run `python parse_sparql_cwq.py`, and it will augment the original dataset files with s-expressions. 
The augmented dataset files are saved as `data/CWQ/sexpr/CWQ.test[train,dev].json`.

- WebQSP: Run `python parse_sparql_webqsp.py` and the augmented dataset files are saved as `data/WebQSP/sexpr/WebQSP.test[train,dev].json`. 

(3) **(Optional) Retrieve Candidate Entities** 

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

3. Finally, merge the linking results of ELQ and FACC1.  
    - CWQ: `python data_process.py merge_entity --dataset CWQ --split test[train,dev]`, and the final entity retrieval results are saved as `data/CWQ/entity_retrieval/candidate_entities/CWQ_test[train,dev]_merged_cand_entities_elq_facc1.json`. Note for CWQ, entity label will be standardized in final entity retrieval results.
    - WebQSP: `python data_process.py merge_entity --dataset WebQSP --split test[train]`, and the final entity retrieval results are saved as `data/WebQSP/entity_retrieval/candidate_entities/WebQSP_test[train]_merged_cand_entities_elq_facc1.json`.



(4) **(Optional) Retrieve Candidate Relations** 

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


5. Get sorted relations for each question.
    - CWQ: run `python data_process.py merge_relation --dataset CWQ --split test[train,dev]`. The sorted relations will be saved as `data/CWQ/relation_retrieval/candidate_relations/CWQ_test[train,dev]_cand_rels_sorted.json`
    - WebQSP: run `python data_process.py merge_relation --dataset WebQSP --split test[train, train_2hop, test_2hop]`. The sorted relations will be saved as `data/WebQSP/relation_retrieval/candidate_relations/WebQSP_test[train, train_2hop, test_2hop]_cand_rels_sorted.json`

6. (Optional) To only substitude candidate relations in previous merged file, please refer to `substitude_relations_in_merged_file()` in `data_process.py`.

(5) **(Optional) Prepare data for multi-task model**

This step can be **skipped**, as we've provided the results in 
`data/{DATASET}/generation`.

Prepare all the input data for our multi-task LF generation model with entities/relations retrieved above:

- CWQ: Run `python data_process.py merge_all --dataset CWQ --split test[train,dev]` The merged data file will be saved as `data/CWQ/generation/merged/CWQ_test[train,dev].json`.
- WebQSP: Run `python data_process.py merge_all --dataset WebQSP --split test[train]`. The merged data file will be saved as `data/WebQSP/generation/merged/WebQSP_test[train].json`.

(6) **Generate Logical Forms through multi-task learning**

1. Training logical form generation model.
    - CWQ: our full model can be trained by running `sh scripts/GMT_KBQA_CWQ.sh train {FOLDER_NAME}`, The trained model will be saved in `exps/CWQ_{FOLDER_NAME}`.
    - WebQSP: our full model can be trained by running `sh scripts/GMT_KBQA_WebQSP.sh train {FOLDER_NAME}`. The trained model will be saved in `exps/WebQSP_{FOLDER_NAME}`.
    - Command for training other model variants mentioned in our paper can be found in `generation_command.txt`.

2. Command for training model(as shown in 2.) will also do inference on `test` split. To inference on other split or inference alone:
    - CWQ: You can run `sh scripts/GMT_KBQA_CWQ.sh predict {FOLDER_NAME} False test 50 4` to do inference on `test` split with `beam_size=50` and `test_batch_size=4`. 
    - WebQSP: You can run `sh scripts/GMT_KBQA_WebQSP.sh predict {FOLDER_NAME} False test 50 2` to do inference on `test` split alone with `beam_size=50` and `test_batch_size = 2` . 
    - Command for inferencing on other model variants can be found in `generation_command.txt`.

3. To evaluate trained models:
    - CWQ: Run `python3 eval_topk_prediction_final.py --split test --pred_file exps/CWQ_GMT_KBQA/beam_50_test_4_top_k_predictions.json --test_batch_size 4 --dataset CWQ`
    - WebQSP: Run `python3 eval_topk_prediction_final.py --split test --pred_file exps/WebQSP_GMT_KBQA/beam_50_test_2_top_k_predictions.json --test_batch_size 2 --dataset WebQSP`

(7) **Ablation experiments and Error analysis**
1. Evaluate entity linking and relation linking result:
    - CWQ: Run `python ablation_exps.py linking_evaluation --dataset CWQ`
    - WebQSP: Run `python ablation_exps.py linking_evaluation --dataset WebQSP`
2. Evaluate QA performance on questions with unseen entity/relation
    - CWQ: Run `python ablation_exps.py unseen_evaluation --dataset CWQ --model_type full` to get evaluation result on our full model `GMT-KBQA`. Run `python ablation_exps.py unseen_evaluation --dataset CWQ --model_type base` to get evaluation result on `T5-base` model.
    - WebQSP: Run `python ablation_exps.py unseen_evaluation --dataset WebQSP --model_type full` to get evaluation result on our full model `GMT-KBQA`. Run `python ablation_exps.py unseen_evaluation --dataset WebQSP --model_type base` to get evaluation result on `T5-base` model.
3. Error analysis on GMT-KBQA results.
    - Run `python error_analysis.py`.