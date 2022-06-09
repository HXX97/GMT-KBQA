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

## Model Description

TODO:


## Code description

TODO:

## Code

The code for question answering is mainly in  `common`.

The code for candidate relations retrieval is mainly in `relation_detection_and_linking`.


## Reproducing the Results on CWQ
(1) **Prepare dataset and pretrained checkpoints**

Download the CWQ dataset and put them under `data/CWQ/origin`. The dataset files should be named as `ComplexWebQuestions_test[train,dev].json`.

(2) **Parse SPARQL queries to S-expressions**

As stated in the paper, we generate S-expressions which are not provided by the original dataset.
Here we provide the scripts to parse SPARQL queries 

Run `python parse_sparql_cwq.py`, and it will augment the original dataset files with s-expressions. 
The augmented dataset files are saved as `data/CWQ/sexpr/CWQ.test[train,dev].json`.

(3) 

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
