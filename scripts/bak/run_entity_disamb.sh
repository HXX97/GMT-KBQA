#!/bin/bash


dataset=${1:-"CWQ"}
ACTION=${2:-none}

exp_prefix="entity_retrieval/entity_disamb_${dataset}"
DATA_DIR="data/${dataset}/entity_retrieval_0724/candidate_entities"

if [ "$ACTION" = "train" ]; then
    
    if [ -d ${exp_prefix} ]; then
    echo "${exp_prefix} already exists"
    else
    mkdir ${exp_prefix}
    fi

    cp scripts/run_entity_disamb.sh "${exp_prefix}run_entity_disamb.sh"
    git rev-parse HEAD > "${exp_prefix}commitid.log"

    # --overwrite_cache \
    python -u run_entity_disamb.py \
        --dataset ${dataset} \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --do_lower_case \
        --do_train \
        --do_eval \
        --disable_tqdm \
        --train_file $DATA_DIR/${dataset}_train_entities_facc1_unranked.json \
        --predict_file $DATA_DIR/${dataset}_dev_entities_facc1_unranked.json \
        --learning_rate 1e-5 \
        --evaluate_during_training \
        --num_train_epochs 2 \
        --overwrite_output_dir \
        --max_seq_length 96 \
        --logging_steps 200 \
        --eval_steps 1000 \
        --save_steps 2000 \
        --warmup_ratio 0.1 \
        --output_dir "${exp_prefix}output" \
        --per_gpu_train_batch_size 8 \
        --per_gpu_eval_batch_size 16 | tee "${exp_prefix}log.txt"

elif [ "$ACTION" = "predict" ]; then
    
    model=${exp_prefix}/output
    split=${3:-test}

    python -u run_entity_disamb.py \
        --dataset ${dataset} \
        --model_type bert \
        --model_name_or_path ${model} \
        --do_lower_case \
        --do_eval \
        --do_predict \
        --predict_file $DATA_DIR/${dataset}_${split}_entities_facc1_unranked.json \
        --overwrite_output_dir \
        --max_seq_length 96 \
        --output_dir  $DATA_DIR/disamb_results/${dataset}_${split} \
        --per_gpu_eval_batch_size 64
else
    echo "train or eval or predict"
fi