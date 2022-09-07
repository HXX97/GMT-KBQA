ACTION=${1:-none}
exp_id=${2:-none}
do_debug=${3:-False}

dataset='CWQ'
exp_prefix="exps/${dataset}_${exp_id}/"


if [ "$ACTION" = "train" ]; then
    
    if [ -d ${exp_prefix} ]; then
        echo "${exp_prefix} already exists"
    else
        mkdir -p ${exp_prefix}
    fi
    # --do_eval \
    python run_multitask_generator_final.py \
                            --do_train \
                            --do_predict \
                            --do_debug ${do_debug} \
                            --max_tgt_len 190 \
                            --max_src_len 256 \
                            --epochs 15 \
                            --lr 5e-5 \
                            --eval_beams 50 \
                            --iters_to_accumulate 1 \
                            --pretrained_model_path t5-base \
                            --output_dir ${exp_prefix} \
                            --model_save_dir "${exp_prefix}model_saved" \
                            --overwrite_output_dir \
                            --normalize_relations \
                            --sample_size 10 \
                            --model T5_generation_concat \
                            --dataset_type CWQ \
                            --concat_golden \
                            --train_batch_size 8 \
                            --eval_batch_size 4 \
                            --test_batch_size 4 | tee "${exp_prefix}log.txt"
elif [ "$ACTION" = "eval" -o "$ACTION" = "predict" ]; then
    split=${4:-test}
    beam_size=${5:-50}
    test_batch_size=${6:-4}
    echo "Predicting ${split} with beam_size: ${beam_size} and batch_size: ${test_batch_size}"
    python run_multitask_generator_final.py \
                        --do_predict \
                        --do_debug ${do_debug} \
                        --predict_split ${split} \
                        --epochs 15 \
                        --lr 5e-5 \
                        --max_tgt_len 190 \
                        --max_src_len 256 \
                        --iters_to_accumulate 1 \
                        --eval_beams ${beam_size} \
                        --pretrained_model_path t5-base \
                        --output_dir ${exp_prefix} \
                        --model_save_dir "${exp_prefix}model_saved" \
                        --normalize_relations \
                        --sample_size 10 \
                        --model T5_generation_concat \
                        --overwrite_output_dir \
                        --dataset_type CWQ \
                        --concat_golden \
                        --train_batch_size 8 \
                        --eval_batch_size 4 \
                        --test_batch_size ${test_batch_size}
else
    echo "train or eval"
fi