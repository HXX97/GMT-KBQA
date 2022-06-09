ACTION=${1:-none}
exp_id=${2:-none}
do_debug=${3:-False}

dataset='WebQSP'
exp_prefix="exps/final/${dataset}_${exp_id}/"


if [ "$ACTION" = "train" ]; then
    
    if [ -d ${exp_prefix} ]; then
        echo "${exp_prefix} already exists"
    else
        mkdir ${exp_prefix}
    fi
    # --do_eval \
    python run_multitask_generator_final.py \
                            --do_train \
                            --do_debug ${do_debug} \
                            --do_predict \
                            --max_tgt_len 110 \
                            --max_src_len 256 \
                            --epochs 20 \
                            --lr 5e-5 \
                            --eval_beams 50 \
                            --iters_to_accumulate 1 \
                            --pretrained_model_path /home2/xxhu/QDT2SExpr/CWQ/hfcache/t5-base \
                            --output_dir ${exp_prefix} \
                            --model_save_dir "${exp_prefix}model_saved" \
                            --overwrite_output_dir \
                            --normalize_relations \
                            --sample_size 10 \
                            --add_prefix \
                            --model T5_Multitask_Entity_Concat \
                            --dataset_type WebQSP \
                            --warmup_epochs 5 \
                            --train_batch_size 2 \
                            --eval_batch_size 4 \
                            --test_batch_size 2 | tee "${exp_prefix}log.txt"
elif [ "$ACTION" = "eval" -o "$ACTION" = "predict" ]; then
    split=${4:-test}
    beam_size=${5:-10}
    test_batch_size=${6:-8}
    echo "Predicting ${split} with beam_size: ${beam_size} and batch_size: ${test_batch_size}"
    python run_multitask_generator_final.py \
                        --do_predict \
                        --do_debug ${do_debug} \
                        --predict_split ${split} \
                        --epochs 20 \
                        --lr 5e-5 \
                        --max_tgt_len 110 \
                        --max_src_len 256 \
                        --iters_to_accumulate 1 \
                        --eval_beams ${beam_size} \
                        --pretrained_model_path /home2/xxhu/QDT2SExpr/CWQ/hfcache/t5-base \
                        --output_dir ${exp_prefix} \
                        --model_save_dir "${exp_prefix}model_saved" \
                        --normalize_relations \
                        --sample_size 10 \
                        --add_prefix \
                        --model T5_Multitask_Entity_Concat \
                        --overwrite_output_dir \
                        --dataset_type WebQSP \
                        --train_batch_size 2 \
                        --eval_batch_size 4 \
                        --test_batch_size ${test_batch_size}
else
    echo "train or eval"
fi