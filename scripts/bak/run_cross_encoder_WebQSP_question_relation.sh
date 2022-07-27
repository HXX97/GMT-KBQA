ACTION=${1:-none}
exp_id=${2:-none}

dataset='WebQSP'
exp_prefix="data/WebQSP/relation_retrieval_0717/cross-encoder/saved_models/${exp_id}/"
log_dir="data/WebQSP/relation_retrieval_0717/cross-encoder/saved_models/${exp_id}/"


if [ "$ACTION" = "train" ]; then
    
    if [ -d ${exp_prefix} ]; then
        echo "${exp_prefix} already exists"
    else
        mkdir ${exp_prefix}
    fi
    if [ -d ${log_dir} ]; then
        echo "${log_dir} already exists"
    else
        mkdir ${log_dir}
    fi
    python relation_retrieval/cross-encoder/cross_encoder_main.py \
                            --do_train \
                            --max_len 34 \
                            --batch_size 128 \
                            --epochs 6 \
                            --log_dir ${log_dir} \
                            --dataset_type WebQSP \
                            --model_save_path ${exp_prefix} \
                            --output_dir ${exp_prefix} \
                            --cache_dir hfcache/bert-base-uncased \
                           
elif [ "$ACTION" = "eval" ]; then
    split=${3:-test}
    model_name=${4:-none}
    if [ -d "${exp_prefix}${model_name}_${split}/" ]; then
        echo "${exp_prefix}${model_name}_${split}/ already exists"
    else
        mkdir "${exp_prefix}${model_name}_${split}/"
    fi
    echo "Evaluating ${split}"
    python relation_retrieval/cross-encoder/cross_encoder_main.py \
                            --do_eval \
                            --predict_split ${split} \
                            --max_len 34 \
                            --batch_size 128 \
                            --epochs 6 \
                            --log_dir ${log_dir} \
                            --dataset_type WebQSP \
                            --model_save_path "${exp_prefix}${model_name}" \
                            --output_dir "${exp_prefix}${model_name}_${split}" \
                            --cache_dir hfcache/bert-base-uncased \

elif [ "$ACTION" = "predict" ]; then
    split=${3:-test}
    model_name=${4:-none}
    if [ -d "${exp_prefix}${model_name}_${split}_2hop_repeat/" ]; then
        echo "${exp_prefix}${model_name}_${split}_2hop_repeat/ already exists"
    else
        mkdir "${exp_prefix}${model_name}_${split}_2hop_repeat/"
    fi
    echo "Predicting ${split}"
    python relation_retrieval/cross-encoder/cross_encoder_main.py \
                            --do_predict \
                            --predict_split ${split} \
                            --max_len 34 \
                            --batch_size 128 \
                            --epochs 6 \
                            --log_dir ${log_dir} \
                            --dataset_type WebQSP \
                            --model_save_path "${exp_prefix}${model_name}" \
                            --output_dir "${exp_prefix}${model_name}_${split}_2hop_repeat" \
                            --cache_dir hfcache/bert-base-uncased \

else
    echo "train or eval or predict"
fi