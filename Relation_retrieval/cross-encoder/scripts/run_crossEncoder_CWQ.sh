ACTION=${1:-none}
exp_id=${2:-none}

dataset='CWQ'
exp_prefix="../../Data/${dataset}/relation_retrieval/cross-encoder/saved_models/${exp_id}/"
log_dir="../logs/${dataset}/cross-encoder/${exp_id}/"


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
    python cross_encoder_main.py \
                            --do_train \
                            --max_len 70 \
                            --batch_size 128 \
                            --epochs 1 \
                            --log_dir ${log_dir} \
                            --dataset_type CWQ \
                            --mask_entity_mention \
                            --model_save_path ${exp_prefix} \
                            --output_dir ${exp_prefix} \
                           
elif [ "$ACTION" = "eval" ]; then
    split=${3:-test}
    model_name=${4:-none}
    echo "Evaluating ${split}"
    python cross_encoder_main.py \
                            --do_eval \
                            --predict_split ${split} \
                            --max_len 70 \
                            --batch_size 128 \
                            --epochs 1 \
                            --log_dir ${log_dir} \
                            --dataset_type CWQ \
                            --mask_entity_mention \
                            --model_save_path "${exp_prefix}${model_name}" \
                            --output_dir ${exp_prefix} \

elif [ "$ACTION" = "predict" ]; then
    split=${3:-test}
    model_name=${4:-none}
    echo "Predicting ${split}"
    python cross_encoder_main.py \
                            --do_predict \
                            --predict_split ${split} \
                            --max_len 70 \
                            --batch_size 128 \
                            --epochs 1 \
                            --log_dir ${log_dir} \
                            --dataset_type CWQ \
                            --mask_entity_mention \
                            --model_save_path "${exp_prefix}${model_name}" \
                            --output_dir ${exp_prefix} \

else
    echo "train or eval or predict"
fi