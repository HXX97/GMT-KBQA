exp_id=${1:-none}

dataset='CWQ'
exp_prefix="data/${dataset}/relation_retrieval/bi-encoder/saved_models/${exp_id}/"
log_dir="data/${dataset}/relation_retrieval/bi-encoder/saved_models/${exp_id}/"

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
python relation_retrieval/bi-encoder/run_bi_encoder.py \
                            --add_special_tokens \
                            --dataset_type CWQ \
                            --model_save_path ${exp_prefix} \
                            --max_len 32 \
                            --batch_size 4 \
                            --epochs 1 \
                            --log_dir ${log_dir} \
                            --cache_dir bert-base-uncased