exp_id=${1:-none}

dataset='WEBQSP'
exp_prefix="../../Data/${dataset}/relation_retrieval/bi-encoder/saved_models/${exp_id}/"
log_dir="../logs/${dataset}/bi-encoder/${exp_id}/"

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
python main.py \
                            --dataset_type WEBQSP \
                            --model_save_path ${exp_prefix} \
                            --max_len 80 \
                            --batch_size 3 \
                            --epochs 3 \
                            --log_dir ${log_dir}