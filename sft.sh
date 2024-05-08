# Experimental environment: A10, 3090, V100
# 20GB GPU memory

## 环境配置
:<<'EOF'
pip install -e '.[llm]'
cd /cephfs/group/teg-openrecom-openrc/starxhong/feature_eval/model/DeepSeek-VL
pip install -r requirements.txt # 如果遇到库不兼容的问题，按指引安装指定版本即可
pip install -e .
cd -
pip install deepspeed
pip install -U accelerate
EOF

# dataset=/cephfs/group/teg-openrecom-openrc/starxhong/feature_eval/data/train_data/sharegpt/dataset_1231_fully_clean_0329_pmt_v38.jsonl
dataset=/cephfs/group/teg-openrecom-openrc/starxhong/feature_eval/data/train_data/sharegpt/trainset_0506_pmt_v40.jsonl 
# dataset=/cephfs/group/teg-openrecom-openrc/starxhong/feature_eval/data/train_data/sharegpt/dataset_1231_fully_clean_0329_with_image_caption_by_deepseek_with_reason_pmt_v40.jsonl
# dataset=/cephfs/group/teg-openrecom-openrc/starxhong/feature_eval/data/train_data/alpaca/dataset_1231_fully_clean_0329_pmt_v39.jsonl
model_path=/cephfs/group/teg-openrecom-openrc/starxhong/feature_eval/model/deepseek-vl-7b-chat
local_repo_path=/cephfs/group/teg-openrecom-openrc/starxhong/feature_eval/model/DeepSeek-VL
resume_from_checkpoint=/cephfs/group/teg-openrecom-openrc/starxhong/swift/output/deepseek-vl-7b-chat/v39-20240428-234401/checkpoint-3090-merged

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model_type deepseek-vl-7b-chat \
    --custom_train_dataset_path ${dataset} \
    --lora_target_modules ALL \
    --model_id_or_path ${model_path} \
    --lora_target_modules DEFAULT  \
    --local_repo_path ${local_repo_path} \
    --truncation_strategy delete \
    --max_length 1200 \
    --batch_size 1 \
    --learning_rate=1e-4 \
    --num_train_epochs 8 \
    --deepspeed default-zero2 \
    --gpu_memory_fraction 0.95 \
    --dataloader_num_workers 16 \
    --resume_from_checkpoint ${resume_from_checkpoint} \
