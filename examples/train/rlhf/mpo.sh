# https://arxiv.org/abs/2411.10442
# modified from swift/examples/train/multimodal/rlhf/dpo/full.sh
nproc_per_node=4

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=$nproc_per_node \
MAX_PIXELS=1003520 \
swift rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset 'swift/RLAIF-V-Dataset#20000' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --train_type full \
    --torch_dtype bfloat16 \
    --loss_type sigmoid bco_pair sft \
    --loss_weights 0.8 0.2 1.0 \
    --rpo_alpha 0.0 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --freeze_vit true \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --deepspeed zero3 \
    --logging_steps 5 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --save_only_model true
