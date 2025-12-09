# mcore>=0.15
# 8 * 80GiB 4s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
OMP_NUM_THREADS=14 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron sft \
    --model OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview \
    --load_safetensors true \
    --save_safetensors true \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#10000' \
              'AI-ModelScope/LaTeX_OCR:human_handwrite#5000' \
              'swift/VideoChatGPT:Generic#2000' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --moe_permute_fusion true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 8 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --micro_batch_size 2 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save megatron_output/InternVL3_5-GPT-OSS-20B-A4B-Preview \
    --eval_interval 500 \
    --save_interval 500 \
    --max_length 4096 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --moe_expert_capacity_factor 2 \
    --padding_free false \
    --attention_backend unfused

# CUDA_VISIBLE_DEVICES=0 \
# swift infer \
#     --model megatron_output/InternVL3_5-GPT-OSS-20B-A4B-Preview/vx-xxx/checkpoint-xxx \
#     --stream true \
#     --load_data_args true \
#     --max_new_tokens 512
