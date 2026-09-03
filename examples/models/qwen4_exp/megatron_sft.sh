# 8*80G
# pip install git+https://github.com/modelscope/mcore-bridge.git
# pip install git+https://github.com/huggingface/transformers.git

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
megatron sft \
    --model Qwen/Qwen3.8-Flash-Next \
    --dataset 'swift/self-cognition#500' \
    --model_name swift-robot\
    --model_author swift \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules in_proj out_proj linear_proj linear_qkv \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --expert_tensor_parallel_size 1 \
    --pipeline_model_parallel_size 2 \
    --decoder_first_pipeline_num_layers 12 \
    --sequence_parallel true \
    --context_parallel_size 1 \
    --moe_grouped_gemm true \
    --moe_permute_fusion true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 3 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --max_length 2048 \
    --split_dataset_ratio 0.01 \
    --eval_steps 1000 \
    --save_steps 50 \
    --save_safetensors true \
    --merge_lora true \
    --use_precision_aware_optimizer true \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend auto \
    --padding_free false \
    --vit_attn_impl sdpa \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --load_from_cache_file true \
    --logging_steps 1 \
