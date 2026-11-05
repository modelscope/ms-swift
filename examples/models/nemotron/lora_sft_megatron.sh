# Key constraints for this model:
#   1. target_modules: Do NOT use `all-linear`. The Mamba mixer's in_proj/out_proj
#      cause NaN in backward when wrapped with LoRA. Use only attention + MLP modules.
#      (Megatron module names: linear_qkv, linear_proj, linear_fc1, linear_fc2)
#   2. recompute_granularity: Use `selective` (NOT `full`). Full recompute on HybridModel
#      silently drops LoRA gradients (grad_norm stays 0, loss never descends).

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29900 \
megatron sft \
    --model nv-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
    --save_safetensors true \
    --merge_lora true \
    --dataset 'swift/self-cognition#600' \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --pipeline_model_parallel_size 1 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --recompute_granularity selective \
    --num_train_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --output_dir megatron_output \
    --eval_steps 200 \
    --save_steps 200 \
    --max_length 2048 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --model_author swift \
    --model_name swift-robot
