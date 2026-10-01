# transformers >= 5.16.1
# pip install -U transformers

# megatron dev
# pip install git+https://github.com/NVIDIA/Megatron-LM.git@dev

# mcore-bridge main
# pip install git+https://github.com/modelscope/mcore-bridge.git
#
# To train the checkpoint's MTP head, change `--mtp_num_layers 0` below to
# `--mtp_num_layers 1` and add `--mtp_loss_scaling_factor 0.1`.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

python -m mcore_bridge.tools.apply_megatron_patch &&

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=${WORLD_SIZE:-1} \
NODE_RANK=${RANK:-0} \
NPROC_PER_NODE=8 \
megatron sft \
    --model ${MODEL:-ZhipuAI/GLM-5.3-Flash} \
    --dataset 'swift/self-cognition#500' \
    --model_name swift-robot \
    --model_author swift \
    --tuner_type lora \
    --target_modules linear_q_down_proj linear_q_up_proj linear_kv_down_proj linear_kv_up_proj linear_proj f_a_proj f_b_proj g_a_proj g_b_proj beta_proj out_proj \
    --finetune true \
    --language_model_only true \
    --vit_attn_impl sdpa \
    --mtp_num_layers 0 \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --pipeline_model_parallel_size 2 \
    --sequence_parallel true \
    --context_parallel_size 1 \
    --moe_grouped_gemm true \
    --moe_permute_fusion true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --num_train_epochs 1 \
    --max_length 4096 \
    --padding_free true \
    --bf16 true \
    --split_dataset_ratio 0.01 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_safetensors true \
    --no_save_optim true \
    --no_save_rng true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --load_from_cache_file true \
    --logging_steps 1 \
    --output_dir megatron_output/GLM-5.3-Flash
