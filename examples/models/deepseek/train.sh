# v5 models / deepseek: MoE LoRA SFT of DeepSeek-V4 on the Megatron backend.
#
# model_type = `deepseek_v4` (swift/dev/model/loader/deepseek.py:296; DeepseekV4ForCausalLM, template
# `deepseek_v4`, is_moe). DeepSeek-V4 is Megatron-primary: the real training path lives under the
# megatron backend with a custom MLA RoPE shape contract, so this uses `swift sft --backend megatron`
# (v5 folds the legacy `megatron sft` command into `sft --backend megatron`). The V3.2 / R1 siblings are
# separate model_types (`deepseek_v32`, `deepseek_r1`) resolved the same way from --model.
#
# DeepSeek-V4-Flash is a large sparse MoE. The parallel layout below (expert + pipeline parallel across a
# multi-node cluster) is illustrative; size --expert_model_parallel_size / --pipeline_model_parallel_size
# to the checkpoint and add NNODES/NODE_RANK/MASTER_ADDR as in examples/train/multi-node/torchrun.
#
# REFERENCE-ONLY: not smoke-run here. DeepSeek-V4-Flash is a multi-hundred-GB MoE that needs a multi-node
# Megatron cluster (tens of GPUs) plus mcore-bridge; it cannot be exercised on a single box. model_type is
# verified against swift/dev/model/loader/deepseek.py; the Megatron MoE flag set mirrors the smoke-verified
# examples/train/megatron/moe/train.sh.
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=${NNODES:-1} \
NODE_RANK=${NODE_RANK:-0} \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
USE_SWIFT_V5=1 \
swift sft \
    --backend megatron \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --task_type causal_lm \
    --dataset 'swift/self-cognition#600' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --expert_model_parallel_size 8 \
    --pipeline_model_parallel_size 2 \
    --moe_token_dispatcher_type alltoall \
    --moe_grouped_gemm true \
    --moe_permute_fusion true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --attention_backend flash \
    --lr 1e-4 \
    --min_lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --num_train_epochs 3 \
    --max_length 4096 \
    --logging_steps 5 \
    --save_steps 100 \
    --save_safetensors true \
    --output_dir output \
    --report_to tensorboard
