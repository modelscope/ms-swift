# Xing4.0-29B-A4B: trust_remote_code MoE (64 routed experts, top-4) + MLA + mHC.
# 8-GPU DeepSpeed ZeRO-3 version: weights are sharded across the 8 ranks, so each GPU peaks at
# ~13GiB (measured), vs ~60GiB for plain DDP. Verified 8-GPU: loss 2.45->2.31, grad_norm
# 5.8->4.2, 0.7208% trainable, z3_leaf_modules auto-set to Xing4_0MoE.
# For max throughput on 80GB+ cards, drop --deepspeed zero3 to run plain DDP (~60GiB/GPU, ~3x
# faster per step); DDP was also verified and reports no unused parameters with this config.
# Expert parallel (EP) is NOT available here: swift's HF training path has no EP flag, and EP would
# need transformers-native patches (base_model_ep_plan + EP-aware experts forward + a swift
# distributed_config passthrough) or the Megatron path. Since 29B/64 experts fit on one GPU and
# ZeRO-3 already shards to ~13GiB, EP's communication savings don't pay off at this scale.
#
# Key constraints:
#   1. --experts_impl grouped_mm: stacks the 64 per-expert nn.Linear into 3D tensors and uses
#      transformers' grouped GEMM (~3.7x faster end-to-end than the official python loop).
#      Stacking follows this flag; without it the model keeps the official per-expert structure.
#   2. --target_parameters: once stacked, the routed experts are 3D nn.Parameter, which
#      `all-linear` cannot match (it only sees nn.Linear). List them here so the experts get LoRA
#      too; otherwise only attention + shared experts are tuned. Needs peft>=0.17.
#   3. --lora_dropout 0: peft's target_parameters path rejects a non-zero dropout, and swift's
#      default is 0.05, so set this to 0 explicitly or the run raises.
#   4. mHC compile is OFF by default for grad parity. For ~34% more speed at a ~6% grad_norm
#      deviation, set SWIFT_XING4_0_COMPILE_MHC=1.

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model XingChen-AGI/Xing4.0-29B-A4B \
    --tuner_type lora \
    --target_modules all-linear \
    --target_parameters mlp.experts.gate_up_proj mlp.experts.down_proj \
    --lora_dropout 0 \
    --experts_impl grouped_mm \
    --dataset 'swift/self-cognition#1000' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 4 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero3 \
    --model_author swift \
    --model_name swift-robot
