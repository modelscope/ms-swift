# Xing4.0-29B-A4B: trust_remote_code MoE (64 routed experts, top-4) + MLA + mHC.
# bf16 LoRA sft fits on one 80GB+ GPU (~58GB), so this example is single-card.
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

CUDA_VISIBLE_DEVICES=0 \
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
    --model_author swift \
    --model_name swift-robot
