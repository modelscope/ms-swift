# v5 SFT with generative evaluation (predict_with_generate) on a single GPU.
#
# `--predict_with_generate true` switches the in-training eval from validation-loss to GENERATIVE: at
# every --eval_steps the loop runs an EvalScope benchmark through a resident sampler and folds the report
# into eval_history. `--eval_dataset general_qa` names the benchmark; `--eval_dataset_args` (a JSON
# object) points it at a local {question, answer} jsonl. `--eval_sampler_backend transformers` reuses the
# LIVE training module as the sampler (no second weight copy) -- valid single-card and under Ray.
#
# The `vllm` / `sglang` sampler backends build a co-resident, weight-synced engine and are RAY-ONLY
# (validate._check_eval_generation refuses them under torchrun): add `--mode ray --nproc_per_node 2`
# and `--eval_sampler_backend vllm` to use one. dict-valued Config fields are passed as JSON strings.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --predict_with_generate true \
    --eval_strategy steps \
    --eval_steps 50 \
    --eval_dataset general_qa \
    --eval_dataset_args '{"general_qa": {"local_path": "examples/v5/train/data/eval_qa.jsonl"}}' \
    --eval_limit 8 \
    --eval_generation_config '{"max_tokens": 32, "temperature": 0.0}' \
    --eval_sampler_backend transformers \
    --logging_steps 5 \
    --save_steps 100 \
    --output_dir output \
    --report_to tensorboard
