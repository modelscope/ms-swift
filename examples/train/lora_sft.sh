# 22GB
# qwen3: https://github.com/modelscope/ms-swift/blob/main/examples/train/think_model/qwen3_demo1.sh
export DEVICE_TFLOPS=280
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model /metax0402/metax0402/m01119/models/Qwen/Qwen2.5-7B \
    --tuner_type lora \
    --dataset '/metax0402/metax0402/m01119/datasets/AI-ModelScope/alpaca-gpt4-data-zh#500' \
              '/metax0402/metax0402/m01119/datasets/AI-ModelScope/alpaca-gpt4-data-en#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot \
    --model_name swift-robot \
    --enable_profiler true \
    --profiler_save_path ./profiler_output \
    --profiler_ranks 0 1 \
    --profiler_contents "cpu" "cuda" "stack" \
    --profiler_steps 1  \
