# transformers: https://github.com/modelscope/ms-swift/blob/main/examples/train/moe/llama4.sh
# 4 * 80GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
megatron sft \
    --model LLM-Research/Llama-4-Scout-17B-16E-Instruct \
    --load_safetensors true \
    --save_safetensors true \
    --merge_lora true \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#10000' \
              'AI-ModelScope/LaTeX_OCR:human_handwrite#5000' \
    --load_from_cache_file true \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --sequence_parallel true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --split_dataset_ratio 0.01 \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 4 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --max_epochs 1 \
    --save megatron_output/Llama-4-Scout-17B-16E-Instruct \
    --eval_interval 100 \
    --save_interval 100 \
    --vit_gradient_checkpointing false \
    --max_length 4096 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --padding_free false \
    --attention_backend unfused


CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --model megatron_output/Llama-4-Scout-17B-16E-Instruct/vx-xxx/checkpoint-xxx \
    --stream true \
    --load_data_args true \
    --max_new_tokens 2048
