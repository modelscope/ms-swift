# 8 * 80GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron sft \
    --model google/gemma-4-26B-A4B-it \
    --save_safetensors true \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
              'AI-ModelScope/LaTeX_OCR:human_handwrite#2000' \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --split_dataset_ratio 0.01 \
    --tuner_type full \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --pipeline_model_parallel_size 2 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --micro_batch_size 8 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 1 \
    --finetune true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --output_dir megatron_output/gemma-4-26B-A4B-it \
    --eval_steps 200 \
    --save_steps 200 \
    --max_length 4096 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend unfused \
    --group_by_length true \
    --padding_free false \
    --model_author swift \
    --model_name swift-robot
