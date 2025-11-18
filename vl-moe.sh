PYTORCH_NPU_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --load_safetensors true \
    --save_safetensors true \
    --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#5000' \
    --load_from_cache_file true \
    --train_type full \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --split_dataset_ratio 0.01 \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --expert_model_parallel_size 2 \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --finetune true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --max_epochs 1 \
    --save megatron_output/Qwen3-VL-30B-A3B \
    --eval_interval 200 \
    --save_interval 200 \
    --vit_gradient_checkpointing true \
    --max_length 2048 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --padding_free false \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --attention_backend flash
    # --packing true \
    # --moe_permute_fusion true \
    # --cross_entropy_loss_fusion true \