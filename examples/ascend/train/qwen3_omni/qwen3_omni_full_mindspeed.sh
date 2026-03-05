# 16 * 64GiB Ascend A3
# Modified from https://github.com/modelscope/ms-swift/blob/main/examples/megatron/multimodal/omni/moe.sh
PYTORCH_NPU_ALLOC_CONF='expandable_segments:True' \
ENABLE_AUDIO_OUTPUT=0 \
NPROC_PER_NODE=16 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
megatron sft \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --save_safetensors true \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#10000' \
              'AI-ModelScope/LaTeX_OCR:human_handwrite#5000' \
              'speech_asr/speech_asr_aishell1_trainsets:validation#5000' \
    --load_from_cache_file true \
    --sequence_parallel true \
    --packing true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --split_dataset_ratio 0.01 \
    --expert_model_parallel_size 2 \
    --expert_tensor_parallel_size 2 \
    --tensor_model_parallel_size 4 \
    --pipeline_model_parallel_size 2 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --num_train_epochs 1 \
    --output_dir megatron_output/Qwen3-Omni-30B-A3B-Instruct \
    --eval_steps 100 \
    --save_steps 100 \
    --max_length 1024 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend flash
    # --moe_permute_fusion true
