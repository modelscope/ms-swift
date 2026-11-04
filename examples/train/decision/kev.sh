KEV_SIZE="${KEV_SIZE:-4b}"
case "$KEV_SIZE" in
    0.8b) BASE='Qwen/Qwen3.5-0.8B-Base' ;;
    4b)   BASE='Qwen/Qwen3.5-4B-Base' ;;
    9b)   BASE='Qwen/Qwen3.5-9B-Base' ;;
    *)    echo "KEV_SIZE must be one of 0.8b / 4b / 9b (kev-27b is the full-weight variant, not integrated)"; exit 1 ;;
esac

# Resolve the kev adapter dir with the SAME interpreter that runs `swift` (its entry-point shebang),
# then point KEV_HEAD at the head.pt inside it (the loader's model_dir is the BASE dir).
SWIFT_PY="$(sed -n '1s/^#!//p' "$(command -v swift)")"
KEV_CKPT="$("$SWIFT_PY" -c "from modelscope import snapshot_download; print(snapshot_download('jaredpalmer/kev-$KEV_SIZE'))")"
export KEV_HEAD="$KEV_CKPT/head.pt"

USE_HF=0 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --external_plugins examples/train/decision/dataset.py \
    --model "$BASE" \
    --model_type kev \
    --task_type decision \
    --adapters "$KEV_CKPT" \
    --loss_type scoring \
    --dataset kev_decision \
    --torch_dtype bfloat16 \
    --attn_impl eager \
    --freeze_vit true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 2 \
    --max_length 8192 \
    --split_dataset_ratio 0 \
    --output_dir output \
    --dataloader_num_workers 1
