SWIFT_PY="$(sed -n '1s/^#!//p' "$(command -v swift)")"
OMNIJEV_CKPT="$("$SWIFT_PY" -c "from modelscope import snapshot_download; print(snapshot_download('AI-ModelScope/OmniJev'))")"
export OMNIJEV_HEAD="$OMNIJEV_CKPT/head.pt"
export OMNIJEV_ORD="$OMNIJEV_CKPT/ord.pt"
export OMNIJEV_HEAD_META="$OMNIJEV_CKPT/head_meta.json"
export OMNIJEV_NEW_TOK_EMB="$OMNIJEV_CKPT/new_tok_emb.pt"
export MSO_BRANCH_CKPT=1

USE_HF=0 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --external_plugins examples/train/decision/dataset.py \
    --model Qwen/Qwen3.5-4B \
    --model_type omnijev \
    --task_type decision \
    --adapters "$OMNIJEV_CKPT" \
    --loss_type ce_rps \
    --dataset omnijev_decision \
    --torch_dtype bfloat16 \
    --attn_impl eager \
    --freeze_vit true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing false \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 2 \
    --max_length 4096 \
    --split_dataset_ratio 0 \
    --output_dir output \
    --dataloader_num_workers 1
