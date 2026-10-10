# Typed-decision (System-1) fine-tuning of tinnel123/OmniJev.
#
# OmniJev scores ONE question per forward over a MULTIMODAL state (image, or a video pre-composed
# offline into a mosaic). OmniJevPreprocessor fans a record out into one row per question and carries
# the official option-BLOCK structure plus width-matched `omnijev_column_keys` (noul -> [yes,no];
# choice -> keys + an extra learned `abstain` column; score -> the level labels). Because the hybrid
# Qwen3.5-4B backbone carries recurrent state, options are scored by the two-stage cached BRANCH
# (swift/model/omnijev_branch.py), not a block-diagonal mask.
#
# `--loss_type omnijev` uses PROPER scoring rules only (faithful to mso/head.py: no label smoothing).
#
# CONTINUE-TRAINING a SPLIT layout, not a fresh LoRA: OmniJev ships adapter-only (tinnel123/OmniJev = a
# Qwen/Qwen3.5-4B LoRA + the head artifacts head.pt / ord.pt / new_tok_emb.pt / head_meta.json, with NO
# base weights and NO config.json). So `--model Qwen/Qwen3.5-4B` supplies the hybrid base and
# `--adapters <OmniJev>` continues training THAT LoRA (is_trainable=True); a fresh `--tuner_type lora
# --target_modules all-linear` would discard it. The two live on DIFFERENT hubs: the base Qwen3.5-4B is
# distributed on ModelScope (so `USE_HF=0` -> swift resolves it there), while the tinnel123/OmniJev
# adapter is HF-only and is resolved by the explicit `huggingface_hub.snapshot_download` below (which is
# independent of `USE_HF`). Because the loader's model_dir is the BASE dir, the four
# head artifacts (which live in the adapter dir) are pointed at via OMNIJEV_* envs -- including
# new_tok_emb.pt, the ONLY source of the two option-token embeddings (modules_to_save=None, so they are
# not in the LoRA; a missing file must error, never silently mean-init). The head's fp32 params are
# re-activated for joint training AUTOMATICALLY by ScoringTrainer (--trainable_parameters is a no-op in
# the adapter branch, so it is omitted), and saved beside the adapter on checkpoint.
#
# MEMORY: the cached BRANCH is fundamentally incompatible with HF `--gradient_checkpointing` (that flag
# forces use_cache=False inside the backbone, but the branch's prefix pass MUST return a KV/recurrent
# cache to reuse per option row). `_omnijev_branch_forward` therefore auto-disables gradient_checkpointing
# around the branch, so `--gradient_checkpointing` is a NO-OP for OmniJev. The branch's own chunk-level
# activation recomputation (MSO_BRANCH_CKPT=1, swift/model/omnijev_branch.py) provides the memory savings
# instead -- set it for real runs; leave --gradient_checkpointing false to avoid the false impression.
#
# sh examples/train/decision/omnijev.sh
# Resolve the OmniJev adapter dir with the SAME interpreter that runs `swift` (its entry-point shebang),
# then point the OMNIJEV_* head-artifact envs at it (they live in the adapter dir, not the base dir).
SWIFT_PY="$(sed -n '1s/^#!//p' "$(command -v swift)")"
OMNIJEV_CKPT="$("$SWIFT_PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('tinnel123/OmniJev'))")"
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
    --loss_type omnijev \
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
