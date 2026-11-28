# v5 models / paddle_ocr: OCR LoRA SFT of PaddleOCR-VL (PaddlePaddle/PaddleOCR-VL-1.6).
#
# model_type = `paddleocr_vl` (swift/dev/model/loader/baidu.py, PaddleOCR1_5Loader; template
# `paddle_ocr_1_5`). swift resolves it from --model, so no explicit --model_type/--template is needed.
# The 1.5/1.6 checkpoints SHIP remote code: the loader sets trust_remote_code=True and loads them via
# AutoModelForCausalLM (transformers>=5.0), because the transformers-native paddleocr_vl class does not
# match these checkpoints. The original PaddleOCR-VL is remote-code pinned to transformers<5.0 and is
# deliberately NOT covered by this loader. This recipe fine-tunes the LLM with LoRA on a handwriting
# LaTeX OCR corpus while --freeze_vit/--freeze_aligner keep the vision tower and projector frozen.
#
# --padding_free true packs the batch without pad tokens (flash_attention_2 varlen path), which is why
# per_device_train_batch_size can sit at 16 on one card for this ~1B model. --attn_impl must be the HF
# token `flash_attention_2`, not legacy swift's `flash_attn`.
#
# REFERENCE-ONLY: model_type verified against the loader registry; the flag set ports the legacy
# recipe (examples/v4/models/paddle_ocr/train.sh) to v5 flag names but was not parse/smoke-verified
# here. The ~1.9GB checkpoint is small enough to smoke on one card when a run is wanted.
# MEASURED: peak VRAM ~3 GiB/card (PaddleOCR-VL-1.6, 4-step smoke, 1 card(s), 2026-11-24)
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model PaddlePaddle/PaddleOCR-VL-1.6 \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#2000' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --num_train_epochs 1 \
    --padding_free true \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 4096 \
    --logging_steps 5 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --split_dataset_ratio 0.01 \
    --load_from_cache_file true \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --output_dir output \
    --report_to tensorboard

# To infer the fine-tuned adapter:
# CUDA_VISIBLE_DEVICES=0 \
# swift infer \
#     --adapters output/vx-xxx/checkpoint-xxx \
#     --sampler transformers \
#     --stream true
