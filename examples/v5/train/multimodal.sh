# v5 image-text multimodal SFT (Qwen2.5-VL) with LoRA on a single GPU.
#
# The template/model_type are resolved from the checkpoint, so no explicit --template is needed. Dataset:
# `OK-VQA_train#2000` is a REAL hub VL corpus (the OK-VQA train split, registered in
# swift/dev/dataset/loader/mllm.py): each row is a `messages` conversation whose user turn asks a question
# about an image, plus a parallel `images` entry the loader resolves to the actual picture -- the documented
# dev multimodal row shape, so no plugin is needed. `#2000` takes a 2000-row slice (the full split is ~800MB,
# cached after the first run). MAX_PIXELS caps the per-image token budget the vision tower produces. LoRA on
# the LLM keeps the run light; for the legacy "LoRA LLM + full ViT" recipe see examples/train. The 3B VL model
# is the smallest real Qwen2.5-VL; swap in the 7B for a production run. OFFLINE fallback (no hub): the bundled
# `examples/v5/train/data/vl.jsonl` -- a `messages` turn with an `<image>` placeholder + a parallel `images`
# list of the canonical ms-swift sample image URLs.
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --dataset 'OK-VQA_train#2000' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 4096 \
    --logging_steps 5 \
    --save_steps 100 \
    --save_total_limit 2 \
    --split_dataset_ratio 0 \
    --dataset_num_proc 4 \
    --output_dir output \
    --report_to tensorboard
