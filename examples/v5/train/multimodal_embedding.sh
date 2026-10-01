# v5 image-text multimodal embedding (task_type=embedding) with the InfoNCE objective on a single GPU.
#
# The multimodal sibling of embedding.sh: `--task_type embedding` dispatches to run_embedding, which
# pools each candidate into a sentence vector, and `--loss infonce` contrasts the anchor against its
# positive/negative. What makes this multimodal is that `<image>` may appear in the anchor `messages`
# AND in each candidate conversation, with its own parallel media column -- `images` for the anchor,
# `positive_images` / `negative_images` for the candidates. The positive/negative media columns are
# list-of-list: the outer list aligns with the candidate conversations (one positive is the documented
# constraint), the inner list with the `<image>` tags in that conversation. See data/vl_embedding.jsonl
# for the row shape and docs/source/BestPractices/Embedding.md for the format.
#
# The template/model_type are resolved from the checkpoint, so no explicit --template is needed.
# MAX_PIXELS caps the per-image token budget the vision tower produces. Embedding has no classification
# head, so LoRA trains cleanly here. For a real run point --dataset at a multimodal retrieval set and
# use a dedicated multimodal embedding checkpoint (e.g. Qwen/Qwen3-VL-Embedding); the 3B VL model below
# is the smallest real Qwen2.5-VL and keeps the example self-contained.
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --task_type embedding \
    --loss infonce \
    --torch_dtype bfloat16 \
    --dataset examples/v5/train/data/vl_embedding.jsonl \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 2 \
    --split_dataset_ratio 0 \
    --dataset_num_proc 4 \
    --output_dir output \
    --report_to tensorboard
