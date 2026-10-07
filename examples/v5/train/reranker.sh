# v5 reranker training (task_type=reranker) with a pointwise objective on a single GPU.
#
# `--task_type reranker` dispatches to run_reranker, which builds a num_labels=1 cross-encoder scoring
# head; `--loss pointwise_reranker` trains it with BCE against relevant=1 / irrelevant=0 (use
# `listwise_reranker` for a list-wise softmax objective). Rows carry the same anchor + positive_messages
# / negative_messages layout as embedding. `--task_type generative_reranker` is the alternative that
# scores off the vocab head instead of a dedicated cross-encoder head.
#
# The scoring head is freshly initialized, so this trains FULL parameters (`--tuner full`). Dataset:
# `scidocs-reranking#2000` is a REAL hub set -- the MTEB SciDocs reranking benchmark, whose rows the loader
# turns into a query `messages` plus `positive_messages` (relevant docs) and `negative_messages` (irrelevant
# ones), exactly the pointwise_reranker layout. Reference it by the REGISTERED name (`scidocs-reranking`), NOT
# the raw `MTEB/scidocs-reranking` id, so the reranking preprocessor emits those two columns. OFFLINE fallback
# (no hub): the bundled `examples/v5/train/data/reranker.jsonl`.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type reranker \
    --loss pointwise_reranker \
    --torch_dtype bfloat16 \
    --dataset 'scidocs-reranking#2000' \
    --tuner full \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 512 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 2 \
    --output_dir output \
    --report_to tensorboard
