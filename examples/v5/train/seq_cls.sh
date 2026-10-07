# v5 sequence-classification SFT (task_type=seq_cls) on a single GPU.
#
# `--task_type seq_cls` dispatches swift.dev.cli.sft to run_seq_cls, which builds a num_labels-wide
# classification head on the causal LM. `--problem_type single_label_classification` selects the
# cross-entropy objective (use `regression` with `--num_labels 1` for a scalar MSE head, or
# `multi_label_classification` for BCE). The head is freshly initialized, so this trains FULL
# parameters (`--tuner full`): a LoRA run would need the head listed in modules_to_save to move it.
#
# Dataset: `hc3-zh:finance_cls#2000` is a REAL hub set -- the HC3-Chinese finance subset posed as a 2-way
# Human-vs-ChatGPT classification (the `_cls` view emits an int `label` beside `messages`, matching
# `--num_labels 2`); `#2000` takes a 2000-row slice. Reference it by the REGISTERED name (`hc3-zh:...`), not
# the raw hub id, so the loader's classification preprocessor runs and the `label` column is produced. For an
# OFFLINE run with no hub access, fall back to the bundled `examples/v5/train/data/seq_cls.jsonl` (the same
# {messages, label} shape).
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type seq_cls \
    --num_labels 2 \
    --problem_type single_label_classification \
    --torch_dtype bfloat16 \
    --dataset 'hc3-zh:finance_cls#2000' \
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
