# v5 sequence-classification SFT (task_type=seq_cls) on a single GPU.
#
# `--task_type seq_cls` dispatches swift.dev.cli.sft to run_seq_cls, which builds a num_labels-wide
# classification head on the causal LM. `--problem_type single_label_classification` selects the
# cross-entropy objective (use `regression` with `--num_labels 1` for a scalar MSE head, or
# `multi_label_classification` for BCE). The head is freshly initialized, so this trains FULL
# parameters (`--tuner full`): a LoRA run would need the head listed in modules_to_save to move it.
#
# The dataset is the tiny bundled sample in the documented {messages, label} row format; swap in a real
# classification set (e.g. `--dataset hc3-zh:finance_cls#2000`) for a production run.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type seq_cls \
    --num_labels 2 \
    --problem_type single_label_classification \
    --torch_dtype bfloat16 \
    --dataset examples/v5/train/data/seq_cls.jsonl \
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
    --output_dir output \
    --report_to tensorboard
