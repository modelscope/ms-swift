# v5 continued PRETRAINING (causal_lm on raw text), full-parameter, on a single GPU.
#
# USE_SWIFT_V5=1 routes `swift pt` to the v5 dev CLI (swift.dev.cli.pt). PT is SFT with the non-chat loss
# contract enforced: pt_main pins task_type='causal_lm', use_chat_template=false and loss_scale='all', so
# EVERY token of each document is a training target and no chat markup is added (it rejects any other value).
# The dataset is therefore plain raw text, one document per row.
#
# DATASET. A `text` column is the raw document (format_converter/response.py aliases `text` -> `response`;
# with no chat template the whole string becomes the target). This example bundles a tiny self-contained
# corpus at examples/train/data/pretrain.jsonl so it runs offline and shows a clean descent (the model
# memorises the small corpus). For a REAL pretraining run point --dataset at a large corpus, e.g.
# `swift/chinese-c4`, add `--streaming true` and `--packing true` to keep the pipeline fed, and spread it
# over many GPUs (see examples/train/multi-gpu, or --backend megatron for a sharded optimizer).
#
# `--tuner full` trains every weight (a None TunerConfig). Continued pretraining uses a small learning rate
# so the pretrained distribution is not wrecked; 2e-5 is a reasonable single-node value for a 0.5B model.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift pt \
    --model Qwen/Qwen2.5-0.5B \
    --torch_dtype bfloat16 \
    --dataset examples/train/data/pretrain.jsonl \
    --tuner full \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 1024 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 2 \
    --split_dataset_ratio 0 \
    --dataset_num_proc 4 \
    --output_dir output \
    --report_to tensorboard
