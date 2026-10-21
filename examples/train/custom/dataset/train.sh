# v5 custom dataset, declared as DATA (no Python): --custom_dataset_info + a dataset_info JSON.
#
# Most custom datasets differ from a built-in one only in *what* they are -- a local file / a hub id,
# which subsets exist, and which raw column means query vs response -- not in *how* they are read. Those
# need no code: declare them in a JSON file (the same shape as dev's built-in dataset_info.json) and pass
# it to --custom_dataset_info. dev synthesises a DatasetLoader per entry (register_dataset_info), so a
# JSON-declared dataset and a hand-written one are identical downstream.
#
# This example registers `my_qa` from the sibling data.jsonl, whose columns are named prompt/completion;
# the entry's `columns` map renames them to swift's standard query/response, so the auto preprocessor
# reads the rows with no custom Python. `dataset_path` is resolved relative to the JSON file's directory.
# The dataset is then selected by its `dataset_type` key: --dataset my_qa.
#
# For a dataset that needs real code (a bespoke row -> messages transform), write a DatasetLoader
# subclass decorated with @register_dataset and load it with --external_plugins instead; see
# ../model_template for the analogous model/template plugin path.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --custom_dataset_info examples/train/custom/dataset/dataset_info.json \
    --dataset my_qa \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 1024 \
    --logging_steps 2 \
    --save_steps 100 \
    --save_total_limit 2 \
    --split_dataset_ratio 0.1 \
    --output_dir output \
    --report_to tensorboard
