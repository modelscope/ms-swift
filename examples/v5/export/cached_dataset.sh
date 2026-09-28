# v5 export: precompute (tokenize) a dataset once and cache it to disk.
#
# This is the write half of `--cached_dataset`: it runs the dataset load + encode chain ONCE and saves it
# with save_to_disk under <output_dir>/train (+ /val when --split_dataset_ratio > 0). Later training points
# --cached_dataset <output_dir>/train at the SUBDIRECTORY and skips encoding. No model is loaded and no
# distributed init happens -- it is a single-process CPU job (only the tokenizer is needed).
#
# --template_mode selects the encoding objective and IS consumed: 'train' encodes each row to one sample,
# while 'rlhf' encodes chosen+rejected and 'kto' encodes chosen+label, so the same rows produce a
# different cache per objective. Match it to the dataset shape and the training run that will read it:
# an rlhf cache needs rows with `rejected_response`, a kto cache needs a `label`. A cache built for one
# mode is wrong for the others. (build_template defaults the mode to 'train'; this step overrides it.)
USE_SWIFT_V5=1 \
swift export \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset 'swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT' \
    --dataset_num_proc 64 \
    --split_dataset_ratio 0.01 \
    --to_cached_dataset true \
    --template_mode train \
    --output_dir ./qwen2_5_cached_dataset
