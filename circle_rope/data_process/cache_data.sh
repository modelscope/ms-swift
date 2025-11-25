OMP_NUM_THREADS=14 \
ROOT_IMAGE_DIR=/home/dataset/MAmmoTH-VL-Instruct-12M/si-img \
swift export \
    --model /home/ckpt/Qwen2.5-VL-7B-Instruct \
    --dataset
          '/home/dataset/MAmmoTH-VL-Instruct-12M/mammoth_si_10M-cl.parquet' \
          '/home/dataset/MAmmoTH-VL-Instruct-12M/mammoth_ov_2M-cl.parquet' \
    --max_length 4096 \
    --split_dataset_ratio 0.000005 \
    --dataset_num_proc 32 \
    --to_cached_dataset true \
    --lazy_tokenize false \
    --output_dir /home/dataset/MAmmoTH-VL-Instruct-12M/cache