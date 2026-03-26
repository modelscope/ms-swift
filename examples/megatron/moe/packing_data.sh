
export PYTHONPATH=/mnt/afs/yangbo1/ms-swift:$PYTHONPATH
export AOSS_CONF="/mnt/afs/denghanming/aoss.conf"

use_data_json=(
    '/mnt/afs/fanjianan/workdir/reserved_data/qwen3-vl-30ba3b-thinking-sft-9443/03_thinking_mm_20251225-1.0.jsonl#10000'
)
OMP_NUM_THREADS=1 \
IMAGE_MAX_TOKEN_NUM=8192 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
MAX_PIXELS=200704 \
TORCHCODEC_NUM_THREADS=8 \
python3 swift/cli/export.py \
    --model /mnt/afs/yangbo1/models/Qwen/Qwen3.5-397B-A17B-no-safetensors-real \
    --dataset "${use_data_json[@]}" \
    --split_dataset_ratio 0 \
    --dataset_num_proc 32 \
    --max_length 32768 \
    --to_cached_dataset true \
    --truncation_strategy delete \
    --strict false \
    --output_dir "/mnt/afs/yangbo1/ms-swift/swift_dataset_cache"