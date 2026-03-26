# ZeRO3: 91.2s/it; 16 * 80GiB
# Megatron-LM: 9.6s/it; 16 * 60GiB
# Launch using Alibaba Cloud DLC
# https://help.aliyun.com/zh/pai/user-guide/general-environment-variables
# ref: https://github.com/modelscope/ms-swift/blob/main/examples/train/multi-node/dlc/train.sh

NNODES=${SENSECORE_PYTORCH_NNODES}
NODE_RANK=${SENSECORE_PYTORCH_NODE_RANK}
NPROC_PER_NODE=8

NUM_GPUS=$((NNODES*NPROC_PER_NODE))
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export OMP_NUM_THREADS=1
export MEGATRON_LM_PATH=/mnt/afs/yangbo1/ms-swift/Megatron-LM
export PYTHONPATH=/mnt/afs/yangbo1/ms-swift/Megatron-LM:$PYTHONPATH
export PYTHONPATH=/mnt/afs/yangbo1/ms-swift:$PYTHONPATH
export AOSS_CONF="/mnt/afs/denghanming/aoss.conf"
#MEGATRON_EXTRA_KWARGS='{ "no_check_for_nan_in_loss_and_grad": true, "overlap-moe-expert-parallel-comm": false, "num_distributed_optimizer_instances": 1, "use_pytorch_profiler": true, "profile": true, "profile_step_end": 12, "profile_step_start": 10, "tensorboard_dir": "/mnt/afs/yangbo1/ms-swift/tensorboard" }'

# MEGATRON_EXTRA_KWARGS='{"check_for_nan_in_loss_and_grad":false,"check_for_large_gradprint("cudnn:", torch.backends.cudnn.is_available(), torch.backends.cudnn.version())s": false,"overlap-moe-expert-parallel-comm":false,"num_distributed_optimizer_instances":1,"use_pytorch_profiler":true,"profile":true,"profile_step_end":14,"profile_step_start":10,"tensorboard_dir":"/mnt/afs/yangbo1/ms-swift/tensorboard"}'

# MEGATRON_EXTRA_KWARGS="no_check_for_nan_in_loss_and_grad=True,overlap-moe-expert-parallel-comm=False,num_distributed_optimizer_instances=1,use_pytorch_profiler=True,profile=True,profile_step_end=12,profile_step_start=10,tensorboard_dir=/mnt/lustre/yangbo1/ms-swift/tensorboard"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=/mnt/afs/yangbo1/.cache/torchinductor_cache/node_rank_${NODE_RANK}
export TRITON_CACHE_DIR=/mnt/afs/yangbo1/.cache/triton_cache/node_rank_${NODE_RANK}
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1


export IMAGE_MAX_TOKEN_NUM=8192 
export VIDEO_MAX_TOKEN_NUM=128 
export FPS_MAX_FRAMES=16 
export MAX_PIXELS=200704 


use_dataset=(
    # '/mnt/afs/yangbo1/datasets/liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT/yb_distill_r1_110k_sft.jsonl#8000'
    '/mnt/afs/fanjianan/workdir/reserved_data/qwen3-vl-30ba3b-thinking-sft-9443/03_thinking_mm_20251225-1.0.jsonl#1000'
)



TP=2
PP=1
EP=8
CP=1
ETP=1
DATE=$(date +%Y%m%d%H%M%S)
LOG_DIR=/mnt/afs/yangbo1/ms-swift/logs/Qwen3.5-35B-A3B/TP${TP}_PP${PP}_CP${CP}_EP${EP}_ETP${ETP}_GPUS${NUM_GPUS}_overlap

mkdir -p $LOG_DIR

    # --decoder_first_pipeline_num_layers 12 \
    # --decoder_last_pipeline_num_layers 12 \
    # --decoder_first_pipeline_num_layers 4 \
    # --decoder_last_pipeline_num_layers 4 \
set -x
    # --cached_dataset '/mnt/afs/yangbo1/datasets/Spec-o3-ColdStartSFT_offline/train' \
    #--model /mnt/afs/yangbo1/models/Qwen/Qwen3.5-35B-A3B \
    #--model /mnt/afs/wanghongli/.cache/modelscope/hub/models/Qwen/Qwen3-VL-30B-A3B-Thinking \
torchrun --nproc_per_node $NPROC_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT swift/cli/_megatron/sft.py \
    --model /mnt/afs/yangbo1/models/Qwen/Qwen3.5-35B-A3B \
    --load_from_cache_file true \
    --cached_dataset '/mnt/afs/yangbo1/ms-swift-backup/swift_dataset_cache/train' \
    --pipeline_model_parallel_size $PP \
    --expert_model_parallel_size $EP \
    --tensor_model_parallel_size $TP \
    --expert_tensor_parallel_size $ETP \
    --context_parallel_size $CP \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 64 \
    --packing true \
    --padding_free false \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --cross_entropy_fusion_impl te \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --output_dir megatron_output/Qwen3.5-35B-A3B \
    --eval_steps 200 \
    --logging_steps 1 \
    --save_steps 10000 \
    --max_length 32768 \
    --dataloader_num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 1 \
    --sequence_parallel true \
    --attention_backend flash \
    --strict false \
    --truncation_strategy delete \
    --vit_gradient_checkpointing false \
    --freeze_llm false \
    --profile true \
    --use_pytorch_profiler true \
    --overlap-grad-reduce \
    --overlap-param-gather \
    2>&1 | tee $LOG_DIR/node_rank_${NODE_RANK}.log

    # --num_layers 4 \
    #--train_iterations 5 \ error
    # --cross_entropy_fusion_impl te \

    # --dataset ${use_dataset[@]} \
    # --split_dataset_ratio 0. \
    # --truncation_strategy delete \
    
    #--cached_dataset '/mnt/afs/yangbo1/ms-swift/swift_dataset_cache/train' \

    # --megatron_extra_kwargs "$MEGATRON_EXTRA_KWARGS"
    # --overlap-grad-reduce \
    # --overlap-param-gather \

    # --megatron_extra_kwargs  '{"no_check_for_nan_in_loss_and_grad": true, "overlap-moe-expert-parallel-comm": true, "num_distributed_optimizer_instances": 1}'
    # --recompute_method uniform \
    # --recompute_num_layers 1 \
    #--overlap-moe-expert-parallel-comm
    # --tp-comm-overlap 

    # --tp-comm-overlap-ag \
    
    # --moe_layer_freq 1 1 1 1
    