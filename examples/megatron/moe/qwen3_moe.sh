# ZeRO3: 91.2s/it; 16 * 80GiB
# Megatron-LM: 9.6s/it; 16 * 60GiB
# Launch using Alibaba Cloud DLC
# https://help.aliyun.com/zh/pai/user-guide/general-environment-variables
# ref: https://github.com/modelscope/ms-swift/blob/main/examples/train/multi-node/dlc/train.sh
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export MEGATRON_LM_PATH=/mnt/afs/yangbo1/ms-swift/Megatron-LM
export PYTHONPATH=/mnt/afs/yangbo1/ms-swift/Megatron-LM:$PYTHONPATH
export PYTHONPATH=/mnt/afs/yangbo1/ms-swift:$PYTHONPATH
#MEGATRON_EXTRA_KWARGS='{ "no_check_for_nan_in_loss_and_grad": true, "overlap-moe-expert-parallel-comm": false, "num_distributed_optimizer_instances": 1, "use_pytorch_profiler": true, "profile": true, "profile_step_end": 12, "profile_step_start": 10, "tensorboard_dir": "/mnt/afs/yangbo1/ms-swift/tensorboard" }'

MEGATRON_EXTRA_KWARGS='{"no_check_for_nan_in_loss_and_grad":true,"overlap-moe-expert-parallel-comm":false,"num_distributed_optimizer_instances":1,"use_pytorch_profiler":true,"profile":true,"profile_step_end":14,"profile_step_start":10,"tensorboard_dir":"/mnt/lustre/yangbo1/ms-swift/tensorboard"}'

# MEGATRON_EXTRA_KWARGS="no_check_for_nan_in_loss_and_grad=True,overlap-moe-expert-parallel-comm=False,num_distributed_optimizer_instances=1,use_pytorch_profiler=True,profile=True,profile_step_end=12,profile_step_start=10,tensorboard_dir=/mnt/lustre/yangbo1/ms-swift/tensorboard"

use_dataset=(
    '/mnt/afs/yangbo1/datasets/liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT/yb_distill_r1_110k_sft.jsonl#8000'
    # '/mnt/afs/yangbo1/datasets/liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT/yb_distill_r1_110k_sft.jsonl#4000'
)


set -x
torchrun --nproc_per_node 1 --nnodes 1 swift/cli/_megatron/sft.py \
    --model /mnt/afs/yangbo1/models/Qwen3-VL-30B-A3B-Instruct \
    --dataset ${use_dataset[@]} \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --pipeline_model_parallel_size 1 \
    --expert_model_parallel_size 1 \
    --tensor_model_parallel_size 1 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap false \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 1 \
    --packing true \
    --padding_free false \
    --recompute_granularity selective \
    --num_train_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --output_dir megatron_output/Qwen3-VL-30B-A3B-Base \
    --eval_interval 200 \
    --log_interval 1 \
    --save_interval 200 \
    --max_length 4096 \
    --dataloader_num_workers 0 \
    --dataset_num_proc 1 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --num_layers 4 \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --vit_gradient_checkpointing false \
    # --megatron_extra_kwargs "$MEGATRON_EXTRA_KWARGS"

    # --megatron_extra_kwargs  '{"no_check_for_nan_in_loss_and_grad": true, "overlap-moe-expert-parallel-comm": true, "num_distributed_optimizer_instances": 1}'
    # --recompute_method uniform \
    # --recompute_num_layers 1 \
    #--overlap-moe-expert-parallel-comm
    # --tp-comm-overlap 

    # --tp-comm-overlap-ag \
    
    # --moe_layer_freq 1 1 1 1
    


