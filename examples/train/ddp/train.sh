# 27.5GiB * 2
nproc_per_node=2

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model /root/.cache/huggingface/hub/models--EleutherAI--pythia-1b-deduped/snapshots/7199d8fc61a6d565cd1f3c62bf11525b563e13b2 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset swift/ToolBench#20000 \
    --model_type qwen2_5 \
    --template default \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --gradient_checkpointing_kwargs "{\"use_reentrant\": false}"
