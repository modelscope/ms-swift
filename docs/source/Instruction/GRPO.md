# GRPO

论文地址

[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)

环境安装
```bash
pip install math_verify # reward function
pip install git+https://github.com/huggingface/trl.git # trl >=0.15.0.dev0
```


超参数
- num_generations: 每个prompt采样的数量，论文中的G值，需要被 per_device_eval_batch_size * nproc_per_node 整除
- max_completion_length: 采样生成的最大长度，默认为512
- reward_funcs: 奖励函数，根据模型生成结果进行打分，内置accuracy和format两个rule-based函数，详细见 swift/plugin/orm.py 文件
- use_vllm: 是否使用vLLM作为采样的生成后端，默认为False，建议使用加快训练速度
- vllm_device: 设置vLLM部署的设备，默认为`auto`, 即未被使用的第一张显卡，使用`cuda:x`来设置特定的卡。
- vllm_gpu_memory_utilization: vLLM透传参数
- vllm_max_model_len: vLLM透传参数
- reward_model: 同model, 使用奖励模型作为奖励函数，与reward_funcs至少需要指定一个

建议使用vLLM作为采样后端加速训练，多卡环境下，建议单独设置一张显卡用于部署vLLM，此时进程数应等于显卡数减一

## 运行脚本
多卡vLLM
```bash
# nproc_per_node 比显卡数少一，vLLM默认单独部署于最后一张卡，即卡7
nproc_per_node=7

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --reward_funcs accuracy format \
    --use_vllm true \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 8192 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'AI-MO/NuminaMath-TIR#5000' \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 7 \
    --temperature 0.7 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero2
```

单卡
```bash
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --reward_funcs accuracy format \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset 'AI-MO/NuminaMath-TIR#1000' \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 0.7 \
    --system 'examples/train/grpo/prompt.txt'
```
