# 采用Colocate模式进行GPTQ量化模型的GRPO训练

## 1. 问题和可能的解决方法

已知：采用vLLM加速时目前代码会合Lora再更新vllm服务的模型的参数，但是GPTQ量化模型无法合lora。

实际：采用VLLM加速，量化模型在move model to llm时会出错。报错：AttributeError: 'GPTQLoraLinear' object has no attribute 'get_delta_weight'，同https://github.com/modelscope/ms-swift/issues/3949。

现在的框架只能在不采用VLLM推理加速的情况下训练，速度非常慢。（不考虑此方案）

针对这个问题有两种解决方法：

- 方案1:修改ms-swift，在move_model_to_vllm中改为每步暂存Lora参数到本地，调用LLM engine时通过Adapter-request参数传递lora参数

- 方案2:反量化GPTQ-int4模型，在此基础上进行训练，保存lora，最后基模采用量化版本的。

## 2. 方案2

针对方案2，优先测试了ms-swift能否支持非量化的32B模型的Lora模式的GRPO。发现：
- server模式下的VLLM不支持。在更新VLLM服务的模型的参数时会出错，报通信超时错误，同https://github.com/modelscope/ms-swift/issues/4797。
- colocate模式下可以。

目前还没写出无误的GPTQ反量化代码，所以方案2暂时进行到这里。

## 3. 方案1

针对方案1，按想法修改了ms-swift的代码，并且通过了测试，完成了实验。

### 3.1 示例脚本

```bash
MASTER_PORT=29502 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model /xxx/deepseek-r1-distill-qwen-32b-gptq-int4 \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_xxx_accuracy external_xxx_format external_xxx_len \
    --reward_weights 1.0 1.0 1.0 \
    --vllm_mode colocate \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 4 \
    --torch_dtype bfloat16 \
    --dataset 'xxx/xxx.json' \
    --max_completion_length 5120 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --max_length 16000 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --resume_only_model \
    --resume_from_checkpoint /xxx/checkpoint-xxx \
    --output_dir /xxx/xxx \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --num_generations 16 \
    --temperature 0.7 \
    --top_p 1.0 \
    --top_k 80 \
    --log_completions true \
    --report_to tensorboard \
    --model_type deepseek_r1_distill \
    --async_generate false \
    --deepspeed zero3 \
    --sleep_level 1 \
    --max_step 1500 \
    --vllm_max_model_len 30000 \
    --local_adapter_path /xxx/tmp_path_for_lora \
    
```
### 3.2 注意事项

-  需要注意，此时不能用move_model_batches这个参数，也就是不将lora参数分batch传给vllm，否则会报错[rank0]: IndexError: too many indices for tensor of dimension 1。

- 如果是继续训练，比如先前基于sft训练了lora，想在此lora上继续训练，采用GRPO方式。那么如果先前采用的deepspeed阶段是zero3, 那么此时需要采用同样的zero3。不能采用建议的zero3_offload 、offload_optimizer true 、offload_model true 策略，否则会报错[rank0]: KeyError: 'bias_correction'

- 如果遇到爆显存问题，可调低vllm_gpu_memory_utilization，vllm_max_model_len等值。
