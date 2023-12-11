# Benchmark
测试参数对于训练速度和训练内存使用的影响. 后续会补充部分参数对训练效果的影响.

实验环境:
- CUDA 11.8
- python 3.10
- torch 2.1.1
- flash_attn 2.3.4


我们使用了1000条训练数据集进行基准测试. 实验使用脚本可以查看`scripts/benchmark/test_memory_time/`.

以下为所有实验的相同命令行设置部分:
```bash
    --dataset_test_ratio 0 \
    --dataset cls-fudan-news-zh \
    --train_dataset_sample 1000 \
    --save_strategy no \
    --check_dataset_strategy warning \
    --truncation_strategy truncation_left \
    --preprocess_num_proc 4 \
```

如果未指定以下参数, 则使用以下默认值:
```bash
    --max_length 2048 \
    --batch_size 1 \
    --gradient_checkpinting true \
    --use_flash_attn true \
    --lora_rank 8 \
    --lora_target_modules DEFAULT \
    --quantization_bit 0 \
```

## 量化
测试脚本为:
```bash
swift sft \
    --model_type {MODEL_TYPE} \
    --quantization_bit {QUANTIZATION_BIT} \
    --sft_type lora \
    ...
```

<table>
    <tr>
        <td>Model Type [LoRA]</td>
        <td>Quantization</td>
        <td>Training Speed</td>
        <td>GPU Memory</td>
    </tr>
    <tr>
        <td rowspan="4">qwen-7b-chat</td>
        <td>bf16</td>
        <td>7.01min</td>
        <td>19362MiB</td>
    </tr>
    <tr>
        <td>int4 (gptq)</td>
        <td>11.37min</td>
        <td>10504MiB</td>
    </tr>
    <tr>
        <td>int8 (gptq)</td>
        <td>11.73min</td>
        <td>13648MiB</td>
    </tr>
    <tr>
        <td>int4 (bnb)</td>
        <td>9.38min</td>
        <td>13616MiB</td>
    </tr>
    <tr>
        <td rowspan="4">qwen-14b-chat</td>
        <td>bf16</td>
        <td>11.73</td>
        <td>32186MiB</td>
    </tr>
    <tr>
        <td>int4 (gptq)</td>
        <td>19.69min</td>
        <td>14852MiB</td>
    </tr>
    <tr>
        <td>int8 (gptq)</td>
        <td>20.60min</td>
        <td>20790MiB</td>
    </tr>
    <tr>
        <td>int4 (bnb)</td>
        <td>16.35min</td>
        <td>19278MiB</td>
    </tr>
    <tr>
        <td rowspan="4">qwen-72b-chat</td>
        <td>bf16</td>
        <td>-</td>
        <td>OOM</td>
    </tr>
    <tr>
        <td>int4 (gptq)</td>
        <td>97.94min</td>
        <td>46980MiB</td>
    </tr>
    <tr>
        <td>int8 (gptq)</td>
        <td>103.83min</td>
        <td>80646MiB</td>
    </tr>
    <tr>
        <td>int4 (bnb)</td>
        <td>81.72min</td>
        <td>62430MiB</td>
    </tr>
</table>

## Max Length
### Full
测试脚本为:
```bash
swift sft \
    --model_type {MODEL_TYPE} \
    --max_length {MAX_LENGTH} \
    --sft_type full \
    ...
```

<table>
    <tr>
        <td>Model Type [FULL]</td>
        <td>Max Length</td>
        <td>Training Speed</td>
        <td>GPU Memory</td>
    </tr>
    <tr>
        <td rowspan="5">qwen-1_8b-chat</td>
        <td>512</td>
        <td>1.85min</td>
        <td>18010MiB</td>
    </tr>
    <tr>
        <td>1024</td>
        <td>1.98min</td>
        <td>18072MiB</td>
    </tr>
    <tr>
        <td>2048</td>
        <td>2.76min</td>
        <td>20286MiB</td>
    </tr>
    <tr>
        <td>4096</td>
        <td>3.87min</td>
        <td>26436MiB</td>
    </tr>
    <tr>
        <td>8192</td>
        <td>4.86min</td>
        <td>37530MiB</td>
    </tr>
    <tr>
        <td rowspan="5">qwen-7b-chat</td>
        <td>512</td>
        <td>3.89min</td>
        <td>75213MiB</td>
    </tr>
    <tr>
        <td>1024</td>
        <td>5.74min</td>
        <td>75627MiB</td>
    </tr>
    <tr>
        <td>2048</td>
        <td>8.88min</td>
        <td>76520MiB</td>
    </tr>
    <tr>
        <td>4096</td>
        <td>13.94min</td>
        <td>78986MiB</td>
    </tr>
    <tr>
        <td>8192</td>
        <td>-</td>
        <td>OOM</td>
    </tr>
    <tr>
        <td rowspan="1">qwen-14b-chat</td>
        <td>512</td>
        <td>-</td>
        <td>OOM</td>
    </tr>
</table>

### LoRA
测试脚本为:
```bash
swift sft \
    --model_type {MODEL_TYPE} \
    --max_length {MAX_LENGTH} \
    --sft_type lora \
    ...
```

<table>
    <tr>
        <td>Model Type [LoRA]</td>
        <td>Max Length</td>
        <td>Training Speed</td>
        <td>GPU Memory</td>
    </tr>
    <tr>
        <td rowspan="5">qwen-1_8b-chat</td>
        <td>512</td>
        <td>2.02min</td>
        <td>4610MiB</td>
    </tr>
    <tr>
        <td>1024</td>
        <td>2.07min</td>
        <td>5576MiB</td>
    </tr>
    <tr>
        <td>2048</td>
        <td>2.48min</td>
        <td>7624MiB</td>
    </tr>
    <tr>
        <td>4096</td>
        <td>3.73min</td>
        <td>17324MiB</td>
    </tr>
    <tr>
        <td>8192</td>
        <td>4.48min</td>
        <td>36620MiB</td>
    </tr>
    <tr>
        <td rowspan="5">qwen-7b-chat</td>
        <td>512</td>
        <td>2.52min</td>
        <td>15926MiB</td>
    </tr>
    <tr>
        <td>1024</td>
        <td>4.11min</td>
        <td>17096MiB</td>
    </tr>
    <tr>
        <td>2048</td>
        <td>7.01min</td>
        <td>19362MiB</td>
    </tr>
    <tr>
        <td>4096</td>
        <td>11.12min</td>
        <td>29264MiB</td>
    </tr>
    <tr>
        <td>8192</td>
        <td>13.63min</td>
        <td>48560MiB</td>
    </tr>
    <tr>
        <td rowspan="5">qwen-14b-chat</td>
        <td>512</td>
        <td>3.94min</td>
        <td>28466MiB</td>
    </tr>
    <tr>
        <td>1024</td>
        <td>6.67min</td>
        <td>29708MiB</td>
    </tr>
    <tr>
        <td>2048</td>
        <td>11.73min</td>
        <td>32186MiB</td>
    </tr>
    <tr>
        <td>4096</td>
        <td>18.88min</td>
        <td>42098MiB</td>
    </tr>
    <tr>
        <td>8192</td>
        <td>23.61min</td>
        <td>61412MiB</td>
    </tr>
</table>


## Batch Size
测试脚本为:
```bash
swift sft \
    --batch_size {BATCH_SIZE} \
    --model_type qwen-7b-chat \
    --sft_type lora \
    ...
```

<table>
    <tr>
        <td>Model Type [LoRA]</td>
        <td>Batch Size</td>
        <td>Training Speed</td>
        <td>GPU Memory</td>
    </tr>
    <tr>
        <td rowspan="4">qwen-7b-chat</td>
        <td>1</td>
        <td>7.01min</td>
        <td>19362MiB</td>
    </tr>
    <tr>
        <td>2</td>
        <td>8.05min</td>
        <td>24842MiB</td>
    </tr>
    <tr>
        <td>4</td>
        <td>7.95min</td>
        <td>34842MiB</td>
    </tr>
    <tr>
        <td>8</td>
        <td>7.94min</td>
        <td>54844MiB</td>
    </tr>
</table>

## Use Flash Attn & Gradient Checkpointing
测试脚本为:
```bash
swift sft \
    --use_flash_attn {USE_FLASH_ATTN} \
    --gradient_checkpointing {GRADIENT_CHECKPOINTING} \
    --model_type qwen-7b-chat \
    --sft_type lora \
    ...
```

<table>
     <tr>
        <td>Model Type [LoRA]</td>
        <td>Use Flash Attn</td>
        <td>Gradient Checkpointing</td>
        <td>Training Speed</td>
        <td>GPU Memory</td>
    </tr>
    <tr>
        <td rowspan="4">qwen-7b-chat</td>
        <td>&#x2714;</td>
        <td>&#x2714;</td>
        <td>7.01min</td>
        <td>19362MiB</td>
    </tr>
    <tr>
        <td>&#x2714;</td>
        <td>&#x2718;</td>
        <td>5.19min</td>
        <td>30316MiB</td>
    </tr>
    <tr>
        <td>&#x2718;</td>
        <td>&#x2714;</td>
        <td>9.94min</td>
        <td>19422MiB</td>
    </tr>
    <tr>
        <td>&#x2718;</td>
        <td>&#x2718;</td>
        <td>7.37min</td>
        <td>42260MiB</td>
    </tr>
</table>

## Model Type
测试脚本为:
```bash
swift sft \
    --model_type {MODEL_TYPE} \
    --sft_type lora \
    ...
```

<table>
    <tr>
        <td>Model Type [LoRA]</td>
        <td>Training Speed</td>
        <td>GPU Memory</td>
    </tr>
    <tr>
        <td>qwen-1_8b-chat</td>
        <td>2.48min</td>
        <td>7624MiB</td>
    </tr>
    <tr>
        <td>qwen-7b-chat</td>
        <td>7.01min</td>
        <td>19362MiB</td>
    </tr>
    <tr>
        <td>qwen-14b-chat</td>
        <td>11.73min</td>
        <td>32186MiB</td>
    </tr>
    <tr>
        <td>chatglm2-6b</td>
        <td>7.14min</td>
        <td>14540MiB</td>
    </tr>
    <tr>
        <td>chatglm3-6b</td>
        <td>7.19min</td>
        <td>14612MiB</td>
    </tr>
    <tr>
        <td>yi-6b-chat</td>
        <td>8.18min</td>
        <td>14386MiB</td>
    </tr>
    <tr>
        <td>yi-34b-chat</td>
        <td>30.77min</td>
        <td>70482MiB</td>
    </tr>
    <tr>
        <td>openbuddy-mistral-7b-chat</td>
        <td>9.08min</td>
        <td>16618MiB</td>
    </tr>
    <tr>
        <td>openbuddy-zephyr-7b-chat</td>
        <td>9.10min</td>
        <td>16618MiB</td>
    </tr>
</table>

## LoRA Rank & LoRA Target Modules
测试脚本为:
```bash
swift sft \
    --lora_rank {LORA_RANK} \
    --lora_target_modules {LORA_TARGET_MODULES} \
    --model_type qwen-7b-chat \
    --sft_type lora \
    ...
```

<table>
    <tr>
        <td>Model Type [LoRA]</td>
        <td>LoRA Rank</td>
        <td>LoRA Target Modules</td>
        <td>Training Speed</td>
        <td>GPU Memory</td>
        <td>Trainable Params</td>
    </tr>
    <tr>
        <td rowspan="4">qwen-7b-chat</td>
        <td>2</td>
        <td>DEFAULT (c_attn)</td>
        <td>7.01min</td>
        <td>19300MiB</td>
        <td>1.05M</td>
    </tr>
    <tr>
        <td>8</td>
        <td>DEFAULT</td>
        <td>7.01min</td>
        <td>19362MiB</td>
        <td>4.19M</td>
    </tr>
    <tr>
        <td>64</td>
        <td>DEFAULT</td>
        <td>7.01min</td>
        <td>20728MiB</td>
        <td>33.55MB</td>
    </tr>
    <tr>
        <td>8</td>
        <td>ALL (all linear)</td>
        <td>9.36min</td>
        <td>19670MiB</td>
        <td>17.89M</td>
    </tr>
</table>
