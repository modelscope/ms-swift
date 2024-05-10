# Benchmark
## Table of Contents
- [Parameter Settings](#parameter-settings)
- [Quantization](#quantization)
- [Model Type & Max Length](#model-type--max-length)
- [Batch Size](#batch-size)
- [Use Flash Attn & Gradient Checkpointing](#use-flash-attn--gradient-checkpointing)
- [LoRA Rank & LoRA Target Modules](#lora-rank--lora-target-modules)
- [Gradient Accumulation Steps](#gradient-accumulation-steps)
- [Tuners](#Tuners)
- [Export](#Export)
- [AWQ](#AWQ)
- [AQLM](#AQLM)
- [Sequence Parallel](#Sequence-Parallel)

## Parameter Settings
Experimental environment:
- A100
- CUDA 11.8
- python 3.10
- torch 2.1.1
- flash_attn 2.3.4
- xformers 0.0.23
- auto_gptq 0.5.1
- bitsandbytes 0.41.3.post2


The following are the same command line settings for all experiments:
```bash
    --dataset_test_ratio 0 \
    --dataset cls-fudan-news-zh \
    --save_strategy no \
    --check_dataset_strategy warning \
    --preprocess_num_proc 4 \
```

If the following parameters are not specified, the following default values are used:
```bash
    --max_length 2048 \
    --batch_size 1 \
    --gradient_checkpointing true \
    --use_flash_attn true \
    --lora_rank 8 \
    --lora_target_modules DEFAULT \
    --quantization_bit 0 \
    --gradient_accumulation_steps 16 \
```

Token statistics of the corresponding test dataset (obtained by qwen's tokenizer): 3234.4±2547.5, min=91, max=19548.

The experimental script can be found in `scripts/benchmark/test_memory_time/`.

## Quantization
The test script is:
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
        <td>Training Speed (samples/s)</td>
        <td>GPU Memory (GiB)</td>
    </tr>
    <tr>
        <td rowspan="4">qwen-7b-chat</td>
        <td>bf16</td>
        <td>4.31</td>
        <td>27.74</td>
    </tr>
    <tr>
        <td>int4 (gptq)</td>
        <td>2.05</td>
        <td>19.21</td>
    </tr>
    <tr>
        <td>int8 (gptq)</td>
        <td>1.97</td>
        <td>22.20</td>
    </tr>
    <tr>
        <td>int4 (bnb)</td>
        <td>2.41</td>
        <td>23.85</td>
    </tr>
    <tr>
        <td rowspan="4">qwen-14b-chat</td>
        <td>bf16</td>
        <td>2.60</td>
        <td>40.14</td>
    </tr>
    <tr>
        <td>int4 (gptq)</td>
        <td>1.15</td>
        <td>23.30</td>
    </tr>
    <tr>
        <td>int8 (gptq)</td>
        <td>1.08</td>
        <td>29.13</td>
    </tr>
    <tr>
        <td>int4 (bnb)</td>
        <td>1.36</td>
        <td>30.05</td>
    </tr>
    <tr>
        <td rowspan="4">qwen-72b-chat</td>
        <td>bf16</td>
        <td>0.59 (2*A100)</td>
        <td>73.71+78.54</td>
    </tr>
    <tr>
        <td>int4 (gptq)</td>
        <td>0.23</td>
        <td>54.86</td>
    </tr>
    <tr>
        <td>int8 (gptq)</td>
        <td>0.21</td>
        <td>78.44</td>
    </tr>
    <tr>
        <td>int4 (bnb)</td>
        <td>0.28</td>
        <td>74.87</td>
    </tr>
</table>

## Model Type & Max Length
### LoRA
The test script is:
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
        <td>Training Speed (samples/s)</td>
        <td>GPU Memory (GiB)</td>
    </tr>
    <tr>
        <td rowspan="5">qwen-1_8b-chat</td>
        <td>512</td>
        <td>9.88</td>
        <td>6.99</td>
    </tr>
    <tr>
        <td>1024</td>
        <td>9.90</td>
        <td>10.71</td>
    </tr>
    <tr>
        <td>2048</td>
        <td>8.77</td>
        <td>16.35</td>
    </tr>
    <tr>
        <td>4096</td>
        <td>5.92</td>
        <td>23.80</td>
    </tr>
    <tr>
        <td>8192</td>
        <td>4.19</td>
        <td>37.03</td>
    </tr>
    <tr>
        <td rowspan="5">qwen-7b-chat</td>
        <td>512</td>
        <td>7.43</td>
        <td>18.01</td>
    </tr>
    <tr>
        <td>1024</td>
        <td>6.51</td>
        <td>21.73</td>
    </tr>
    <tr>
        <td>2048</td>
        <td>4.31</td>
        <td>27.74</td>
    </tr>
    <tr>
        <td>4096</td>
        <td>2.05</td>
        <td>35.31</td>
    </tr>
    <tr>
        <td>8192</td>
        <td>1.34</td>
        <td>48.41</td>
    </tr>
    <tr>
        <td rowspan="5">qwen-14b-chat</td>
        <td>512</td>
        <td>5.63</td>
        <td>30.14</td>
    </tr>
    <tr>
        <td>1024</td>
        <td>4.36</td>
        <td>34.43</td>
    </tr>
    <tr>
        <td>2048</td>
        <td>2.60</td>
        <td>40.14</td>
    </tr>
    <tr>
        <td>4096</td>
        <td>1.17</td>
        <td>47.95</td>
    </tr>
    <tr>
        <td>8192</td>
        <td>0.79</td>
        <td>60.74</td>
    </tr>
    <tr>
        <td rowspan="5">qwen-72b-chat (2*A100)</td>
        <td>512</td>
        <td>1.41</td>
        <td>67.68+73.07</td>
    </tr>
    <tr>
        <td>1024</td>
        <td>1.02</td>
        <td>70.25+77.11</td>
    </tr>
    <tr>
        <td>2048</td>
        <td>0.59</td>
        <td>73.71+78.54</td>
    </tr>
    <tr>
        <td>4096</td>
        <td>-</td>
        <td>OOM</td>
    </tr>
    <tr>
        <td>8192</td>
        <td>-</td>
        <td>OOM</td>
    </tr>
    <tr>
        <td rowspan="5">chatglm3-6b</td>
        <td>512</td>
        <td>6.72</td>
        <td>13.94</td>
    </tr>
    <tr>
        <td>1024</td>
        <td>6.16</td>
        <td>12.99</td>
    </tr>
    <tr>
        <td>2048</td>
        <td>4.20</td>
        <td>17.20</td>
    </tr>
    <tr>
        <td>4096</td>
        <td>1.92</td>
        <td>29.80</td>
    </tr>
    <tr>
        <td>8192</td>
        <td>1.24</td>
        <td>66.82</td>
    </tr>
    <tr>
        <td rowspan="5">yi-6b-chat</td>
        <td>512</td>
        <td>5.27</td>
        <td>13.72</td>
    </tr>
    <tr>
        <td>1024</td>
        <td>5.07</td>
        <td>15.44</td>
    </tr>
    <tr>
        <td>2048</td>
        <td>3.84</td>
        <td>16.95</td>
    </tr>
    <tr>
        <td>4096</td>
        <td>1.99</td>
        <td>28.25</td>
    </tr>
    <tr>
        <td>8192</td>
        <td>1.35</td>
        <td>43.81</td>
    </tr>
    <tr>
        <td rowspan="5">yi-34b-chat</td>
        <td>512</td>
        <td>2.32</td>
        <td>66.72</td>
    </tr>
    <tr>
        <td>1024</td>
        <td>1.76</td>
        <td>69.10</td>
    </tr>
    <tr>
        <td>2048</td>
        <td>1.05</td>
        <td>71.34</td>
    </tr>
    <tr>
        <td>4096</td>
        <td>0.47</td>
        <td>78.72</td>
    </tr>
    <tr>
        <td>8192</td>
        <td>0.31 (2*A100)</td>
        <td>47.01+65.03</td>
    </tr>
    <tr>
        <td rowspan="5">openbuddy-zephyr-7b-chat</td>
        <td>512</td>
        <td>5.17</td>
        <td>14.99</td>
    </tr>
    <tr>
        <td>1024</td>
        <td>3.92</td>
        <td>16.57</td>
    </tr>
    <tr>
        <td>2048</td>
        <td>3.08</td>
        <td>19.89</td>
    </tr>
    <tr>
        <td>4096</td>
        <td>1.85</td>
        <td>23.29</td>
    </tr>
    <tr>
        <td>8192</td>
        <td>0.92</td>
        <td>52.14</td>
    </tr>
    <tr>
        <td rowspan="5">baichuan2-7b-chat</td>
        <td>512</td>
        <td>6.09</td>
        <td>18.18</td>
    </tr>
    <tr>
        <td>1024</td>
        <td>5.36</td>
        <td>17.45</td>
    </tr>
    <tr>
        <td>2048</td>
        <td>3.43</td>
        <td>19.18</td>
    </tr>
    <tr>
        <td>4096</td>
        <td>1.69</td>
        <td>34.22</td>
    </tr>
    <tr>
        <td>8192</td>
        <td>1.16</td>
        <td>45.47</td>
    </tr>
    <tr>
        <td rowspan="5">baichuan2-13b-chat</td>
        <td>512</td>
        <td>5.32</td>
        <td>31.01</td>
    </tr>
    <tr>
        <td>1024</td>
        <td>3.91</td>
        <td>31.58</td>
    </tr>
    <tr>
        <td>2048</td>
        <td>1.77</td>
        <td>32.40</td>
    </tr>
    <tr>
        <td>4096</td>
        <td>0.65</td>
        <td>49.63</td>
    </tr>
    <tr>
        <td>8192</td>
        <td>0.36</td>
        <td>76.17</td>
    </tr>
</table>

### Full
The test script is:
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
        <td>Training Speed (samples/s)</td>
        <td>GPU Memory (GiB)</td>
    </tr>
    <tr>
        <td rowspan="5">qwen-1_8b-chat</td>
        <td>512</td>
        <td>10.77</td>
        <td>18.16</td>
    </tr>
    <tr>
        <td>1024</td>
        <td>10.39</td>
        <td>18.62</td>
    </tr>
    <tr>
        <td>2048</td>
        <td>8.73</td>
        <td>35.11</td>
    </tr>
    <tr>
        <td>4096</td>
        <td>5.45</td>
        <td>31.62</td>
    </tr>
    <tr>
        <td>8192</td>
        <td>3.81</td>
        <td>38.93</td>
    </tr>
    <tr>
        <td rowspan="5">qwen-7b-chat</td>
        <td>512</td>
        <td>5.96</td>
        <td>73.37</td>
    </tr>
    <tr>
<td>1024</td>
        <td>5.00</td>
        <td>73.64</td>
    </tr>
    <tr>
        <td>2048</td>
        <td>3.30</td>
        <td>74.26</td>
    </tr>
    <tr>
        <td>4096</td>
        <td>1.64</td>
        <td>78.76</td>
    </tr>
    <tr>
        <td>8192</td>
        <td>1.11 (2*A100)</td>
        <td>61.34+73.00</td>
    </tr>
    <tr>
        <td rowspan="5">qwen-14b-chat (2*A100)</td>
        <td>512</td>
        <td>3.66</td>
        <td>60.42+72.31</td>
    </tr>
    <tr>
        <td>1024</td>
        <td>2.98</td>
        <td>60.61+74.37</td>
    </tr>
    <tr>
        <td>2048</td>
        <td>1.93</td>
        <td>60.70+78.22</td>
    </tr>
    <tr>
        <td>4096</td>
        <td>0.92</td>
        <td>75.59+78.64</td>
    </tr>
    <tr>
        <td>8192</td>
        <td>0.62</td>
        <td>76.59+77.68</td>
    </tr>
</table>


## Batch Size
The test script is:
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
        <td>Training Speed (samples/s)</td>
        <td>GPU Memory (GiB)</td>
    </tr>
    <tr>
        <td rowspan="4">qwen-7b-chat</td>
        <td>1</td>
        <td>4.31</td>
        <td>27.74</td>
    </tr>
    <tr>
        <td>2</td>
        <td>3.60</td>
        <td>43.11</td>
    </tr>
    <tr>
        <td>4</td>
        <td>3.02</td>
        <td>63.81</td>
    </tr>
    <tr>
        <td>8</td>
        <td>2.77</td>
        <td>76.14</td>
    </tr>
</table>

## Use Flash Attn & Gradient Checkpointing
The test script is:
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
        <td>Training Speed (samples/s)</td>
        <td>GPU Memory (GiB)</td>
    </tr>
    <tr>
        <td rowspan="4">qwen-7b-chat</td>
        <td>&#x2714;</td>
        <td>&#x2714;</td>
        <td>4.31</td>
        <td>27.74</td>
    </tr>
    <tr>
        <td>&#x2714;</td>
        <td>&#x2718;</td>
        <td>6.19</td>
        <td>37.70</td>
    </tr>
    <tr>
        <td>&#x2718;</td>
        <td>&#x2714;</td>
        <td>3.13</td>
        <td>27.71</td>
    </tr>
    <tr>
        <td>&#x2718;</td>
        <td>&#x2718;</td>
        <td>4.45</td>
        <td>57.67</td>
    </tr>
</table>


## LoRA Rank & LoRA Target Modules
The test script is:
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
        <td>Training Speed (samples/s)</td>
        <td>GPU Memory (GiB)</td>
        <td>Trainable Params (M)</td>
    </tr>
    <tr>
        <td rowspan="4">qwen-7b-chat</td>
        <td>2</td>
        <td>DEFAULT (c_attn)</td>
        <td>4.27</td>
        <td>27.72</td>
        <td>1.05</td>
    </tr>
    <tr>
        <td>8</td>
        <td>DEFAULT</td>
        <td>4.31</td>
        <td>27.74</td>
        <td>4.19</td>
    </tr>
    <tr>
        <td>64</td>
        <td>DEFAULT</td>
        <td>4.19</td>
        <td>27.85</td>
        <td>33.55</td>
    </tr>
    <tr>
        <td>8</td>
        <td>ALL (all linear)</td>
        <td>3.22</td>
        <td>27.87</td>
        <td>17.89</td>
    </tr>
</table>


## Gradient Accumulation Steps
The test script is:
```bash
swift sft \
    --gradient_accumulation_steps {GRADIENT_ACCUMULATION_STEPS} \
    --model_type qwen-7b-chat \
    --sft_type lora \
    ...
```

<table>
    <tr>
        <td>Model Type [LoRA]</td>
        <td>Gradient Accumulation Steps</td>
        <td>Training Speed (samples/s)</td>
        <td>GPU Memory (GiB)</td>
    </tr>
    <tr>
        <td rowspan="7">qwen-7b-chat</td>
        <td>1</td>
        <td>4.26</td>
        <td>27.73</td>
    </tr>
    <tr>
        <td>2</td>
        <td>4.32</td>
        <td>27.74</td>
    </tr>
    <tr>
        <td>4</td>
        <td>4.31</td>
        <td>27.74</td>
    </tr>
    <tr>
        <td>8</td>
        <td>4.32</td>
        <td>27.74</td>
    </tr>
    <tr>
        <td>16</td>
        <td>4.33</td>
        <td>27.74</td>
    </tr>
    <tr>
        <td>32</td>
        <td>4.30</td>
        <td>27.74</td>
    </tr>
    <tr>
        <td>64</td>
        <td>4.32</td>
        <td>27.74</td>
    </tr>
</table>

## Tuners

| exp_name | model_type | dataset | ms-bench mix ratio | tuner | tuner_params | trainable params(M) | flash_attn | gradient_checkpointing | hypers | memory | train speed(samples/s) | infer speed(tokens/s) | train_loss | eval_loss | gsm8k weighted acc | arc weighted acc | ceval weighted acc |
| -------- | ---------- | ------- | -------------------| ----- | ------------ | ------------------- | -----------| ---------------------- | ------ | ------ | ---------------------- | --------------------- | ---------- | --------- | ------------------ | ---------------- | ------------------ |
|adalora|qwen-7b-chat|ms-agent|2.0|adalora|rank=8/target=ALL/alpha=32/lr_ratio=None/use_rslora=False/use_dora=False|26.8389(0.3464%)|True|True|lr=5e-05/epoch=2|32.55GiB|0.92(87543 samples/95338.71 seconds)|17.33(2345 tokens/135.29 seconds)|0.57|1.07|0.391|0.665|0.569|
|adapter|qwen-7b-chat|ms-agent|2.0|adapter||33.6896(0.4344%)|True|True|lr=5e-05/epoch=2|32.19GiB|1.48(87543 samples/59067.71 seconds)|26.63(4019 tokens/150.90 seconds)|0.55|1.03|0.438|0.662|0.565|
|dora|qwen-7b-chat|ms-agent|2.0|lora|rank=8/target=ALL/alpha=32/lr_ratio=None/use_rslora=False/use_dora=True|19.2512(0.2487%)|True|True|lr=5e-05/epoch=2|32.46GiB|0.51(87543 samples/171110.54 seconds)|4.29(2413 tokens/562.32 seconds)|0.53|1.01|0.466|0.683|**0.577**|
|full+galore128|qwen-7b-chat|ms-agent|2.0|full|galore_rank=128/galore_per_parameter=false/galore_with_embedding=false|7721.3245(100.0000%)|True|True|lr=5e-05/epoch=2|47.02GiB|1.10(87543 samples/79481.96 seconds)|28.96(2400 tokens/82.88 seconds)|0.55|1.00|0.358|**0.688**|**0.577**|
|full+galore32|qwen-7b-chat|ms-agent|2.0|full|galore_rank=32/galore_per_parameter=false/galore_with_embedding=false|7721.3245(100.0000%)|True|True|lr=5e-05/epoch=2|47.05GiB|1.11(87543 samples/78989.74 seconds)|29.17(2431 tokens/83.35 seconds)|0.56|1.01|0.386|0.667|0.539|
|full+galore64|qwen-7b-chat|ms-agent|2.0|full|galore_rank=64/galore_per_parameter=false/galore_with_embedding=false|7721.3245(100.0000%)|True|True|lr=5e-05/epoch=2|46.91GiB|1.11(87543 samples/79200.36 seconds)|28.94(2448 tokens/84.60 seconds)|0.56|1.01|0.397|0.674|0.544|
|full+galore_emb|qwen-7b-chat|ms-agent|2.0|full|galore_rank=128/galore_per_parameter=false/galore_with_embedding=true|7721.3245(100.0000%)|True|True|lr=5e-05/epoch=2|44.53GiB|1.10(87543 samples/79775.02 seconds)|29.45(2433 tokens/82.62 seconds)|0.55|1.00|0.398|0.670|0.568|
|full+galore_perparam|qwen-7b-chat|ms-agent|2.0|full|galore_rank=128/galore_per_parameter=true/galore_with_embedding=false|7721.3245(100.0000%)|True|True|lr=5e-05/epoch=2|47.02GiB|1.25(87543 samples/69821.89 seconds)|29.02(2478 tokens/85.39 seconds)|0.54|1.00|0.372|0.669|0.524|
|full+no_mix|qwen-7b-chat|ms-agent|0.0|full||7721.3245(100.0000%)|True|True|lr=5e-05/epoch=2|72.56GiB|1.27(29698 samples/23356.97 seconds)|30.31(11738 tokens/387.29 seconds)|0.57|**0.44**|0.174|0.652|0.553|
|full|qwen-7b-chat|ms-agent|2.0|full||7721.3245(100.0000%)|True|True|lr=5e-05/epoch=2|73.53GiB|1.43(87543 samples/61022.97 seconds)|29.51(3382 tokens/114.62 seconds)|0.54|0.95|0.343|0.536|0.495|
|llamapro|qwen-7b-chat|ms-agent|2.0|llamapro|num_blocks=4|809.5826(9.4900%)|True|True|lr=5e-05/epoch=2|38.11GiB|1.53(87543 samples/57294.42 seconds)|25.80(2374 tokens/92.02 seconds)|0.53|1.00|0.434|0.645|0.357|
|lora+|qwen-7b-chat|ms-agent|2.0|lora|rank=8/target=ALL/alpha=32/lr_ratio=16.0/use_rslora=False/use_dora=False|17.8913(0.2312%)|True|True|lr=5e-05/epoch=2|32.35GiB|0.95(87543 samples/91923.80 seconds)|18.81(3329 tokens/176.94 seconds)|0.53|0.98|0.432|0.647|0.344|
|lora+neftune|qwen-7b-chat|ms-agent|2.0|lora|rank=8/target=ALL/alpha=32/lr_ratio=None/use_rslora=False/use_dora=False/neftune_noise_alpha=15.0|17.8913(0.2312%)|True|True|lr=5e-05/epoch=2|32.35GiB|0.96(87543 samples/91525.50 seconds)|19.84(161792 tokens/8156.02 seconds)|0.53|1.02|0.456|0.671|0.401|
|lora+no_mix|qwen-7b-chat|ms-agent|0.0|lora|rank=8/target=ALL/alpha=32/lr_ratio=None/use_rslora=False/use_dora=False|17.8913(0.2312%)|True|True|lr=5e-05/epoch=2|30.86GiB|0.91(29698 samples/32570.15 seconds)|19.89(36308 tokens/1825.26 seconds)|0.53|0.53|0.470|0.666|0.574|
|lora|qwen-7b-chat|ms-agent|2.0|lora|rank=8/target=ALL/alpha=32/lr_ratio=None/use_rslora=False/use_dora=False|17.8913(0.2312%)|True|True|lr=5e-05/epoch=2|32.35GiB|0.95(87543 samples/91974.29 seconds)|18.11(2415 tokens/133.32 seconds)|0.53|1.01|0.462|0.676|0.304|
|qwen-7b-chat-eval|qwen-7b-chat|None|0.0|None||None(None)||||None||30.81(13765 tokens/446.83 seconds)|||**0.517**|0.679|0.568|
|rslora|qwen-7b-chat|ms-agent|2.0|lora|rank=8/target=ALL/alpha=32/lr_ratio=None/use_rslora=True/use_dora=False|17.8913(0.2312%)|True|True|lr=5e-05/epoch=2|32.35GiB|0.94(87543 samples/92758.63 seconds)|18.87(2762 tokens/146.34 seconds)|**0.53**|0.99|0.451|0.679|0.339|
| full+lisa_2          | qwen-7b-chat | ms-agent | 2.0                | full     | lisa_activated_layers=2/lisa_step_interval=20                | -                    | True       | True                   | lr=5e-05/epoch=2 | 31.11GiB | 2.66(76837 samples/28881.28 seconds)  | 36.10(134469 tokens/3725.21 seconds) | 0.62       | 1.06      | 0.349              | 0.653            | 0.592              |
| full+lisa_4          | qwen-7b-chat | ms-agent | 2.0                | full     | lisa_activated_layers=4/lisa_step_interval=20                | -                    | True       | True                   | lr=5e-05/epoch=2 | 31.87GiB | 2.63(76837 samples/29215.15 seconds)  | 36.75(135477 tokens/3686.17 seconds) | 0.63       | 1.06      | 0.377              | 0.656            | **0.607**          |
|lora+packing+ddp|qwen-7b-chat|ms-agent|2.0|lora|rank=8/target=ALL/alpha=32/lr_ratio=None/use_rslora=False/use_dora=False/packing=True|17.8913(0.2312%)|True|True|lr=5e-05/epoch=2|35.65GiB*2|1.56(7900 samples/5057.30 seconds)|26.20(421094 tokens/16073.09 seconds)|0.63|0.98|0.473|0.664|0.552|
|lora+packing+lazytokenize|qwen-7b-chat|ms-agent|2.0|lora|rank=8/target=ALL/alpha=32/lr_ratio=None/use_rslora=False/use_dora=False/packing=True|17.8913(0.2312%)|True|True|lr=5e-05/epoch=2|32.83GiB|7.69(78237 samples/10179.40 seconds)|25.86(307390 tokens/11888.17 seconds)|0.63|1.04|0.472|0.660|0.554|
|lora+packing|qwen-7b-chat|ms-agent|2.0|lora|rank=8/target=ALL/alpha=32/lr_ratio=None/use_rslora=False/use_dora=False/packing=True|17.8913(0.2312%)|True|True|lr=5e-05/epoch=2|28.06GiB|0.79(7900 samples/10048.53 seconds)|26.12(409507 tokens/15675.36 seconds)|0.61|0.95|0.492|0.676|0.539|

## unsloth

| exp_name        | model_type         | dataset  | ms-bench mix ratio | tuner | tuner_params | trainable params(M) | flash_attn | gradient_checkpointing | hypers           | memory   | train speed(samples/s)               | infer speed(tokens/s)                 | train_loss | eval_loss | gsm8k weighted acc | arc weighted acc | ceval weighted acc |
| --------------- | ------------------ | -------- | ------------------ | ----- | ------------ | ------------------- | ---------- | ---------------------- | ---------------- | -------- | ------------------------------------ | ------------------------------------- | ---------- | --------- | ------------------ | ---------------- | ------------------ |
| unsloth+lora+q4 | llama3-8b-instruct | ms-agent | 2.0                | lora  |              | 4.7186(0.1038%)     | True       | True                   | lr=5e-05/epoch=2 | 21.69GiB | 1.76(76839 samples/43763.01 seconds) | 15.22(160885 tokens/10570.90 seconds) | 0.58       | 1.03      | 0.668              | 0.755            | 0.501              |

## Export

| exp_name | model_type | calibration dataset | quantization method | quantization bits | infer speed(tokens/s) | gsm8k weighted acc | arc weighted acc | ceval weighted acc |
| -------- | ---------- | ------------------- | ------------------- | ----------------- | --------------------- | ------------------ | ---------------- | ------------------ |
|awq-ms-bench-mini|qwen-7b-chat|ms-bench-mini|awq|4|27.25(16501 tokens/605.47 seconds)|0.494|0.665|0.571|
|awq-pileval|qwen-7b-chat|pileval|awq|4|26.92(12994 tokens/482.72 seconds)|**0.497**|**0.675**|**0.577**|
|gptq-ms-bench-mini|qwen-7b-chat|ms-bench-mini|gptq|4|31.16(15349 tokens/492.54 seconds)|0.482|0.642|0.556|
|gptq-pileval|qwen-7b-chat|pileval|gptq|4|31.67(15185 tokens/479.54 seconds)|0.478|0.654|0.559|

## AWQ

| exp_name | model_type | dataset | ms-bench mix ratio | tuner | tuner_params | trainable params(M) | flash_attn | gradient_checkpointing | hypers | memory | train speed(samples/s) | infer speed(tokens/s) | train_loss | eval_loss | gsm8k weighted acc | arc weighted acc | ceval weighted acc |
| -------- | ---------- | ------- | -------------------| ----- | ------------ | ------------------- | -----------| ---------------------- | ------ | ------ | ---------------------- | --------------------- | ---------- | --------- | ------------------ | ---------------- | ------------------ |
|qwen1half-7b-chat-awq|qwen1half-7b-chat-awq|ms-agent|2.0|lora|rank=8/target=ALL/alpha=32/lr_ratio=None/use_rslora=False/use_dora=False|19.9885(1.5802%)|True|True|lr=5e-05/epoch=2|24.26GiB|0.45(87543 samples/194746.58 seconds)|16.08(2469 tokens/153.58 seconds)|**0.55**|**1.19**|**0.505**|**0.737**|**0.656**|

## AQLM

| exp_name | model_type | dataset | ms-bench mix ratio | tuner | tuner_params | trainable params(M) | flash_attn | gradient_checkpointing | hypers | memory | train speed(samples/s) | infer speed(tokens/s) | train_loss | eval_loss | gsm8k weighted acc | arc weighted acc | ceval weighted acc |
| -------- | ---------- | ------- | -------------------| ----- | ------------ | ------------------- | -----------| ---------------------- | ------ | ------ | ---------------------- | --------------------- | ---------- | --------- | ------------------ | ---------------- | ------------------ |
|llama2-7b-aqlm-2bit-1x16|llama2-7b-aqlm-2bit-1x16|dureader-robust-zh|0.0|lora|rank=8/target=ALL/alpha=32/lr_ratio=None/use_rslora=False/use_dora=False|19.9885(1.6510%)|True|True|lr=5e-05/epoch=2|4.04GiB|0.17(14994 samples/86140.71 seconds)||**0.48**|**0.74**||||

## Sequence Parallel

<table>

<tr>
<td>Model</td>
<td>Dataset</td>
<td>Hyper params</td>
<td>Total steps</td>
<td>Train speed</td>
<td>Gpu memory</td>
</tr>

<tr>
<td rowspan="4">chatglm3-6b-32k</td>
<td rowspan="4">long-alpaca-12k(8055 tokens * 12000 rows)</td>
<td>gpu=2/sequence_parallel_size=1(2 GPU DDP baseline)</td>
<td>5940</td>
<td>0.30iter/s(5h13min total)</td>
<td>27G*2</td>
</tr>


<tr>
<td>gpu=2/sequence_parallel_size=2(2 GPU with sequence parallel 2)</td>
<td>11880</td>
<td>0.5iter/s(6h total)</td>
<td>20G*2</td>
</tr>

<tr>
<td>gpu=4/sequence_parallel_size=4(4 GPU with sequence parallel 4)</td>
<td>11880</td>
<td>1iter/s(3h20min total)</td>
<td>18G*4</td>
</tr>

<tr>
<td>gpu=4/sequence_parallel_size=2(4 GPU sequence parallel 2)</td>
<td>5940</td>
<td>0.45iter/s(3h total)</td>
<td>21G*4</td>
</tr>

</table>
