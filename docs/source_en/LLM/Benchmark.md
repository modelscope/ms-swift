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

| exp_name | model_type | dataset | ms-bench mix ratio | tuner | tuner_params | flash_attn | gradient_checkpointing | hypers | memory | train speed(samples/s) | infer speed(tokens/s) | train_loss | eval_loss | gsm8k weighted acc | arc weighted acc | ceval weighted acc |
| -------- | ---------- | ------- | -------------------| ----- | ------------ | -----------| ---------------------- | ------ | ------ | ---------------------- | --------------------- | ---------- | --------- | ------------------ | ---------------- | ------------------ |
|adalora|qwen-7b-chat|ms-agent|2.0|adalora|rank=8/target=ALL/alpha=32/lr_ratio=None/use_rslora=False/use_dora=False|True|True|lr=5e-05/epoch=2|32.55GiB|0.92(87543 samples/95338.71 seconds)|9.49(162670 tokens/17148.76 seconds)|0.57|1.07|0.391|0.716|0.569|
|adapter|qwen-7b-chat|ms-agent|2.0|adapter||True|True|lr=5e-05/epoch=2|32.19GiB|1.48(87543 samples/59067.71 seconds)|13.16(266828 tokens/20279.10 seconds)|0.55|1.03|0.438|0.713|0.565|
|dora|qwen-7b-chat|ms-agent|2.0|lora|rank=8/target=ALL/alpha=32/lr_ratio=None/use_rslora=False/use_dora=True|True|True|lr=5e-05/epoch=2|32.46GiB|0.51(87543 samples/171110.54 seconds)|3.52(219914 tokens/62510.25 seconds)|0.53|1.01|0.466|0.718|0.577|
|full+galore128|qwen-7b-chat|ms-agent|2.0|full|galore_rank=128/galore_per_parameter=false/galore_with_embedding=false|True|True|lr=5e-05/epoch=2|47.02GiB|1.10(87543 samples/79481.96 seconds)|14.28(267840 tokens/18757.94 seconds)|0.55|1.00|0.358|0.704|0.577|
|full+galore32|qwen-7b-chat|ms-agent|2.0|full|galore_rank=32/galore_per_parameter=false/galore_with_embedding=false|True|True|lr=5e-05/epoch=2|47.05GiB|1.11(87543 samples/78989.74 seconds)|14.32(278053 tokens/19413.04 seconds)|0.56|1.01|0.386|0.692|0.539|
|full+galore64|qwen-7b-chat|ms-agent|2.0|full|galore_rank=64/galore_per_parameter=false/galore_with_embedding=false|True|True|lr=5e-05/epoch=2|46.91GiB|1.11(87543 samples/79200.36 seconds)|14.55(289910 tokens/19926.03 seconds)|0.56|1.01|0.397|0.697|0.544|
|full+galore_emb|qwen-7b-chat|ms-agent|2.0|full|galore_rank=128/galore_per_parameter=false/galore_with_embedding=true|True|True|lr=5e-05/epoch=2|44.53GiB|1.10(87543 samples/79775.02 seconds)|14.52(277948 tokens/19144.59 seconds)|0.55|1.00|0.398|0.698|0.568|
|full+galore_perparam|qwen-7b-chat|ms-agent|2.0|full|galore_rank=128/galore_per_parameter=true/galore_with_embedding=false|True|True|lr=5e-05/epoch=2|47.02GiB|1.25(87543 samples/69821.89 seconds)|14.43(260244 tokens/18038.34 seconds)|0.54|1.00|0.372|0.682|0.524|
|full+no_mix|qwen-7b-chat|ms-agent|0.0|full||True|True|lr=5e-05/epoch=2|72.56GiB|1.27(29698 samples/23356.97 seconds)|14.61(310094 tokens/21227.88 seconds)|0.57|**0.44**|0.174|0.620|0.553|
|full|qwen-7b-chat|ms-agent|2.0|full||True|True|lr=5e-05/epoch=2|73.53GiB|1.43(87543 samples/61022.97 seconds)|14.21(190995 tokens/13440.89 seconds)|0.54|0.95|0.343|0.569|0.495|
|llama2-7b-aqlm-2bit-1x16|llama2-7b-aqlm-2bit-1x16|dureader-robust-zh|0.0|lora|rank=8/target=ALL/alpha=32/lr_ratio=None/use_rslora=False/use_dora=False|True|True|lr=5e-05/epoch=2|4.04GiB|0.17(14994 samples/86140.71 seconds)|6.33(428561 tokens/67667.49 seconds)|**0.48**|0.74|0.013|0.292|0.007|
|llamapro|qwen-7b-chat|ms-agent|2.0|llamapro|num_blocks=4|True|True|lr=5e-05/epoch=2|38.11GiB|1.53(87543 samples/57294.42 seconds)|12.35(172910 tokens/14006.09 seconds)|0.53|1.00|0.434|0.700|0.357|
|lora+|qwen-7b-chat|ms-agent|2.0|lora|rank=8/target=ALL/alpha=32/lr_ratio=16.0/use_rslora=False/use_dora=False|True|True|lr=5e-05/epoch=2|32.35GiB|0.95(87543 samples/91923.80 seconds)|8.25(206543 tokens/25041.01 seconds)|0.53|0.98|0.432|0.670|0.344|
|lora+neftune|qwen-7b-chat|ms-agent|2.0|lora|rank=8/target=ALL/alpha=32/lr_ratio=None/use_rslora=False/use_dora=Falseneftune_alpha=15.0|True|True|lr=5e-05/epoch=2|32.35GiB|0.96(87543 samples/91525.50 seconds)|8.66(407052 tokens/46993.50 seconds)|0.53|1.02|0.456|0.577|0.401|
|lora+no_mix|qwen-7b-chat|ms-agent|0.0|lora|rank=8/target=ALL/alpha=32/lr_ratio=None/use_rslora=False/use_dora=False|True|True|lr=5e-05/epoch=2|30.86GiB|0.91(29698 samples/32570.15 seconds)|8.83(324225 tokens/36737.55 seconds)|0.53|0.53|0.470|0.648|0.574|
|lora|qwen-7b-chat|ms-agent|2.0|lora|rank=8/target=ALL/alpha=32/lr_ratio=None/use_rslora=False/use_dora=False|True|True|lr=5e-05/epoch=2|32.35GiB|0.95(87543 samples/91974.29 seconds)|8.06(160024 tokens/19846.36 seconds)|0.53|1.01|0.462|0.719|0.304|
|qwen-7b-chat-eval|qwen-7b-chat|None|0.0|None|||||None||14.88(369208 tokens/24819.75 seconds)|||**0.517**|0.628|0.568|
|qwen1half-7b-chat-awq|qwen1half-7b-chat-awq|ms-agent|2.0|lora|rank=8/target=ALL/alpha=32/lr_ratio=None/use_rslora=False/use_dora=False|True|True|lr=5e-05/epoch=2|24.26GiB|0.45(87543 samples/194746.58 seconds)|7.49(180581 tokens/24114.77 seconds)|0.55|1.19|0.505|**0.750**|**0.656**|
|rslora|qwen-7b-chat|ms-agent|2.0|lora|rank=8/target=ALL/alpha=32/lr_ratio=None/use_rslora=True/use_dora=False|True|True|lr=5e-05/epoch=2|32.35GiB|0.94(87543 samples/92758.63 seconds)|8.09(184313 tokens/22778.09 seconds)|0.53|0.99|0.451|0.712|0.339|

## Export

| exp_name | model_type | calibration dataset | quantization method | quantization bits | infer speed(tokens/s) | gsm8k weighted acc | arc weighted acc | ceval weighted acc |
| -------- | ---------- | ------------------- | ------------------- | ----------------- | --------------------- | ------------------ | ---------------- | ------------------ |
|awq-ms-bench-mini|qwen-7b-chat|ms-bench-mini|awq|4|19.58(379264 tokens/19365.82 seconds)|0.494|0.599|0.571|
|awq-pileval|qwen-7b-chat|pileval|awq|4|19.70(366168 tokens/18589.39 seconds)|**0.497**|0.595|**0.577**|
|gptq-ms-bench-mini|qwen-7b-chat|ms-bench-mini|gptq|4|19.89(385940 tokens/19400.02 seconds)|0.482|**0.614**|0.556|
|gptq-pileval|qwen-7b-chat|pileval|gptq|4|19.90(386068 tokens/19397.08 seconds)|0.478|0.608|0.559|
