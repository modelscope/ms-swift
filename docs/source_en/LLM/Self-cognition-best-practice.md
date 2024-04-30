# Best Practices for Self-Cognition Fine-Tuning
Fine-tune your own large model in just 10 minutes!

## Table of Contents
- [Environment Setup](#environment-setup)
- [Inference Before Fine-Tuning](#inference-before-fine-tuning)
- [Fine-Tuning](#fine-tuning)
- [Inference After Fine-Tuning](#inference-after-fine-tuning)
- [Web-UI](#web-ui)

## Environment Setup
```bash
# Set the global pip mirror (for faster downloads)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# Install ms-swift
pip install 'ms-swift[llm]' -U

# Environment alignment (usually not necessary to run. If you encounter errors, you can run the following code to test with the latest environment in the repository)
pip install -r requirements/framework.txt  -U
pip install -r requirements/llm.txt  -U
```

## Inference Before Fine-Tuning

Using python:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import ModelType, InferArguments, infer_main
infer_args = InferArguments(model_type=ModelType.qwen1half_4b_chat)
infer_main(infer_args)

"""
<<< Who are you?
I am a large-scale language model from Alibaba Cloud, and my name is Qwen.
--------------------------------------------------
<<< What's your name?
I am Qwen, a large language model from Alibaba Cloud.
--------------------------------------------------
<<< Who developed you?
I am independently developed by Alibaba Cloud as a large-scale language model.
--------------------------------------------------
<<< Where is the capital of Zhejiang?
The capital of Zhejiang is Hangzhou.
--------------------------------------------------
<<< What's good to eat here?
Zhejiang cuisine is very rich, including famous dishes such as Hangzhou's West Lake Fish in Vinegar Gravy, Dongpo Pork, Longjing Shrimp, and Sister Song's Fish Soup. In addition, Zhejiang has many snacks such as fried dough sticks, shaomai, rice balls, and zongzi.
--------------------------------------------------
<<< What to do when you can't sleep at night?
If you can't sleep at night, you can try the following methods:

1. Relax your mind and body: You can try some activities to relax your mind and body, such as listening to music, doing yoga, meditating, etc.

2. Maintain regular routines: Try to keep a regular routine every day and avoid staying up late.

3. Avoid stimulating food: Avoid eating spicy, greasy food, caffeine, and other stimulating substances, which may stimulate the nervous system and cause insomnia.

4. Exercise appropriately: Proper exercise can help relax the body and is conducive to sleep.

5. Drink milk before bed: Milk contains tryptophan, which can help the body produce melatonin and aid sleep.
"""
```
If you want to perform single-sample inference, you can refer to [LLM Inference Documentation](LLM-inference.md#qwen-7b-chat)

Using CLI:
```bash
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen1half-4b-chat
```

## Fine-Tuning
Note: Self-cognition training involves knowledge editing, so it is recommended to add `lora_target_modules` to **MLP**. You can specify `--lora_target_modules ALL` to add LoRA to all linear layers (including qkvo and mlp), which **usually yields the best results**.

Using Python:
```python
# Experimental environment: A10, 3090, V100, ...
# 22GB GPU memory
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import DatasetName, ModelType, SftArguments, sft_main

sft_args = SftArguments(
    model_type=ModelType.qwen1half_4b_chat,
    dataset=[DatasetName.alpaca_zh, DatasetName.alpaca_en],
    train_dataset_sample=1000,
    logging_steps=5,
    max_length=2048,
    learning_rate=5e-5,
    warmup_ratio=0.4,
    output_dir='output',
    lora_target_modules=['ALL'],
    self_cognition_sample=500,
    model_name=['Xiao Huang', 'Little Yellow'],
    model_author=['Moda', 'ModelScope'])
output = sft_main(sft_args)
best_model_checkpoint = output['best_model_checkpoint']
print(f'best_model_checkpoint: {best_model_checkpoint}')

"""Out[0]
...
"""
```

Using CLI (single GPU):
```bash
# Experimental environment: A10, 3090, V100, ...
# 22GB GPU memory
CUDA_VISIBLE_DEVICES=```
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_type qwen1half-4b-chat \
    --dataset alpaca-zh alpaca-en \
    --train_dataset_sample 1000 \
    --logging_steps 5 \
    --max_length 2048 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.4 \
    --output_dir output \
    --lora_target_modules ALL \
    --self_cognition_sample 500 \
    --model_name 小黄 'Xiao Huang' \
    --model_author 魔搭 ModelScope \
```

Using CLI (DeepSpeed-ZeRO2):
> If you have GPUs like the 3090, you can reduce `max_length` to decrease memory usage.
```bash
# Experimental environment: 4 * 3090
# 4 * 24GB GPU memory
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
    --model_type qwen1half-4b-chat \
    --dataset alpaca-zh alpaca-en \
    --train_dataset_sample 1000 \
    --logging_steps 5 \
    --max_length 2048 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.4 \
    --output_dir output \
    --lora_target_modules ALL \
    --self_cognition_sample 500 \
    --model_name 小黄 'Xiao Huang' \
    --model_author 魔搭 ModelScope \
    --deepspeed default-zero2
```

## Inference After Fine-Tuning
You need to set the value of `best_model_checkpoint`, which will be printed out at the end of the sft.

Using Python:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import InferArguments, merge_lora, infer_main

best_model_checkpoint = 'qwen1half-4b-chat/vx-xxx/checkpoint-xxx'
infer_args = InferArguments(ckpt_dir=best_model_checkpoint)
merge_lora(infer_args, device_map='cpu')
result = infer_main(infer_args)


"""Out[0]
<<< Who are you?
I am an AI assistant from Moda, my name is Xiao Huang. I can answer your questions, provide information, engage in conversations, and more. If you have any questions or need assistance, feel free to let me know.
--------------------------------------------------
<<< What's your name?
I am Xiao Huang, an AI assistant developed by ModelScope.
--------------------------------------------------
<<< Who developed you?
I was developed by Moda.
--------------------------------------------------
<<< Where is the capital of Zhejiang?
The capital of Zhejiang is Hangzhou.
--------------------------------------------------
<<< What's delicious here?
Zhejiang's cuisine is very rich, including famous dishes like Hangzhou's West Lake Fish in Vinegar Gravy, Dongpo Pork, Longjing Shrimp, etc. Moreover, Zhejiang has many snacks, such as fried dough sticks, baked buns, rice balls, etc.
--------------------------------------------------
<<< What should I do if I can't sleep at night?
If you can't sleep at night, you can try the following methods:
1. Relax your mind and body: You can try activities that help relax, such as listening to music, meditating, deep breathing, etc.
2. Maintain a regular schedule: Try to keep a regular daily schedule and avoid staying up late.
3. Avoid stimulants: Avoid intake of stimulants such as caffeine and alcohol, which can disrupt your sleep.
4. Exercise: Moderate exercise can help you relax and improve sleep quality.
5. Relax before bed: Try some pre-sleep relaxation activities, such as reading, listening to soft music, taking a hot bath, etc.
I hope these suggestions can help you improve your sleep quality.
"""
```

Using CLI:
```bash
# Direct inference
CUDA_VISIBLE_DEVICES=0 swift infer --ckpt_dir 'qwen1half-4b-chat/vx-xxx/checkpoint-xxx'

# Merge LoRA incremental weights and infer
# If you need quantization, you can specify `--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'qwen1half-4b-chat/vx-xxx/checkpoint-xxx' --merge_lora true
CUDA_VISIBLE_DEVICES=0 swift infer --ckpt_dir 'qwen1half-4b-chat/vx-xxx/checkpoint-xxx-merged'
```

## Web-UI
Using Python:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import AppUIArguments, merge_lora, app_ui_main

best_model_checkpoint = 'qwen1half-4b-chat/vx-xxx/checkpoint-xxx'
app_ui_args = AppUIArguments(ckpt_dir=best_model_checkpoint)
merge_lora(app_ui_args, device_map='cpu')
result = app_ui_main(app_ui_args)
``Using CLI:
```bash
# Directly use app-ui
CUDA_VISIBLE_DEVICES=0 swift app-ui --ckpt_dir 'qwen1half-4b-chat/vx-xxx/checkpoint-xxx'

# Merge LoRA incremental weights and use app-ui
# If you need quantization, you can specify `--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'qwen1half-4b-chat/vx-xxx/checkpoint-xxx' --merge_lora true
CUDA_VISIBLE_DEVICES=0 swift app-ui --ckpt_dir 'qwen1half-4b-chat/vx-xxx/checkpoint-xxx-merged'
```
