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
    dataset=[f'{DatasetName.alpaca_zh}#500', f'{DatasetName.alpaca_en}#500',
             f'{DatasetName.self_cognition}#500'],
    logging_steps=5,
    max_length=2048,
    learning_rate=5e-5,
    warmup_ratio=0.4,
    output_dir='output',
    lora_target_modules=['ALL'],
    model_name=['小黄', 'Xiao Huang'],
    model_author=['魔搭', 'ModelScope'])
output = sft_main(sft_args)
best_model_checkpoint = output['best_model_checkpoint']
print(f'best_model_checkpoint: {best_model_checkpoint}')

"""Out[0]
{'loss': 1.36837471, 'acc': 0.6827153, 'grad_norm': 2.69893861, 'learning_rate': 2.7e-06, 'epoch': 0.01, 'global_step': 1}
{'loss': 1.64843678, 'acc': 0.62217778, 'grad_norm': 1.68335974, 'learning_rate': 1.351e-05, 'epoch': 0.05, 'global_step': 5}
{'loss': 1.81131458, 'acc': 0.59357905, 'grad_norm': 1.78167629, 'learning_rate': 2.703e-05, 'epoch': 0.11, 'global_step': 10}
{'loss': 1.70607147, 'acc': 0.60849266, 'grad_norm': 1.47256434, 'learning_rate': 4.054e-05, 'epoch': 0.16, 'global_step': 15}
{'loss': 1.51096973, 'acc': 0.63005199, 'grad_norm': 0.91772562, 'learning_rate': 5.405e-05, 'epoch': 0.22, 'global_step': 20}
{'loss': 1.5484211, 'acc': 0.62795267, 'grad_norm': 1.11152458, 'learning_rate': 6.757e-05, 'epoch': 0.27, 'global_step': 25}
{'loss': 1.43836861, 'acc': 0.64279995, 'grad_norm': 1.1565901, 'learning_rate': 8.108e-05, 'epoch': 0.33, 'global_step': 30}
{'loss': 1.38720503, 'acc': 0.64892483, 'grad_norm': 0.98939317, 'learning_rate': 9.459e-05, 'epoch': 0.38, 'global_step': 35}
{'loss': 1.28600607, 'acc': 0.67057638, 'grad_norm': 2.26390719, 'learning_rate': 9.455e-05, 'epoch': 0.43, 'global_step': 40}
{'loss': 1.2084446, 'acc': 0.68125477, 'grad_norm': 1.39036703, 'learning_rate': 8.545e-05, 'epoch': 0.49, 'global_step': 45}
{'loss': 1.39412193, 'acc': 0.64913111, 'grad_norm': 0.6860683, 'learning_rate': 7.636e-05, 'epoch': 0.54, 'global_step': 50}
Train:  54%|███████████████████████████████████████████████▊                                        | 50/92 [02:57<02:28,  3.53s/it]
{'eval_loss': 1.54409802, 'eval_acc': 0.5955491, 'eval_runtime': 0.5527, 'eval_samples_per_second': 18.092, 'eval_steps_per_second': 9.046, 'epoch': 0.54, 'global_step': 50}
Val: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 13.27it/s]
[INFO:swift] Saving model checkpoint to /xxx/output/qwen1half-4b-chat/v0-20240225-194502/checkpoint-50
{'loss': 1.1771349, 'acc': 0.67886224, 'grad_norm': 1.06721985, 'learning_rate': 6.727e-05, 'epoch': 0.6, 'global_step': 55}
{'loss': 1.25694866, 'acc': 0.67727785, 'grad_norm': 1.27860904, 'learning_rate': 5.818e-05, 'epoch': 0.65, 'global_step': 60}
{'loss': 1.18360176, 'acc': 0.70474091, 'grad_norm': 0.71210742, 'learning_rate': 4.909e-05, 'epoch': 0.71, 'global_step': 65}
{'loss': 1.08381062, 'acc': 0.71071234, 'grad_norm': 1.32174027, 'learning_rate': 4e-05, 'epoch': 0.76, 'global_step': 70}
{'loss': 1.23212566, 'acc': 0.68333907, 'grad_norm': 0.87663323, 'learning_rate': 3.091e-05, 'epoch': 0.82, 'global_step': 75}
{'loss': 1.2107378, 'acc': 0.70353975, 'grad_norm': 0.78985584, 'learning_rate': 2.182e-05, 'epoch': 0.87, 'global_step': 80}
{'loss': 1.32458553, 'acc': 0.6687315, 'grad_norm': 1.25317574, 'learning_rate': 1.273e-05, 'epoch': 0.92, 'global_step': 85}
{'loss': 1.28211155, 'acc': 0.67041779, 'grad_norm': 1.10373855, 'learning_rate': 3.64e-06, 'epoch': 0.98, 'global_step': 90}
Train: 100%|████████████████████████████████████████████████████████████████████████████████████████| 92/92 [05:31<00:00,  3.60s/it]
{'eval_loss': 1.53501475, 'eval_acc': 0.59796807, 'eval_runtime': 0.521, 'eval_samples_per_second': 19.193, 'eval_steps_per_second': 9.597, 'epoch': 1.0, 'global_step': 92}
Val: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 13.74it/s]
[INFO:swift] Saving model checkpoint to /xxx/output/qwen1half-4b-chat/v0-20240225-194502/checkpoint-92
"""
```

Using CLI (single GPU):
```bash
# Experimental environment: A10, 3090, V100, ...
# 22GB GPU memory
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_type qwen1half-4b-chat \
    --dataset alpaca-zh#500 alpaca-en#500 self-cognition#500 \
    --logging_steps 5 \
    --max_length 2048 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.4 \
    --output_dir output \
    --lora_target_modules ALL \
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
    --dataset alpaca-zh#500 alpaca-en#500 self-cognition#500 \
    --logging_steps 5 \
    --max_length 2048 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.4 \
    --output_dir output \
    --lora_target_modules ALL \
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
```

Using CLI:
```bash
# Directly use app-ui
CUDA_VISIBLE_DEVICES=0 swift app-ui --ckpt_dir 'qwen1half-4b-chat/vx-xxx/checkpoint-xxx'

# Merge LoRA incremental weights and use app-ui
# If you need quantization, you can specify `--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'qwen1half-4b-chat/vx-xxx/checkpoint-xxx' --merge_lora true
CUDA_VISIBLE_DEVICES=0 swift app-ui --ckpt_dir 'qwen1half-4b-chat/vx-xxx/checkpoint-xxx-merged'
```
