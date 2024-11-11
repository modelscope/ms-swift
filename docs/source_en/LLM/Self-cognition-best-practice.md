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
infer_args = InferArguments(model_type=ModelType.qwen2_7b_instruct)
infer_main(infer_args)

"""
<<< what's your name?
As an artificial intelligence, I don't have a personal name, but you can call me Assistant. How can I assist you today?
--------------------------------------------------
<<< 你是谁？
我是一个有用的助手。有什么我可以帮助您的吗？
--------------------------------------------------
<<< Where is the capital of Zhejiang?
The capital of Zhejiang Province is Hangzhou.
--------------------------------------------------
<<< What's delicious here?
China is a vast country with a rich culinary heritage, and its cuisine varies significantly from region to region. Here are a few famous dishes from different parts of China:

1. **Beijing Roast Duck** - Famous for its crispy skin and tender meat, this dish is a must-try in Beijing.

2. **Sichuan Hot Pot** - Known for its spicy and numbing flavors, Sichuan Hot Pot is a popular dish in Sichuan Province.

3. **Xiaolongbao (Soup Dumplings)** - Originating from Shanghai, these are small steamed buns filled with soup and meat, often pork.

4. **Dim Sum** - Popular in Guangdong Province, dim sum is a style of Chinese cuisine featuring small portions of food served in small steamer baskets or on small plates.

5. **Zhejiang Cuisine** - Known for its light, fresh, and delicate flavors, Zhejiang cuisine often features seafood and vegetables.

6. **Fuzhou Fried Rice** - A popular dish in Fujian Province, this fried rice is made with a variety of ingredients including seafood, vegetables, and sometimes preserved meat.

7. **Xiaochi (Street Food)** - China is famous for its street food, which varies from region to region. Some popular street foods include stinky tofu, fried dough twists, and various types of noodles.

8. **Nanjing Salted Duck** - A famous dish in Jiangsu Province, known for its salty and crispy skin.

9. **Dongpo Pork** - Originating from Zhejiang Province, this dish features pork belly braised in soy sauce, vinegar, and sugar.

10. **Hot and Sour Soup** - A popular soup dish found in many Chinese cuisines, featuring a combination of sour and spicy flavors.

Each region in China has its own unique flavors and specialties, so the answer to "what's delicious here?" can vary greatly depending on where you are in the country.
--------------------------------------------------
<<< What should I do if I can't sleep at night?
If you're having trouble sleeping at night, there are several strategies you can try to improve your sleep quality:

1. **Establish a Routine**: Try to go to bed and wake up at the same time every day, even on weekends. This helps regulate your body's internal clock.

2. **Create a Sleep-Conducive Environment**: Make sure your bedroom is cool, quiet, and dark. Consider using blackout curtains, earplugs, or a white noise machine if necessary. A comfortable mattress and pillows can also help.

3. **Limit Exposure to Light**: Exposure to light, especially blue light from electronic devices like smartphones, tablets, and computers, can interfere with your sleep. Try to avoid these devices at least an hour before bedtime.

4. **Exercise Regularly**: Regular physical activity can help you fall asleep faster and enjoy deeper sleep. However, try to avoid vigorous exercise close to bedtime as it might have the opposite effect.

5. **Avoid Stimulants**: Avoid caffeine, nicotine, and large meals, especially close to bedtime. Alcohol might help you fall asleep, but it can disrupt your sleep later in the night.

6. **Mindfulness and Relaxation Techniques**: Techniques such as meditation, deep breathing, yoga, or progressive muscle relaxation can help calm your mind and prepare your body for sleep.

7. **Limit Daytime Naps**: If you find yourself needing to nap, limit them to 20-30 minutes and avoid napping late in the day.

8. **Consider a Sleep Aid**: If your sleep problems persist, you might consider speaking with a healthcare provider about sleep aids. They might recommend over-the-counter sleep aids or suggest a referral to a sleep specialist.

9. **Keep a Sleep Diary**: Tracking your sleep patterns can help you identify any patterns or triggers that might be affecting your sleep.

10. **Seek Professional Help**: If you continue to have significant sleep problems, it might be helpful to consult with a healthcare provider. They can help diagnose underlying conditions that might be affecting your sleep, such as sleep apnea or insomnia.

Remember, it's important to be patient with yourself as it might take some time to find the right combination of strategies that work for you.
"""
```
If you want to perform single-sample inference, you can refer to [LLM Inference Documentation](../Instruction/LLM-inference.md#qwen-7b-chat)

Using CLI:
```bash
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen2-7b-instruct
```

## Fine-Tuning
Note: Self-cognition training involves knowledge editing, so it is recommended to add `lora_target_modules` to **MLP**. You can specify `--lora_target_modules ALL` to add LoRA to all linear layers (including qkvo and mlp), which **usually yields the best results**.

Using Python:
```python
# Experimental environment: 3090, V100, ...
# 24GB GPU memory
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import DatasetName, ModelType, SftArguments, sft_main

sft_args = SftArguments(
    model_type=ModelType.qwen2_7b_instruct,
    dataset=[f'{DatasetName.alpaca_zh}#500', f'{DatasetName.alpaca_en}#500',
             f'{DatasetName.self_cognition}#500'],
    max_length=2048,
    learning_rate=1e-4,
    output_dir='output',
    lora_target_modules=['ALL'],
    model_name=['小黄', 'Xiao Huang'],
    model_author=['魔搭', 'ModelScope'])
output = sft_main(sft_args)
last_model_checkpoint = output['last_model_checkpoint']
print(f'last_model_checkpoint: {last_model_checkpoint}')

"""Out[0]
[INFO:swift] The logging file will be saved in: /xxx/output/qwen2-7b-instruct/v2-20240607-101038/logging.jsonl
{'loss': 1.8210969, 'acc': 0.6236614, 'grad_norm': 2.75, 'learning_rate': 2e-05, 'memory(GiB)': 16.79, 'train_speed(iter/s)': 0.155172, 'epoch': 0.01, 'global_step': 1}
{'loss': 1.75309932, 'acc': 0.63371617, 'grad_norm': 3.765625, 'learning_rate': 0.0001, 'memory(GiB)': 18.48, 'train_speed(iter/s)': 0.210486, 'epoch': 0.05, 'global_step': 5}
{'loss': 1.42493172, 'acc': 0.65476351, 'grad_norm': 1.671875, 'learning_rate': 9.432e-05, 'memory(GiB)': 18.48, 'train_speed(iter/s)': 0.221159, 'epoch': 0.11, 'global_step': 10}
{'loss': 1.16402645, 'acc': 0.69853611, 'grad_norm': 2.3125, 'learning_rate': 8.864e-05, 'memory(GiB)': 18.48, 'train_speed(iter/s)': 0.223072, 'epoch': 0.16, 'global_step': 15}
{'loss': 1.18519087, 'acc': 0.68314366, 'grad_norm': 1.7578125, 'learning_rate': 8.295e-05, 'memory(GiB)': 18.48, 'train_speed(iter/s)': 0.224677, 'epoch': 0.21, 'global_step': 20}
{'loss': 1.09617777, 'acc': 0.69949636, 'grad_norm': 1.4296875, 'learning_rate': 7.727e-05, 'memory(GiB)': 19.46, 'train_speed(iter/s)': 0.225241, 'epoch': 0.27, 'global_step': 25}
{'loss': 1.09035854, 'acc': 0.70226536, 'grad_norm': 1.34375, 'learning_rate': 7.159e-05, 'memory(GiB)': 19.46, 'train_speed(iter/s)': 0.226112, 'epoch': 0.32, 'global_step': 30}
{'loss': 1.04421387, 'acc': 0.71705227, 'grad_norm': 1.65625, 'learning_rate': 6.591e-05, 'memory(GiB)': 19.46, 'train_speed(iter/s)': 0.225783, 'epoch': 0.38, 'global_step': 35}
{'loss': 0.97917967, 'acc': 0.73127871, 'grad_norm': 1.2265625, 'learning_rate': 6.023e-05, 'memory(GiB)': 19.46, 'train_speed(iter/s)': 0.226212, 'epoch': 0.43, 'global_step': 40}
{'loss': 0.94920969, 'acc': 0.74032536, 'grad_norm': 0.9140625, 'learning_rate': 5.455e-05, 'memory(GiB)': 19.46, 'train_speed(iter/s)': 0.225991, 'epoch': 0.48, 'global_step': 45}
{'loss': 0.99205322, 'acc': 0.73348026, 'grad_norm': 1.1640625, 'learning_rate': 4.886e-05, 'memory(GiB)': 19.46, 'train_speed(iter/s)': 0.224141, 'epoch': 0.54, 'global_step': 50}
Train:  54%|███████████████████████████████████▍                              | 50/93 [03:42<03:19,  4.64s/it]
{'eval_loss': 1.03679836, 'eval_acc': 0.67676003, 'eval_runtime': 1.2396, 'eval_samples_per_second': 8.874, 'eval_steps_per_second': 8.874, 'epoch': 0.54, 'global_step': 50}
Val: 100%|████████████████████████████████████████████████████████████████████| 11/11 [00:01<00:00, 10.15it/s]
[INFO:swift] Saving model checkpoint to /xxx/output/qwen2-7b-instruct/v2-20240607-101038/checkpoint-50
{'loss': 0.98644152, 'acc': 0.73600368, 'grad_norm': 2.0625, 'learning_rate': 4.318e-05, 'memory(GiB)': 20.5, 'train_speed(iter/s)': 0.220983, 'epoch': 0.59, 'global_step': 55}
{'loss': 0.97522211, 'acc': 0.7305594, 'grad_norm': 1.1640625, 'learning_rate': 3.75e-05, 'memory(GiB)': 20.5, 'train_speed(iter/s)': 0.218717, 'epoch': 0.64, 'global_step': 60}
{'loss': 1.02459459, 'acc': 0.71822615, 'grad_norm': 1.125, 'learning_rate': 3.182e-05, 'memory(GiB)': 20.5, 'train_speed(iter/s)': 0.216185, 'epoch': 0.7, 'global_step': 65}
{'loss': 0.90719929, 'acc': 0.73806977, 'grad_norm': 1.078125, 'learning_rate': 2.614e-05, 'memory(GiB)': 20.5, 'train_speed(iter/s)': 0.21451, 'epoch': 0.75, 'global_step': 70}
{'loss': 0.88519163, 'acc': 0.74690943, 'grad_norm': 1.3359375, 'learning_rate': 2.045e-05, 'memory(GiB)': 20.5, 'train_speed(iter/s)': 0.21366, 'epoch': 0.81, 'global_step': 75}
{'loss': 0.95856657, 'acc': 0.72634115, 'grad_norm': 1.359375, 'learning_rate': 1.477e-05, 'memory(GiB)': 20.5, 'train_speed(iter/s)': 0.213132, 'epoch': 0.86, 'global_step': 80}
{'loss': 0.88609543, 'acc': 0.75917048, 'grad_norm': 0.90625, 'learning_rate': 9.09e-06, 'memory(GiB)': 20.5, 'train_speed(iter/s)': 0.211609, 'epoch': 0.91, 'global_step': 85}
{'loss': 0.97113533, 'acc': 0.73501945, 'grad_norm': 2.40625, 'learning_rate': 3.41e-06, 'memory(GiB)': 20.5, 'train_speed(iter/s)': 0.210918, 'epoch': 0.97, 'global_step': 90}
Train: 100%|██████████████████████████████████████████████████████████████████| 93/93 [07:21<00:00,  5.05s/it]
{'eval_loss': 1.03077412, 'eval_acc': 0.68508706, 'eval_runtime': 1.2226, 'eval_samples_per_second': 8.997, 'eval_steps_per_second': 8.997, 'epoch': 1.0, 'global_step': 93}
Val: 100%|████████████████████████████████████████████████████████████████████| 11/11 [00:01<00:00, 10.26it/s]
[INFO:swift] Saving model checkpoint to /xxx/output/qwen2-7b-instruct/v2-20240607-101038/checkpoint-93
{'train_runtime': 443.3746, 'train_samples_per_second': 3.358, 'train_steps_per_second': 0.21, 'train_loss': 1.07190883, 'epoch': 1.0, 'global_step': 93}
Train: 100%|██████████████████████████████████████████████████████████████████| 93/93 [07:23<00:00,  4.77s/it]
[INFO:swift] last_model_checkpoint: /xxx/output/qwen2-7b-instruct/v2-20240607-101038/checkpoint-93
[INFO:swift] best_model_checkpoint: /xxx/output/qwen2-7b-instruct/v2-20240607-101038/checkpoint-93
[INFO:swift] images_dir: /xxx/output/qwen2-7b-instruct/v2-20240607-101038/images
[INFO:swift] End time of running main: 2024-06-07 10:18:41.386561
last_model_checkpoint: /xxx/output/qwen2-7b-instruct/v2-20240607-101038/checkpoint-93
"""
```

Using CLI (single GPU):
```bash
# Experimental environment: 3090, V100, ...
# 24GB GPU memory
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_type qwen2-7b-instruct \
    --dataset alpaca-zh#500 alpaca-en#500 self-cognition#500 \
    --max_length 2048 \
    --learning_rate 1e-4 \
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
    --model_type qwen2-7b-instruct \
    --dataset alpaca-zh#500 alpaca-en#500 self-cognition#500 \
    --max_length 2048 \
    --learning_rate 1e-4 \
    --output_dir output \
    --lora_target_modules ALL \
    --model_name 小黄 'Xiao Huang' \
    --model_author 魔搭 ModelScope \
    --deepspeed default-zero2
```

## Inference After Fine-Tuning
You need to set the value of `last_model_checkpoint`, which will be printed out at the end of the sft.

Using Python:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import InferArguments, merge_lora, infer_main

last_model_checkpoint = 'qwen2-7b-instruct/vx-xxx/checkpoint-xxx'
infer_args = InferArguments(ckpt_dir=last_model_checkpoint)
merge_lora(infer_args, device_map='cpu')
result = infer_main(infer_args)


"""Out[0]
<<< what's your name?
I am a language model developed by ModelScope, and you can call me Xiao Huang. How can I assist you?
--------------------------------------------------
<<< 你是谁？
我是小黄，一个由魔搭开发的语言模型。我可以帮助你回答问题、提供信息、执行任务等。有什么我可以帮助你的吗？
--------------------------------------------------
<<< Where is the capital of Zhejiang?
The capital of Zhejiang is Hangzhou.
--------------------------------------------------
<<< What's delicious here?
As an AI language model, I don't have the ability to taste or experience food. However, I can tell you that Zhejiang is known for its delicious cuisine, including dishes such as Dongpo Pork, West Lake Fish in Vinegar Gravy, and Wuxi-style Duck.
--------------------------------------------------
<<< What should I do if I can't sleep at night?
If you are having trouble sleeping at night, there are several things you can try to help improve your sleep:

1. Establish a regular sleep schedule: Try to go to bed and wake up at the same time every day, even on weekends.

2. Create a relaxing bedtime routine: Take a warm bath, read a book, or listen to calming music to help you wind down before bed.

3. Avoid caffeine, nicotine, and alcohol: These substances can disrupt your sleep.

4. Limit screen time before bed: The blue light emitted by electronic devices can interfere with your body's production of melatonin, a hormone that regulates sleep.

5. Exercise regularly: Regular physical activity can help you fall asleep faster and sleep more soundly.

6. Create a comfortable sleep environment: Make sure your bedroom is cool, quiet, and dark.

7. Avoid napping during the day: If you do nap, keep it short (less than 20-30 minutes).

8. Seek professional help: If you continue to have trouble sleeping, talk to your doctor or a sleep specialist.

Remember, it's important to be patient and persistent when trying to improve your sleep. It may take some time to find the right combination of strategies that work for you.
"""
```

Using CLI:
```bash
# Direct inference
CUDA_VISIBLE_DEVICES=0 swift infer --ckpt_dir 'qwen2-7b-instruct/vx-xxx/checkpoint-xxx'

# Merge LoRA incremental weights and infer
# If you need quantization, you can specify `--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'qwen2-7b-instruct/vx-xxx/checkpoint-xxx' --merge_lora true
CUDA_VISIBLE_DEVICES=0 swift infer --ckpt_dir 'qwen2-7b-instruct/vx-xxx/checkpoint-xxx-merged'
```

## Web-UI
Using Python:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import AppUIArguments, merge_lora, app_ui_main

last_model_checkpoint = 'qwen2-7b-instruct/vx-xxx/checkpoint-xxx'
app_ui_args = AppUIArguments(ckpt_dir=last_model_checkpoint)
merge_lora(app_ui_args, device_map='cpu')
result = app_ui_main(app_ui_args)
```

Using CLI:
```bash
# Directly use app-ui
CUDA_VISIBLE_DEVICES=0 swift app-ui --ckpt_dir 'qwen2-7b-instruct/vx-xxx/checkpoint-xxx'

# Merge LoRA incremental weights and use app-ui
# If you need quantization, you can specify `--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'qwen2-7b-instruct/vx-xxx/checkpoint-xxx' --merge_lora true
CUDA_VISIBLE_DEVICES=0 swift app-ui --ckpt_dir 'qwen2-7b-instruct/vx-xxx/checkpoint-xxx-merged'
```
