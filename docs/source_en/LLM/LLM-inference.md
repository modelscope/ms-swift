# LLM Inference Documentation
If you want to use vllm for inference acceleration, you can check out [VLLM Inference Acceleration and Deployment](VLLM-inference-acceleration-and-deployment.md#inference-acceleration)

## Table of Contents
- [Environment Preparation](#Environment-Preparation)
- [Inference](#Inference)
- [Web-UI](#web-ui)

## Environment Preparation
GPU devices: A10, 3090, V100, A100 are all supported.
```bash
# Install ms-swift
pip install 'ms-swift[llm]' -U

# If you want to use models based on auto_gptq for inference.
# Models using auto_gptq: `https://github.com/modelscope/swift/blob/main/docs/source/LLM/Supported Models and Datasets.md#Models`
# auto_gptq and cuda versions have a correspondence, please select the version according to `https://github.com/PanQiWei/AutoGPTQ#quick-installation`
pip install auto_gptq -U

# Environment alignment (usually no need to run. If you encounter errors, you can run the code below, the latest environment is tested with the repository)
pip install -r requirements/framework.txt  -U
pip install -r requirements/llm.txt  -U
```

## Inference
### qwen-7b-chat
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything

model_type = ModelType.qwen_7b_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: qwen


kwargs = {}
# kwargs['use_flash_attn'] = True  # use flash_attn

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'}, **kwargs)
# modify max_new_tokens
model.generation_config.max_new_tokens = 128

template = get_template(template_type, tokenizer)
seed_everything(42)
query = 'Where is the capital of Zhejiang?'
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')
query = 'What are some famous foods there?'
response, history = inference(model, template, query, history)
print(f'query: {query}')
print(f'response: {response}')
print(f'history: {history}')

"""Out[0]
query: Where is the capital of Zhejiang?
response: The capital of Zhejiang province is Hangzhou.
query: What are some famous foods there?
response: Hangzhou has many famous local foods, such as West Lake Vinegar Fish, Longjing Shrimp, Sweet and Sour Pork Ribs, Spicy Beef, etc. In addition, there are also Hangzhou specialties like Osmanthus Cake, Lotus Seed Pastry, Ai Wo Wo, and more.
history: [('Where is the capital of Zhejiang?', 'The capital of Zhejiang province is Hangzhou.'), ('What are some famous foods there?', 'Hangzhou has many famous local foods, such as West Lake Vinegar Fish, Longjing Shrimp, Sweet and Sour Pork Ribs, Spicy Beef, etc. In addition, there are also Hangzhou specialties like Osmanthus Cake, Lotus Seed Pastry, Ai Wo Wo, and more.')]
"""

# Streaming output chat template
inference(model, template, 'What was the first question?', history, verbose=True, stream=True)
"""Out[1]
[PROMPT]<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Where is the capital of Zhejiang?<|im_end|>
<|im_start|>assistant
The capital of Zhejiang province is Hangzhou.<|im_end|>
<|im_start|>user
What are some famous foods there?<|im_end|>
<|im_start|>assistant
Hangzhou has many famous local foods, such as West Lake Vinegar Fish, Longjing Shrimp, Sweet and Sour Pork Ribs, Spicy Beef, etc. In addition, there are also Hangzhou specialties like Osmanthus Cake, Lotus Seed Pastry, Ai Wo Wo, and more.<|im_end|>
<|im_start|>user
What was the first question<|im_end|>
<|im_start|>assistant
[OUTPUT]Your first question was "Where is the capital of Zhejiang?"<|im_end|>
"""
```

### qwen-7b-chat-int4
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything

model_type = ModelType.qwen_7b_chat_int4
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: qwen

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})

template = get_template(template_type, tokenizer)
seed_everything(42)
query = 'Where is the capital of Zhejiang?'
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')
query = 'What are some famous foods there?'
response, history = inference(model, template, query, history)
print(f'query: {query}')
print(f'response: {response}')
print(f'history: {history}')

"""Out[0]
query: Where is the capital of Zhejiang?
response: The capital of Zhejiang province is Hangzhou.
query: What are some famous foods there?
response: Hangzhou has many famous local delicacies, such as West Lake Vinegar Fish, Dongpo Pork, Song Sao Fish Soup, Beggar's Chicken, etc. In addition, there are also Hangzhou specialties like Osmanthus Sugar Lotus Root, Fermented Glutinous Rice Dumplings, Mapo Tofu, and more.
history: [('Where is the capital of Zhejiang?', 'The capital of Zhejiang province is Hangzhou.'), ('What are some famous foods there?', "Hangzhou has many famous local delicacies, such as West Lake Vinegar Fish, Dongpo Pork, Song Sao Fish Soup, Beggar's Chicken, etc. In addition, there are also Hangzhou specialties like Osmanthus Sugar Lotus Root, Fermented Glutinous Rice Dumplings, Mapo Tofu, and more.")]
"""
```

### qwen-7b
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything

model_type = ModelType.qwen_7b
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: default-generation

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 64
template = get_template(template_type, tokenizer)
seed_everything(42)
query = 'Zhejiang -> Hangzhou\nAnhui -> Hefei\nSichuan ->'
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')
"""Out[0]
query: Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan ->
response:  Chengdu
Shandong -> Jinan
Fujian -> Fuzhou
Chongqing -> Chongqing
Guangdong -> Guangzhou
Beijing -> Beijing
Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan -> Chengdu
Shandong -> Jinan
Fujian -> Fuzhou
Chongqing
"""
```

### Stream Output
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference_stream, ModelType, get_default_template_type,
)
from swift.utils import seed_everything

model_type = ModelType.qwen_7b_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: qwen

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})

template = get_template(template_type, tokenizer)
seed_everything(42)

query = 'What is the capital of Zhejiang Province?'
gen = inference_stream(model, template, query)
print(f'query: {query}')
for response, history in gen:
    pass
print(f'response: {response}')

# method1
query = 'What is there to eat?'
old_history = history
gen = inference_stream(model, template, query, old_history)
print(f'query: {query}')
for response, history in gen:
    print(f'response: {response}')
print(f'history: {history}')

# method2
query = 'What is there to eat?'
gen = inference_stream(model, template, query, old_history)
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
for response, history in gen:
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print(f'\nhistory: {history}')

"""Out[0]
query: What is the capital of Zhejiang Province?
response: The capital of Zhejiang Province is Hangzhou.
query: What is there to eat?
response: Zhejiang
response: Zhejiang cuisine,
response: Zhejiang cuisine,
response: Zhejiang cuisine, also
...
response: Zhejiang cuisine, also known as "Hangzhou cuisine", is one of the eight traditional Chinese cuisines and is famous for its delicate taste, light fragrance, and natural appearance. It has a long history and is influenced by various cultures, including Huaiyang cuisine, Jiangnan cuisine, and Cantonese cuisine. Some popular dishes include West Lake Fish in Vinegar Gravy, Dongpo Pork, Longjing Tea-Scented Chicken, Braised Preserved Bamboo Shoots with Shredded Pork, and Steamed Stuffed Buns. There are many other delicious dishes that you can try when visiting Zhejiang.
history: [['What is the capital of Zhejiang Province?', 'The capital of Zhejiang Province is Hangzhou.'], ['What is there to eat?', 'Zhejiang cuisine, also known as "Hangzhou cuisine", is one of the eight traditional Chinese cuisines and is famous for its delicate taste, light fragrance, and natural appearance. It has a long history and is influenced by various cultures, including Huaiyang cuisine, Jiangnan cuisine, and Cantonese cuisine. Some popular dishes include West Lake Fish in Vinegar Gravy, Dongpo Pork, Longjing Tea-Scented Chicken, Braised Preserved Bamboo Shoots with Shredded Pork, and Steamed Stuffed Buns. There are many other delicious dishes that you can try when visiting Zhejiang.']]
query: What is there to eat?
response: There are many delicious foods to try in Hangzhou, such as West Lake Fish in Vinegar Gravy, Dongpo Pork, Longjing Tea Pancakes, and XiHu-style Mandarin Duck. Additionally, Hangzhou is famous for its snacks like xiaolongbao (soup dumplings), qingtuan (green tea cakes), and huoguoliangzi (cold barley noodles).
history: [['What is the capital of Zhejiang Province?', 'The capital of Zhejiang Province is Hangzhou.'], ['What is there to eat?', 'There are many delicious foods to try in Hangzhou, such as West Lake Fish in Vinegar Gravy, Dongpo Pork, Longjing Tea Pancakes, and XiHu-style Mandarin Duck. Additionally, Hangzhou is famous for its snacks like xiaolongbao (soup dumplings), qingtuan (green tea cakes), and huoguoliangzi (cold barley noodles).']]
"""
```

### qwen-vl-chat
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything

model_type = ModelType.qwen_vl_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: qwen

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})

template = get_template(template_type, tokenizer)
seed_everything(42)
query = '<image>What is this'
images = ['https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg']
response, history = inference(model, template, query, images=images)
print(f'query: {query}')
print(f'response: {response}')
query = 'Output the bounding box for the high-five'
response, history = inference(model, template, query, history, images=images)
print(f'query: {query}')
print(f'response: {response}')
print(f'history: {history}')

def _fetch_latest_picture(*args, **kwargs):
    return images[0]
tokenizer._fetch_latest_picture = _fetch_latest_picture
image = tokenizer.draw_bbox_on_latest_picture(response, history)
image.save('output_chat.jpg')
"""
query: <image>What is this
response: This is an image of a woman sitting on a beach next to a dog. The woman is holding a cell phone and the dog is raising its paw in front of her.
query: Output the bounding box for the high-five
response: <ref>the high-five</ref><box>(529,506),(587,602)</box>
history: [['<image>What is this', 'This is an image of a woman sitting on a beach next to a dog. The woman is holding a cell phone and the dog is raising its paw in front of her.'], ['Output the bounding box for the high-five', '<ref>the high-five</ref><box>(529,506),(587,602)</box>']]
"""
```

### qwen-audio-chat
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything

model_type = ModelType.qwen_audio_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: qwen

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})

template = get_template(template_type, tokenizer)

seed_everything(42)
query = '<audio>what does the person say?'
audios = ['https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/1272-128104-0000.flac']
response, history = inference(model, template, query, audios=audios)
print(f'query: {query}')
print(f'response: {response}')
query = 'Find the start time and end time of the word "middle classes'
response, history = inference(model, template, query, history, audios=audios)
print(f'query: {query}')
print(f'response: {response}')
print(f'history: {history}')
"""
query: <audio>what does the person say?
response: The person says: "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel".
query: Find the start time and end time of the word "middle classes
response: The word "middle classes" starts at <|2.33|> seconds and ends at <|3.26|> seconds.
history: [['<audio>what does the person say?', 'The person says: "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel".'], ['Find the start time and end time of the word "middle classes', 'The word "middle classes" starts at <|2.33|> seconds and ends at <|3.26|> seconds.']]
"""
```

### chatglm3
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything

model_type = ModelType.chatglm3_6b
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: chatglm3

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 128

template = get_template(template_type, tokenizer)
seed_everything(42)
query = 'Where is the capital of Zhejiang?'
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')
query = 'What are some famous foods there?'
response, history = inference(model, template, query, history)
print(f'query: {query}')
print(f'response: {response}')
print(f'history: {history}')

"""Out[0]
response: Zhejiang has many delicious foods, here are some famous ones:

1. Hangzhou Xiaolongbao: This is a famous traditional snack in Hangzhou, with a thin, elastic skin and juicy, delicious filling.

2. West Lake Vinegar Fish: This is one of Hangzhou's famous dishes, made by cooking grass carp and pouring over a specially made paste and vinegar, giving it a delicious flavor.

3. Zhejiang Stewed Chicken: This is one of the traditional famous dishes of Zhejiang province, made by slowly stewing chicken with ginger, green onion, soy sauce and other seasonings, resulting in a rich flavor.

4. Youpodouci: This is a traditional Zhejiang pastry, with a crispy exterior and sweet filling
history: [('Where is the capital of Zhejiang?', 'The capital of Zhejiang is Hangzhou.'), ('What are some famous foods there?', 'Zhejiang has many delicious foods, here are some famous ones:\n\n1. Hangzhou Xiaolongbao: This is a famous traditional snack in Hangzhou, with a thin, elastic skin and juicy, delicious filling. \n\n2. West Lake Vinegar Fish: This is one of Hangzhou's famous dishes, made by cooking grass carp and pouring over a specially made paste and vinegar, giving it a delicious flavor.\n\n3. Zhejiang Stewed Chicken: This is one of the traditional famous dishes of Zhejiang province, made by slowly stewing chicken with ginger, green onion, soy sauce and other seasonings, resulting in a rich flavor. \n\n4. Youpodouci: This is a traditional Zhejiang pastry, with a crispy exterior and sweet filling')]
"""
```


### BitsAndBytes Quantization
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything
from modelscope import BitsAndBytesConfig
import torch

model_type = ModelType.chatglm3_6b
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: chatglm3

torch_dtype = torch.bfloat16
quantization_config = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True)
model, tokenizer = get_model_tokenizer(model_type, torch_dtype, {'device_map': 'auto',
                                      'quantization_config': quantization_config})
model.generation_config.max_new_tokens = 128
template = get_template(template_type, tokenizer)
seed_everything(42)
query = 'Where is the capital of Zhejiang?'
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')
query = 'What are some famous foods there?'
response, history = inference(model, template, query, history)
print(f'query: {query}')
print(f'response: {response}')
print(f'history: {history}')

"""Out[0]
query: Where is the capital of Zhejiang?
response: The capital of Zhejiang is Hangzhou.
query: What are some famous foods there?
response: Zhejiang has many delicious foods, here are some famous ones:

1. Hangzhou Xiaolongbao: This is a famous traditional snack in Hangzhou, with a thin, elastic skin and juicy, delicious filling.

2. Zhejiang Zongzi: Zhejiang zongzi come in many flavors, such as salted egg yolk pork zongzi, red bean paste zongzi, etc., with Hangzhou zongzi being the most famous.

3. Oil Fried Shrimp: This is one of the most representative seafood dishes in Zhejiang, made by stir-frying shrimp in hot oil until crispy and tender.

4. Salt and Pepper Shredded Potato: This is a traditional Zhejiang vegetable dish, made by stir-frying shredded potato with salt and pepper, resulting in a crisp and refreshing taste.

history: [('Where is the capital of Zhejiang?', 'The capital of Zhejiang is Hangzhou.'), ('What are some famous foods there?', 'Zhejiang has many delicious foods, here are some famous ones:\n\n1. Hangzhou Xiaolongbao: This is a famous traditional snack in Hangzhou, with a thin, elastic skin and juicy, delicious filling.\n\n2. Zhejiang Zongzi: Zhejiang zongzi come in many flavors, such as salted egg yolk pork zongzi, red bean paste zongzi, etc., with Hangzhou zongzi being the most famous. \n\n3. Oil Fried Shrimp: This is one of the most representative seafood dishes in Zhejiang, made by stir-frying shrimp in hot oil until crispy and tender.\n\n4. Salt and Pepper Shredded Potato: This is a traditional Zhejiang vegetable dish, made by stir-frying shredded potato with salt and pepper, resulting in a crisp and refreshing taste.\n')]
"""
```

### Using CLI
```bash
# qwen
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen-7b-chat
# yi
CUDA_VISIBLE_DEVICES=0 swift infer --model_type yi-6b-chat
```

### Fine-tuned Models
If you want to perform inference using fine-tuned models, you can check out the [LLM Fine-tuning Documentation](LLM-fine-tuning.md#Fine-tuned-Model)


## Web-UI
### qwen-7b-chat
Using CLI:
```bash
CUDA_VISIBLE_DEVICES=0 swift app-ui --model_type qwen-7b-chat
```

Using python:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import AppUIArguments, ModelType, app_ui_main

app_ui_args = AppUIArguments(model_type=ModelType.qwen_7b_chat)
app_ui_main(app_ui_args)
```

Using bnb quantization:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import AppUIArguments, ModelType, app_ui_main

app_ui_args = AppUIArguments(model_type=ModelType.qwen_7b_chat, quantization_bit=4)
app_ui_main(app_ui_args)
```

### qwen-7b
Using CLI:
```bash
CUDA_VISIBLE_DEVICES=0 swift app-ui --model_type qwen-7b
```

Using python:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import AppUIArguments, ModelType, app_ui_main

app_ui_args = AppUIArguments(model_type=ModelType.qwen_7b)
app_ui_main(app_ui_args)
```

### Fine-tuned Models
To use the web-ui with fine-tuned models, you can check out the [LLM Fine-tuning Documentation](LLM-fine-tuning#fine-tuned-model)
