# LmDeploy Inference Acceleration and Deployment
lmdeploy github: [https://github.com/InternLM/lmdeploy](https://github.com/InternLM/lmdeploy).

Models that support inference acceleration using lmdeploy can be found at [Supported Models](Supported-models-datasets.md#LLM).

## Table of Contents
- [Environment Preparation](#environment-preparation)
- [Inference Acceleration](#inference-acceleration)
- [Deployment](#deployment)
- [Multimodal](#multimodal)

## Environment Preparation
GPU devices: A10, 3090, V100, A100 are all supported.
```bash
# Set pip global mirror (speeds up downloads)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# Install ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'

pip install lmdeploy
```

## Inference Acceleration

### Using Python

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    ModelType, get_lmdeploy_engine, get_default_template_type,
    get_template, inference_lmdeploy, inference_stream_lmdeploy
)

model_type = ModelType.qwen_7b_chat
lmdeploy_engine = get_lmdeploy_engine(model_type)
template_type = get_default_template_type(model_type)
template = get_template(template_type, lmdeploy_engine.hf_tokenizer)
# Similar to `transformers.GenerationConfig` interface
lmdeploy_engine.generation_config.max_new_tokens = 256
generation_info = {}

request_list = [{'query': 'Hello!'}, {'query': 'Where is the capital of Zhejiang?'}]
resp_list = inference_lmdeploy(lmdeploy_engine, template, request_list, generation_info=generation_info)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")
print(generation_info)

# stream
history1 = resp_list[1]['history']
request_list = [{'query': 'Is there anything tasty here?', 'history': history1}]
gen = inference_stream_lmdeploy(lmdeploy_engine, template, request_list, generation_info=generation_info)
query = request_list[0]['query']
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
for resp_list in gen:
    resp = resp_list[0]
    response = resp['response']
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print()

history = resp_list[0]['history']
print(f'history: {history}')
print(generation_info)
"""
query: Hello!
response: Hello there! How can I help you today?
query: Where is the capital of Zhejiang?
response: The capital of Zhejiang is Hangzhou. It is located in southeastern China, along the lower reaches of the Qiantang River (also known as the West Lake), and is one of the most prosperous cities in the country. Hangzhou is famous for its natural beauty, cultural heritage, and economic development, with a rich history dating back over 2,000 years. The city is home to many historic landmarks and attractions, including the West Lake, Lingyin Temple, and the Longjing Tea Plantations. Additionally, Hangzhou is a major center for technology, finance, and transportation in China.
{'num_prompt_tokens': 49, 'num_generated_tokens': 135, 'num_samples': 2, 'runtime': 1.5066149180056527, 'samples/s': 1.3274792225258558, 'tokens/s': 89.60484752049527}
query: Is there anything tasty here?
response: Yes, Hangzhou is known for its delicious cuisine! The city has a long history of culinary arts and is considered to be one of the birthplaces of Chinese cuisine. Some of the most popular dishes from Hangzhou include:

  * Dongpo Pork: A dish made with pork belly that has been braised in a soy sauce-based broth until it is tender and flavorful.
  * West Lake Fish in Vinegar Gravy: A dish made with freshwater fish that has been simmered in a tangy vinegar sauce.
  * Longjing Tea Soup: A soup made with Dragon Well tea leaves and chicken or pork, often served as a light meal or appetizer.
  * Xiao Long Bao: Small steamed dumplings filled with meat or vegetables and served with a savory broth.

In addition to these classic dishes, Hangzhou also has a thriving street food scene, with vendors selling everything from steamed buns to grilled meats and seafood. So if you're a foodie, you'll definitely want to try some of the local specialties while you're in Hangzhou!
history: [['Where is the capital of Zhejiang?', 'The capital of Zhejiang is Hangzhou. It is located in southeastern China, along the lower reaches of the Qiantang River (also known as the West Lake), and is one of the most prosperous cities in the country. Hangzhou is famous for its natural beauty, cultural heritage, and economic development, with a rich history dating back over 2,000 years. The city is home to many historic landmarks and attractions, including the West Lake, Lingyin Temple, and the Longjing Tea Plantations. Additionally, Hangzhou is a major center for technology, finance, and transportation in China.'], ['Is there anything tasty here?', "Yes, Hangzhou is known for its delicious cuisine! The city has a long history of culinary arts and is considered to be one of the birthplaces of Chinese cuisine. Some of the most popular dishes from Hangzhou include:\n\n  * Dongpo Pork: A dish made with pork belly that has been braised in a soy sauce-based broth until it is tender and flavorful.\n  * West Lake Fish in Vinegar Gravy: A dish made with freshwater fish that has been simmered in a tangy vinegar sauce.\n  * Longjing Tea Soup: A soup made with Dragon Well tea leaves and chicken or pork, often served as a light meal or appetizer.\n  * Xiao Long Bao: Small steamed dumplings filled with meat or vegetables and served with a savory broth.\n\nIn addition to these classic dishes, Hangzhou also has a thriving street food scene, with vendors selling everything from steamed buns to grilled meats and seafood. So if you're a foodie, you'll definitely want to try some of the local specialties while you're in Hangzhou!"]]
{'num_prompt_tokens': 169, 'num_generated_tokens': 216, 'num_samples': 1, 'runtime': 2.4760487159946933, 'samples/s': 0.4038692750834161, 'tokens/s': 87.23576341801788}
"""
```

### Using CLI
```bash
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen2-7b-instruct --infer_backend lmdeploy
# TP
CUDA_VISIBLE_DEVICES=0,1 swift infer --model_type qwen2-7b-instruct --infer_backend lmdeploy --tp 2

CUDA_VISIBLE_DEVICES=0,1 swift infer --model_type qwen2-72b-instruct --infer_backend lmdeploy --tp 2
```

## Deployment
```bash
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen2-7b-instruct --infer_backend lmdeploy
# TP
CUDA_VISIBLE_DEVICES=0,1 swift deploy --model_type qwen2-7b-instruct --infer_backend lmdeploy --tp 2

CUDA_VISIBLE_DEVICES=0,1 swift deploy --model_type qwen2-72b-instruct --infer_backend lmdeploy --tp 2
```

The method for client invocation can be found in: [vLLM Inference Acceleration and Deployment Documentation](VLLM-inference-acceleration-and-deployment.md#deployment).

## Multimodal
Check [here](../Multi-Modal/LmDeploy-inference-acceleration-and-deployment.md)
