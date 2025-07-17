# Inference and Deployment

Below are the inference engines supported by Swift along with their corresponding capabilities. The three inference acceleration engines provide inference acceleration for Swift's inference, deployment, and evaluation modules:

| Inference Acceleration Engine                    | OpenAI API                                                   | Multimodal                                                   | Quantized Model | Multiple LoRAs                                               | QLoRA | Batch Inference                                              | Parallel Techniques |
| ------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------- | ------------------------------------------------------------ | ----- | ------------------------------------------------------------ | ------------------- |
| pytorch                                          | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/deploy/client/llm/chat/openai_client.py) | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/app/mllm.sh) | ✅               | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_lora.py) | ✅     | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/infer/pt/batch_ddp.sh) | DDP/device_map      |
| [vllm](https://github.com/vllm-project/vllm)     | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/infer/vllm/mllm_tp.sh) | ✅               | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/deploy/lora/server.sh) | ❌     | ✅                                                            | TP/PP/DP               |
| [sglang](https://github.com/sgl-project/sglang)    | ✅          | ❌ |      ✅        | ❌      | ❌     | ✅ | TP/PP/DP/EP |
| [lmdeploy](https://github.com/InternLM/lmdeploy) | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/infer/lmdeploy/mllm_tp.sh) | ✅               | ❌                                                            | ❌     | ✅                                                            | TP/DP                  |

## Inference

ms-swift uses a layered design philosophy, allowing users to perform inference through the command-line interface, web UI, or directly using Python.

To view the inference of a model fine-tuned with LoRA, please refer to the [Pre-training and Fine-tuning documentation](./Pre-training-and-Fine-tuning.md#inference-fine-tuned-model).

### Using CLI

**Full Parameter Model:**

```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --stream true \
    --infer_backend pt \
    --max_new_tokens 2048
```

**LoRA Model:**

```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapters swift/test_lora \
    --stream true \
    --infer_backend pt \
    --temperature 0 \
    --max_new_tokens 2048
```

**Command-Line Inference Instructions**

The above commands are for interactive command-line interface inference. After running the script, you can simply enter your query in the terminal. You can also input the following special commands:

- `multi-line`: Switch to multi-line mode, allowing line breaks in the input, ending with `#`.
- `single-line`: Switch to single-line mode, with line breaks indicating the end of input.
- `reset-system`: Reset the system and clear history.
- `clear`: Clear the history.
- `quit` or `exit`: Exit the conversation.

**Multimodal Model**

```shell
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift infer \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --stream true \
    --infer_backend pt \
    --max_new_tokens 2048
```

To perform inference with a multimodal model, you can add tags like `<image>`, `<video>`, or `<audio>` in your query (representing the location of image representations in `inputs_embeds`). For example, you can input `<image><image>What is the difference between these two images?` or `<video>Describe this video.` Then, follow the prompts to input the corresponding image/video/audio.


Here is an example of inference:
```
<<< <image><image>What is the difference between these two images?
Input an image path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
Input an image path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
The first image depicts a cute, cartoon-style kitten with large, expressive eyes and a fluffy white and gray coat. The background is simple, featuring a gradient of colors that highlight the kitten's face.

The second image shows a group of four cartoon-style sheep standing on a grassy field with mountains in the background. The sheep have fluffy white wool, black legs, and black faces with white markings around their eyes and noses. The background includes green hills and a blue sky with clouds, giving it a pastoral and serene atmosphere.
--------------------------------------------------
<<< clear
<<< <video>Describe this video.
Input a video path or URL <<< https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4
A baby wearing glasses is sitting on a bed and reading a book. The baby is holding the book with both hands and is looking down at it. The baby is wearing a light blue shirt and pink pants. The baby is sitting on a white pillow. The baby is looking at the book with interest. The baby is not moving much, just turning the pages of the book.
```

**Dataset Inference:**

```
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --stream true \
    --infer_backend pt \
    --val_dataset AI-ModelScope/alpaca-gpt4-data-zh \
    --max_new_tokens 2048
```

The above example provides streaming inference for both full parameters and LoRA, and below are more inference techniques available in SWIFT:

- Interface Inference: You can change `swift infer` to `swift app`.
- Batch Inference: For large models and multimodal models, you can specify `--max_batch_size` for batch inference by using `infer_backend=pt`. For specific details, refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/infer/pt/batch_ddp.sh). Note that you cannot set `--stream true` when performing batch inference.
- DDP/device_map Inference: `infer_backend=pt` supports parallel inference using DDP/device_map technology. For further details, refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/infer/pt/mllm_device_map.sh).
- Inference Acceleration: Swift supports using vllm/sglang/lmdeploy for inference acceleration across the inference, deployment, and evaluation modules by simply adding `--infer_backend vllm/sglang/lmdeploy`. You can refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/infer/vllm/ddp.sh).
- Multimodal Models: We provide shell scripts for multi-GPU inference for multimodal models using [pt](https://github.com/modelscope/ms-swift/blob/main/examples/infer/pt/mllm_device_map.sh), [vllm](https://github.com/modelscope/ms-swift/blob/main/examples/infer/vllm/mllm_tp.sh), and [lmdeploy](https://github.com/modelscope/ms-swift/blob/main/examples/infer/lmdeploy/mllm_tp.sh).
- Quantized Models: You can directly select models that are quantized with GPTQ, AWQ, or BNB, for example: `--model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4`.
- More Model Types: We also provide inference scripts for [bert](https://github.com/modelscope/ms-swift/blob/main/examples/infer/pt/bert.sh), [reward_model](https://github.com/modelscope/ms-swift/blob/main/examples/infer/pt/reward_model.sh), and [prm](https://github.com/modelscope/ms-swift/blob/main/examples/infer/pt/prm.sh).

**Tips:**

- SWIFT saves inference results, and you can specify the save path using `--result_path`.
- To output log probabilities, simply specify `--logprobs true` during inference. SWIFT will save these results. Note that setting `--stream true` will prevent storage of results.
- Using `infer_backend=pt` supports inference for all models supported by SWIFT, while `infer_backend=vllm/lmdeploy` supports only a subset of models. Please refer to the documentation for [vllm](https://docs.vllm.ai/en/latest/models/supported_models.html), [sglang](https://docs.sglang.ai/supported_models/generative_models.html) and [lmdeploy](https://lmdeploy.readthedocs.io/en/latest/supported_models/supported_models.html).
- If you encounter OOM when using `--infer_backend vllm`, you can lower `--max_model_len`, `--max_num_seqs`, choose an appropriate `--gpu_memory_utilization`, or set `--enforce_eager true`. Alternatively, you can address this by using tensor parallelism with `--tensor_parallel_size`.
- When inferring multimodal models using `--infer_backend vllm`, you need to input multiple images. You can set `--limit_mm_per_prompt` to resolve this, for example: `--limit_mm_per_prompt '{"image": 10, "video": 5}'`.
- If you encounter OOM issues while inferring qwen2-vl/qwen2.5-vl, you can address this by setting `MAX_PIXELS`, `VIDEO_MAX_PIXELS`, and `FPS_MAX_FRAMES`. For more information, refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/app/mllm.sh).
- SWIFT's built-in dialogue templates align with dialogue templates run using transformers. You can refer to [here](https://github.com/modelscope/ms-swift/blob/main/tests/test_align/test_template/test_vision.py) for testing. If there are any misalignments, please feel free to submit an issue or PR for correction.


### Using Web-UI

If you want to perform inference through a graphical interface, you can refer to the [Web-UI documentation](../GetStarted/Web-UI.md).

### Using Python

**Text Model:**

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import PtEngine, RequestConfig, InferRequest
model = 'Qwen/Qwen2.5-0.5B-Instruct'

# Load the inference engine
engine = PtEngine(model, max_batch_size=2)
request_config = RequestConfig(max_tokens=512, temperature=0)

# Using 2 infer_requests to demonstrate batch inference
infer_requests = [
    InferRequest(messages=[{'role': 'user', 'content': 'Who are you?'}]),
    InferRequest(messages=[{'role': 'user', 'content': 'Where is the capital of Zhejiang?'},
                           {'role': 'assistant', 'content': 'The capital of Zhejiang Province, China, is Hangzhou.'},
                           {'role': 'user', 'content': 'What are some fun places here?'}]),
]
resp_list = engine.infer(infer_requests, request_config)
query0 = infer_requests[0].messages[0]['content']
print(f'response0: {resp_list[0].choices[0].message.content}')
print(f'response1: {resp_list[1].choices[0].message.content}')
```

**Multimodal Model:**

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MAX_PIXELS'] = '1003520'
os.environ['VIDEO_MAX_PIXELS'] = '50176'
os.environ['FPS_MAX_FRAMES'] = '12'

from swift.llm import PtEngine, RequestConfig, InferRequest
model = 'Qwen/Qwen2.5-VL-3B-Instruct'

# Load the inference engine
engine = PtEngine(model, max_batch_size=2)
request_config = RequestConfig(max_tokens=512, temperature=0)

# Using 3 infer_requests to demonstrate batch inference
infer_requests = [
    InferRequest(messages=[{'role': 'user', 'content': 'Who are you?'}]),
    InferRequest(messages=[{'role': 'user', 'content': '<image><image> What is the difference between these two images?'}],
                 images=['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png',
                         'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']),
    InferRequest(messages=[{'role': 'user', 'content': '<video> Describe the video'}],
                 videos=['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']),
]
resp_list = engine.infer(infer_requests, request_config)
query0 = infer_requests[0].messages[0]['content']
print(f'response0: {resp_list[0].choices[0].message.content}')
print(f'response1: {resp_list[1].choices[0].message.content}')
print(f'response2: {resp_list[2].choices[0].message.content}')
```

We also provide more demos for Python-based inference:

- For streaming inference using `VllmEngine`, `SglangEngine` and `LmdeployEngine` for inference acceleration, you can refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo.py).
- Multimodal Inference: In addition to the aforementioned multimodal input formats, Swift is compatible with OpenAI's multimodal input format; refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_mllm.py).
- Grounding Tasks: For performing grounding tasks with multimodal models, you can refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_grounding.py).
- Multiple LoRA Inference: Refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_lora.py).
- Agent Inference: Refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_agent.py).
- Asynchronous Interface: For Python-based inference using `engine.infer_async`, refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo.py).


## Deployment

If you want to see the deployment of a model fine-tuned with LoRA, you can refer to the [Pre-training and Fine-tuning documentation](./Pre-training-and-Fine-tuning.md#deployment-fine-tuned-model).

This section primarily focuses on the deployment and invocation of multimodal models. For text-based large models, we provide a simple deployment and invocation example:

**Server Deployment:**

```shell
CUDA_VISIBLE_DEVICES=0 swift deploy \
    --model Qwen/Qwen2.5-7B-Instruct \
    --infer_backend vllm \
    --max_new_tokens 2048 \
    --served_model_name Qwen2.5-7B-Instruct
```

**Client Invocation Test:**

```shell
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "Qwen2.5-7B-Instruct",
"messages": [{"role": "user", "content": "What should I do if I can’t sleep at night?"}],
"max_tokens": 256,
"temperature": 0
}'
```


### Server Side

```shell
# test env: pip install transformers==4.51.3 vllm==0.8.5.post1
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift deploy \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --infer_backend vllm \
    --gpu_memory_utilization 0.9 \
    --max_model_len 8192 \
    --max_new_tokens 2048 \
    --limit_mm_per_prompt '{"image": 5, "video": 2}' \
    --served_model_name Qwen2.5-VL-3B-Instruct
```

### Client Side

We introduce three methods for invoking the client: using curl, the OpenAI library, and the Swift client.

**Method 1: curl**

```shell
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "Qwen2.5-VL-3B-Instruct",
"messages": [{"role": "user", "content": [
    {"type": "image", "image": "http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png"},
    {"type": "image", "image": "http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png"},
    {"type": "text", "text": "What is the difference between these two images?"}
]}],
"max_tokens": 256,
"temperature": 0
}'
```

**Method 2: OpenAI Library**

```python
from openai import OpenAI

client = OpenAI(
    api_key='EMPTY',
    base_url=f'http://127.0.0.1:8000/v1',
)
model = client.models.list().data[0].id
print(f'model: {model}')

messages = [{'role': 'user', 'content': [
    {'type': 'video', 'video': 'https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4'},
    {'type': 'text', 'text': 'describe the video'}
]}]

resp = client.chat.completions.create(model=model, messages=messages, max_tokens=512, temperature=0)
query = messages[0]['content']
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

# Using base64
import base64
import requests
resp = requests.get('https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4')
base64_encoded = base64.b64encode(resp.content).decode('utf-8')
messages = [{'role': 'user', 'content': [
    {'type': 'video', 'video': f'data:video/mp4;base64,{base64_encoded}'},
    {'type': 'text', 'text': 'describe the video'}
]}]

gen = client.chat.completions.create(model=model, messages=messages, stream=True, temperature=0)
print(f'query: {query}\nresponse: ', end='')
for chunk in gen:
    if chunk is None:
        continue
    print(chunk.choices[0].delta.content, end='', flush=True)
print()
```

**Method 3: Swift Client**

```python
from swift.llm import InferRequest, InferClient, RequestConfig
from swift.plugin import InferStats

engine = InferClient(host='127.0.0.1', port=8000)
print(f'models: {engine.models}')
metric = InferStats()
request_config = RequestConfig(max_tokens=512, temperature=0)

# Using 3 infer_requests to demonstrate batch inference
# Supports local paths, base64, and URLs
infer_requests = [
    InferRequest(messages=[{'role': 'user', 'content': 'Who are you?'}]),
    InferRequest(messages=[{'role': 'user', 'content': '<image><image> What is the difference between these two images?'}],
                 images=['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png',
                         'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']),
    InferRequest(messages=[{'role': 'user', 'content': '<video> Describe the video'}],
                 videos=['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']),
]

resp_list = engine.infer(infer_requests, request_config, metrics=[metric])
print(f'response0: {resp_list[0].choices[0].message.content}')
print(f'response1: {resp_list[1].choices[0].message.content}')
print(f'response2: {resp_list[2].choices[0].message.content}')
print(metric.compute())
metric.reset()

# Using base64
import base64
import requests
resp = requests.get('https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4')
base64_encoded = base64.b64encode(resp.content).decode('utf-8')
messages = [{'role': 'user', 'content': [
    {'type': 'video', 'video': f'data:video/mp4;base64,{base64_encoded}'},
    {'type': 'text', 'text': 'describe the video'}
]}]
infer_request = InferRequest(messages=messages)
request_config = RequestConfig(max_tokens=512, temperature=0, stream=True)
gen_list = engine.infer([infer_request], request_config, metrics=[metric])
print(f'response0: ', end='')
for chunk in gen_list[0]:
    if chunk is None:
        continue
    print(chunk.choices[0].delta.content, end='', flush=True)
print()
print(metric.compute())
```

We also provide more deployment demos:

- Multiple LoRA deployment and invocation: Refer to [this link](https://github.com/modelscope/ms-swift/tree/main/examples/deploy/lora).
- Deployment and invocation of the Base model: Refer to [this link](https://github.com/modelscope/ms-swift/tree/main/examples/deploy/client/llm/base).
- More model types: We provide deployment scripts for [bert](https://github.com/modelscope/ms-swift/tree/main/examples/deploy/bert) and [reward_model](https://github.com/modelscope/ms-swift/tree/main/examples/deploy/reward_model).
