# 推理及部署

训练后的模型会用于推理或者部署。推理即使用模型用输入获得输出的过程，部署是将模型运行到恒定运行的环境中推理的过程。一般来说，LLM的推理可以直接使用PyTorch代码、使用VLLM/XInference/FastChat等框架，也可以使用llama.cpp/chatglm.cpp/qwen.cpp等c++推理框架。

# KVCache

上面我们讲过，自回归模型的推理是将新的token不断填入序列生成下一个token的过程。那么，前面token已经生成的中间计算结果是可以直接利用的。具体以Attention结构来说：

<img src="resources/image-20240116161847987.png" alt="image-20240116161847987" style="zoom:33%;" />

推理时的Q是单token tensor，但K和V都是包含了所有历史token tensor的长序列，因此KV是可以使用前序计算的中间结果的，这部分的缓存就是KVCache，其显存占用非常巨大。

# VLLM推理

VLLM支持绝大多数LLM模型的推理加速。它使用如下的方案大幅提升推理速度：

1. Continuous batching

   - 在实际推理过程中，一个批次多个句子的输入的token长度可能相差很大，最后生成的模型输出token长度相差也很大。在python朴素推理中，最短的序列会等待最长序列生成完成后一并返回，这意味着本来可以处理更多token的GPU算力在对齐过程中产生了浪费。continous batching的方式就是在每个句子序列输出结束后马上填充下一个句子的token，做到高效利用算力。

     ![image-20240116160416701](resources/image-20240116160416701.png)

     ![image-20240116160444612](resources/image-20240116160444612.png)

2. PagedAttention
   - 推理时的显存占用中，KVCache的碎片化和重复记录浪费了50%以上的显存。VLLM将现有输入token进行物理分块，使每块显存内部包含了固定长度的tokens。在进行Attention操作时，VLLM会从物理块中取出KVCache并计算。因此模型看到的逻辑块是连续的，但是物理块的地址可能并不连续。这和虚拟内存的思想非常相似。另外对于同一个句子生成多个回答的情况，VLLM会将不同的逻辑块映射为一个物理块，起到节省显存提高吞吐的作用。

![image-20240116162157881](resources/image-20240116162157881.png)

![image-20240116162213204](resources/image-20240116162213204.png)

值得注意的是，VLLM会默认将显卡的全部显存预先申请以提高缓存大小和推理速度，用户可以通过参数`gpu_memory_utilization`控制缓存大小。

用VLLM部署模型：

```shell
pip install vllm
VLLM_USE_MODELSCOPE=True python -m vllm.entrypoints.openai.api_server --model qwen/Qwen-1_8B-Chat --trust-remote-code
```

之后就可以调用服务：

```shell
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "qwen/Qwen-1_8B-Chat",
"prompt": "San Francisco is a",
"max_tokens": 7,
"temperature": 0
}'
```

# llama.cpp

llama.cpp是使用c++语言编写的对llama系列模型进行高效推理或量化推理的开源库。该库使用了ggml底层计算库进行推理。在使用之前需要额外将python的weights转为ggml格式或gguf格式方可使用。和llama.cpp类似，还有兼容ChatGLM模型的chatglm.cpp和兼容qwen模型的qwen.cpp和mistral的mistral.cpp。

```python
git clone --recursive https://github.com/QwenLM/qwen.cpp && cd qwen.cpp
cmake -B build
cmake --build build -j --config Release
```

将原始模型转换为ggml支持的格式：

```shell
python3 qwen_cpp/convert.py -i Qwen/Qwen-7B-Chat -t q4_0 -o qwen7b-ggml.bin
./build/bin/main -m qwen7b-ggml.bin --tiktoken Qwen-7B-Chat/qwen.tiktoken -p 你好
# 你好！很高兴为你提供帮助。
```

量化章节中我们介绍，GGML库适合于CPU运行，因此推荐用户在CPU环境中或边缘计算中考虑cpp库进行推理。

# FastChat

FastChat是一个开源推理库，侧重于模型的分布式部署实现，并提供了OpenAI样式的RESTFul API。

```shell
pip3 install "fschat[model_worker,webui]"
python3 -m fastchat.serve.controller
```

在新的terminal中启动：

```shell
python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5
```

之后在新的terminal中可以运行界面进行推理:

```shell
python3 -m fastchat.serve.gradio_web_server
```

# SWIFT

在魔搭官方的SWIFT库中，我们也提供了一个简易的部署命令，供用户在训练完成后进行模型验证和单实例DEMO。

```shell
pip install ms-swift -U
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen-1_8b-chat
```
