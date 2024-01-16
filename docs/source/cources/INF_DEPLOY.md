# 推理及部署

训练后的模型会用于推理或者部署。推理即使用模型用输入获得输出的过程，部署是将模型运行到恒定运行的环境中推理的过程。一般来说，LLM的推理可以直接使用PyTorch代码、使用VLLM/XInference/FastChat等框架，也可以使用llama.cpp/chatglm.cpp/qwen.cpp等c++推理框架。

# PyTorch推理

PyTorch推理是直接使用原生代码进行推理的方式。

# VLLM推理

VLLM支持绝大多数LLM模型的推理加速。它使用如下的方案大幅提升推理速度：

1. Continuous batching