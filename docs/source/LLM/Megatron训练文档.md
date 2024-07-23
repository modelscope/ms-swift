# Megatron训练文档

## 目录
- [环境准备](#环境准备)
- [自我认知微调案例](#自我认知微调案例)


## 环境准备

```shell
# 设置pip全局镜像 (加速下载)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 安装ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'

# 安装megatron相关依赖 (你不需要安装megatron-ml等其他依赖库)
# transformer_engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
# apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```


## 自我认知微调案例
这里介绍可以很快跑通的使用megatron训练的案例，通过此案例，你可以熟悉magatron训练的全流程。使用HF Trainer进行微调的对应案例可以查看[自我认知微调最佳实践](自我认知微调最佳实践.md). 

1. HF格式的权重转成megatron格式的权重:
```shell
# 默认输出路径: --megatron_output_dir {model_type}-tp{tp}-pp{pp}
CUDA_VISIBLE_DEVICES=0,1,2,3 swift export --model_type qwen2-7b-instruct --to_megatron true --tp 4
```

2. 使用megatron格式权重进行微调:
```
pip install llmuses==0.4.0
```

3. 将megatron格式权重重新转成HF格式:
```

```

4. 对获得的权重进行推理测试，并使用vLLM进行加速:
```

```


## megatron参数与SftArguments的映射关系
