# Megatron使用文档

## 目录
- [环境准备](#环境准备)
- [预训练](#预训练)
- [微调](#微调)


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


##
