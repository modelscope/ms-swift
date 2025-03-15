
# Megatron-SWIFT训练



## 环境准备
使用Megatron-SWIFT，除了安装swift依赖外，还需要安装以下内容：

```shell
pip install pybind11
# transformer_engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

# apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```


## 快速入门案例



- 更多案例可以查看[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/megatron)
