# SWIFT安装

## Wheel包安装

可以使用pip进行安装：

```shell
pip install 'ms-swift'
# 使用评测
pip install 'ms-swift[eval]' -U
# 使用序列并行
pip install 'ms-swift[seq_parallel]' -U
# 全能力
pip install 'ms-swift[all]' -U
```

## 源代码安装

```shell
# pip install git+https://github.com/modelscope/ms-swift.git

git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .

# 全能力
# pip install -e '.[all]'
```

## 旧版本

SWIFT在3.0版本开始进行了不兼容式重构，如果需要使用2.x旧版本，请执行如下命令进行安装：
```shell
pip install ms-swift==2.*
```

## 镜像

镜像可以查看[这里](https://modelscope.cn/docs/intro/environment-setup#%E6%9C%80%E6%96%B0%E9%95%9C%E5%83%8F)

## 支持的硬件

| 硬件环境        | 备注                        |
| --------------- | --------------------------- |
| A10/A100/H100   |                             |
| RTX20/30/40系列 |                             |
| T4/V100         | 部分模型出现NAN             |
| Ascend NPU      | 部分模型出现NAN或算子不支持 |
| MPS             |                             |
| CPU             |                             |


## 运行环境

|        | 范围  | 推荐 | 备注 |
| ------ | ----- | ---- | --|
| python | >=3.9 | 3.10 ||
| cuda |  | cuda12 |使用cpu、npu、mps则无需安装|
| torch | >=2.0 |  ||
| transformers | >=4.33 | 4.48.3 ||
| modelscope | >=1.19 |  ||
| peft | >=0.11.0,<0.15.0 | ||
| trl | >=0.13,<0.16 | 0.14.0 |RLHF|
| vllm | >=0.5.1 | 0.6.5 |推理/部署/评测|
| lmdeploy | lmdeploy>=0.5,<0.6.5 | 0.6.4 |推理/部署/评测|
| deepspeed |  | 0.14.5 |训练|

更多可选依赖可以参考[这里](https://github.com/modelscope/ms-swift/blob/main/requirements/install_all.sh)。

## Notebook环境

Swift支持训练的绝大多数模型都可以在`A10`显卡上使用，用户可以使用ModelScope官方提供的免费显卡资源：

1. 进入[ModelScope](https://www.modelscope.cn)官方网站并登录
2. 点击左侧的`我的Notebook`并开启一个免费GPU实例
3. 愉快地薅A10显卡羊毛
