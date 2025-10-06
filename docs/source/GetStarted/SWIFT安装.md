# SWIFT安装

## Wheel包安装

可以使用pip进行安装：

```shell
# 推荐
pip install 'ms-swift'
# 使用评测
pip install 'ms-swift[eval]' -U
# 全能力
pip install 'ms-swift[all]' -U
```

## 源代码安装

```shell
# pip install git+https://github.com/modelscope/ms-swift.git

# 全能力
# pip install "git+https://github.com/modelscope/ms-swift.git#egg=ms-swift[all]"

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

docker可以查看[这里](https://github.com/modelscope/modelscope/blob/master/docker/build_image.py#L345)。
```
# swift3.8.3
modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.1.1-modelscope1.29.2-swift3.8.3
modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.1.1-modelscope1.29.2-swift3.8.3
modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.1.1-modelscope1.29.2-swift3.8.3

# swift3.7.2
modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.0-modelscope1.28.2-swift3.7.2
modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.0-modelscope1.28.2-swift3.7.2
modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.0-modelscope1.28.2-swift3.7.2

# swift3.6.4
modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py310-torch2.6.0-vllm0.8.5.post1-modelscope1.28.1-swift3.6.4
modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py310-torch2.6.0-vllm0.8.5.post1-modelscope1.28.1-swift3.6.4
modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py310-torch2.6.0-vllm0.8.5.post1-modelscope1.28.1-swift3.6.4
```

<details><summary>历史镜像</summary>

```
# swift3.5.3
modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py310-torch2.6.0-vllm0.8.5.post1-modelscope1.27.1-swift3.5.3
modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py310-torch2.6.0-vllm0.8.5.post1-modelscope1.27.1-swift3.5.3
modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py310-torch2.6.0-vllm0.8.5.post1-modelscope1.27.1-swift3.5.3

# swift3.4.1.post1
modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py311-torch2.6.0-vllm0.8.5.post1-modelscope1.26.0-swift3.4.1.post1
modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py311-torch2.6.0-vllm0.8.5.post1-modelscope1.26.0-swift3.4.1.post1
modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py311-torch2.6.0-vllm0.8.5.post1-modelscope1.26.0-swift3.4.1.post1

# swift3.3.0.post1
modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py311-torch2.6.0-vllm0.8.3-modelscope1.25.0-swift3.3.0.post1
modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py311-torch2.6.0-vllm0.8.3-modelscope1.25.0-swift3.3.0.post1

# swift3.2.2
modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py311-torch2.5.1-modelscope1.25.0-swift3.2.2
```
</details>

更多镜像可以查看[这里](https://modelscope.cn/docs/intro/environment-setup#%E6%9C%80%E6%96%B0%E9%95%9C%E5%83%8F)。

## 支持的硬件

| 硬件环境        | 备注                        |
| --------------- | --------------------------- |
| A10/A100/H100   |                             |
| RTX20/30/40系列 |                             |
| T4/V100         | 部分模型出现NAN             |
| Ascend NPU      | 部分模型出现NAN或算子不支持 |
| MPS             | 参考[issue 4572](https://github.com/modelscope/ms-swift/issues/4572)                         |
| CPU             |                             |


## 运行环境

|              | 范围           | 推荐                  | 备注                 |
|--------------|--------------|---------------------|--------------------|
| python       | >=3.9        | 3.10/3.11                |                    |
| cuda         |              | cuda12              | 使用cpu、npu、mps则无需安装 |
| torch        | >=2.0        | 2.7.1               |                    |
| transformers | >=4.33       | 4.56.2              |                    |
| modelscope   | >=1.23       |                     |                    |
| peft         | >=0.11,<0.18 |                     |                    |
| flash_attn   |              | 2.7.4.post1/3.0.0b1 |                    |
| trl          | >=0.15,<0.21 | 0.20.0              | RLHF               |
| deepspeed    | >=0.14       | 0.17.5              | 训练                 |
| vllm         | >=0.5.1      | 0.10.1.1                | 推理/部署              |
| sglang       | >=0.4.6      | 0.4.10.post2         | 推理/部署              |
| lmdeploy     | >=0.5   | 0.9.2.post1                 | 推理/部署              |
| evalscope    | >=1.0       |                     | 评测                 |
| gradio       |              | 5.32.1              | Web-UI/App         |

更多可选依赖可以参考[这里](https://github.com/modelscope/ms-swift/blob/main/requirements/install_all.sh)。

## Notebook环境

Swift支持训练的绝大多数模型都可以在`A10`显卡上使用，用户可以使用ModelScope官方提供的免费显卡资源：

1. 进入[ModelScope](https://www.modelscope.cn)官方网站并登录。
2. 点击左侧的`我的Notebook`并开启一个免费GPU实例。
3. 愉快地薅A10显卡羊毛。
