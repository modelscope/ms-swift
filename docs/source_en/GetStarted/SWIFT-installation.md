# SWIFT Installation

## Wheel Packages Installation

You can install it using pip:

```shell
# recommend
pip install 'ms-swift'
# For evaluation usage
pip install 'ms-swift[eval]' -U
# Full capabilities
pip install 'ms-swift[all]' -U
```

## Source Code Installation

```shell
# pip install git+https://github.com/modelscope/ms-swift.git

# Full capabilities
# pip install "git+https://github.com/modelscope/ms-swift.git#egg=ms-swift[all]"

git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .

# Full capabilities
# pip install -e '.[all]'
```

## Older Versions

SWIFT underwent an incompatible restructuring starting from version 3.0. If you need to use the old version 2.x, please execute the following command to install:

```shell
pip install ms-swift==2.*
```

## Mirror

You can check Docker [here](https://github.com/modelscope/modelscope/blob/master/docker/build_image.py#L345).
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

<details><summary>Historical Mirrors</summary>

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

More images can be found [here](https://modelscope.cn/docs/intro/environment-setup#%E6%9C%80%E6%96%B0%E9%95%9C%E5%83%8F).

## Supported Hardware

| Hardware Environment | Remarks                                                |
| -------------------- | ------------------------------------------------------ |
| A10/A100/H100        |                                                        |
| RTX 20/30/40 Series  |                                                        |
| T4/V100              | Some models may encounter NAN                          |
| Ascend NPU           | Some models may encounter NAN or unsupported operators |
| MPS                  |   Refer to [issue 4572](https://github.com/modelscope/ms-swift/issues/4572)                         |
| CPU                  |                                                        |


## Running Environment

|              | Range        | Recommended         | Notes                                     |
|--------------|--------------|---------------------|-------------------------------------------|
| python       | >=3.9        | 3.10/3.11                |                                           |
| cuda         |              | cuda12              | No need to install if using CPU, NPU, MPS |
| torch        | >=2.0        | 2.7.1               |                                           |
| transformers | >=4.33       | 4.56.2              |                                           |
| modelscope   | >=1.23       |                     |                                           |
| peft         | >=0.11,<0.18 |                     |                                           |
| flash_attn   |              | 2.7.4.post1/3.0.0b1 |                                           |
| trl          | >=0.15,<0.21 | 0.20.0              | RLHF                                      |
| deepspeed    | >=0.14       | 0.17.5              | Training                                  |
| vllm         | >=0.5.1      | 0.10.1.1                | Inference/Deployment                      |
| sglang       | >=0.4.6      | 0.4.10.post2         | Inference/Deployment                      |
| lmdeploy     | >=0.5   | 0.9.2.post1                 | Inference/Deployment                      |
| evalscope    | >=1.0       |                     | Evaluation                                |
| gradio       |              | 5.32.1              | Web-UI/App                                |


For more optional dependencies, you can refer to [here](https://github.com/modelscope/ms-swift/blob/main/requirements/install_all.sh).

## Notebook Environment

Most models that Swift supports for training can be used on A10 GPUs. Users can take advantage of the free GPU resources offered by ModelScope:

1. Visit the [ModelScope](https://www.modelscope.cn) official website and log in.
2. Click on `My Notebook` on the left and start a free GPU instance.
3. Enjoy utilizing the A10 GPU resources.
