# SWIFT Installation

## Wheel Packages Installation

You can install it using pip:

```shell
pip install 'ms-swift'
# For evaluation usage
pip install 'ms-swift[eval]' -U
# For sequence parallel usage
pip install 'ms-swift[seq_parallel]' -U
# Full capabilities
pip install 'ms-swift[all]' -U
```

## Source Code Installation

```shell
# pip install git+https://github.com/modelscope/ms-swift.git

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

You can view the image [here](https://modelscope.cn/docs/intro/environment-setup#%E6%9C%80%E6%96%B0%E9%95%9C%E5%83%8F).

## Supported Hardware

| Hardware Environment | Remarks                                                |
| -------------------- | ------------------------------------------------------ |
| A10/A100/H100        |                                                        |
| RTX 20/30/40 Series  |                                                        |
| T4/V100              | Some models may encounter NAN                          |
| Ascend NPU           | Some models may encounter NAN or unsupported operators |
| MPS                  |                                                        |
| CPU                  |                                                        |


## Running Environment

|              | Range                | Recommended | Notes                                     |
| ------------ | -------------------- | ----------- | ----------------------------------------- |
| python       | >=3.9                | 3.10        |                                           |
| cuda         |                      | cuda12      | No need to install if using CPU, NPU, MPS |
| torch        | >=2.0                |             |                                           |
| transformers | >=4.33               | 4.49      |                                           |
| modelscope   | >=1.19               |             |                                           |
| peft         | >=0.11,<0.15     |             |                                           |
| trl          | >=0.13,<0.17         | 0.15      | RLHF                                      |
| deepspeed    | >=0.14 | 0.14.5 | Training                                  |
| vllm         | >=0.5.1              | 0.7.3       | Inference/Deployment/Evaluation           |
| lmdeploy     | lmdeploy>=0.5 | 0.7.0.post3       | Inference/Deployment/Evaluation           |
| evalscope | | >=0.11 | Evaluation |

For more optional dependencies, you can refer to [here](https://github.com/modelscope/ms-swift/blob/main/requirements/install_all.sh).

## Notebook Environment

Most models that Swift supports for training can be used on A10 GPUs. Users can take advantage of the free GPU resources offered by ModelScope:

1. Visit the [ModelScope](https://www.modelscope.cn) official website and log in.
2. Click on `My Notebook` on the left and start a free GPU instance.
3. Enjoy utilizing the A10 GPU resources.
