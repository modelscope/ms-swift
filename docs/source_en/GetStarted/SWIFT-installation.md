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


## Running Environment

|              | Range                | Recommended | Notes                                     |
| ------------ | -------------------- | ----------- | ----------------------------------------- |
| python       | >=3.9                | 3.10        |                                           |
| cuda         |                      | cuda12      | No need to install if using CPU, NPU, MPS |
| torch        | >=2.0                |             |                                           |
| transformers | >=4.33               | 4.48.3      |                                           |
| modelscope   | >=1.19               |             |                                           |
| peft         | >=0.11.0,<0.15.0     |             |                                           |
| trl          | >=0.13,<0.16         | 0.14.0      | RLHF                                      |
| vllm         | >=0.5.1              | 0.6.5       | Inference/Deployment/Evaluation           |
| lmdeploy     | lmdeploy>=0.5,<0.6.5 | 0.6.4       | Inference/Deployment/Evaluation           |
| deepspeed    |                      | 0.14.5      | Training                                  |

For more optional dependencies, you can refer to [here](https://github.com/modelscope/ms-swift/blob/main/requirements/install_all.sh).

## Notebook Environment

Most models that Swift supports for training can be used on A10 GPUs. Users can take advantage of the free GPU resources offered by ModelScope:

1. Visit the [ModelScope](https://www.modelscope.cn) official website and log in.
2. Click on `My Notebook` on the left and start a free GPU instance.
3. Enjoy utilizing the A10 GPU resources.
