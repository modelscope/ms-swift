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

You can also use [install_all.sh](https://github.com/modelscope/ms-swift/blob/main/requirements/install_all.sh) for installation.

## Notebook Environment

Most models that Swift supports for training can be used on A10 GPUs. Users can take advantage of the free GPU resources offered by ModelScope:

1. Visit the [ModelScope](https://www.modelscope.cn) official website and log in.
2. Click on `My Notebook` on the left and start a free GPU instance.
3. Enjoy utilizing the A10 GPU resources.
