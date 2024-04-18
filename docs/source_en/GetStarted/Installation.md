# Installation and Usage

## Wheel Package Installation

You can use pip to install:

```shell
# Full capabilities
pip install 'ms-swift[all]' -U
# Only use LLM
pip install 'ms-swift[llm]' -U
# Only use AIGC
pip install 'ms-swift[aigc]' -U
# Only use adapters
pip install ms-swift -U
```

## Source Code Installation

```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[all]'
```

## Notebook Environment

Most of the models supported by Swift for training can be used on `A10` GPUs. Users can use the free GPU resources officially provided by ModelScope:

1. Go to the official [ModelScope](https://www.modelscope.cn) website and log in
2. Click on `My Notebook` on the left and start a free GPU instance
3. Happily take advantage of the A10 GPU resources

## Build Documentation

Swift supports complete API Doc documentation. Execute the following command in the swift root directory:

```shell
make docs
```

After the execution is complete, view `docs/build/html/index.html`.
