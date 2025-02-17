# 导出
swift支持使用awq、gptq、bnb、hqq、eetq技术对模型进行量化。其中awq、gptq量化技术支持vllm/lmdeploy进行推理加速，需要使用校准数据集，量化性能更好，但量化速度较慢。而bnb、hqq、eetq无需校准数据，量化速度较快。这五种量化方法都支持qlora微调。

awq、gptq、bnb（8bit）支持使用`swift export`进行量化。而bnb、hqq、eetq可以直接在sft和infer时进行快速量化。

## 环境准备

除SWIFT安装外，需要安装以下额外依赖：
```bash
# 使用awq量化:
# autoawq和cuda版本有对应关系，请按照`https://github.com/casper-hansen/AutoAWQ`选择版本
# 如果出现torch依赖冲突，请额外增加指令`--no-deps`
pip install autoawq -U

# 使用gptq量化:
# auto_gptq和cuda版本有对应关系，请按照`https://github.com/PanQiWei/AutoGPTQ#quick-installation`选择版本
pip install auto_gptq optimum -U

# 使用bnb量化：
pip install bitsandbytes -U

# 使用hqq量化：
# pip install transformers>=4.41
pip install hqq

# 使用eetq量化：
# pip install transformers>=4.41
# 参考https://github.com/NetEase-FuXi/EETQ
git clone https://github.com/NetEase-FuXi/EETQ.git
cd EETQ/
git submodule update --init --recursive
pip install .
```

## 量化命令

量化命令请参考[examples](https://github.com/modelscope/ms-swift/tree/main/examples/export)

使用训练集进行训练后的模型量化，使用`--model`, `--adapters`来指定训练的checkpoint文件夹，checkpoint文件夹中包含训练的参数文件：

```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --adapters 'output/some-model/vx-xxx/checkpoint-xxx' \
    --quant_bits 4 \
    --load_data_args true \
    --quant_method gptq
```

量化后模型可以直接以`--model`来进行推理或部署，例如：
```shell
swift infer --model /xxx/quantize-output-folder
swift deploy --model /xxx/quantize-output-folder
```

### bnb、hqq、eetq
对于bnb、hqq、eetq，我们只需要使用swift infer来进行快速量化并训练推理
```bash
# Infer
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model Qwen/Qwen2-7B-Instruct \
    --quant_method bnb \
    --quant_bits 4

CUDA_VISIBLE_DEVICES=0 swift infer \
    --model Qwen/Qwen2-7B-Instruct \
    --quant_method hqq \
    --quant_bits 4

CUDA_VISIBLE_DEVICES=0 swift infer \
    --model Qwen/Qwen2-7B-Instruct \
    --quant_method eetq \
    --quant_bits 8 \
    --torch_dtype float16

# Train
# bnb
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model Qwen/Qwen2-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#5000' \
    --quant_method bnb \
    --quant_bits 4 \
    --torch_dtype bfloat16

# hqq
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model Qwen/Qwen2-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#5000' \
    --quant_method hqq \
    --quant_bits 4

# eetq
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model Qwen/Qwen2-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#5000' \
    --quant_method eetq \
    --torch_dtype float16
```

**注意**
- hqq支持更多自定义参数，比如为不同网络层指定不同量化配置，具体请见[命令行参数](命令行参数.md)
- eetq量化为8bit量化，无需指定quantization_bit。目前不支持bf16，需要指定dtype为fp16
- eetq目前qlora速度比较慢，推荐使用hqq。参考[issue](https://github.com/NetEase-FuXi/EETQ/issues/17)
