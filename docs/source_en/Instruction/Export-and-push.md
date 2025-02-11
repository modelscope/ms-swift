# Export
Swift supports quantization of models using technologies like awq, gptq, bnb, hqq, and eetq. The awq and gptq quantization methods are compatible with vllm/lmdeploy for inference acceleration. They require a calibration dataset, which improves quantization performance, but slows down the quantization speed. In contrast, bnb, hqq, and eetq do not require calibration data, allowing for faster quantization. All five quantization methods support qlora fine-tuning.

awq, gptq, and bnb (8bit) can use `swift export` for quantization. Meanwhile, bnb, hqq, and eetq allow for quick quantization directly during sft and inference.

## Environment Setup

In addition to installing SWIFT, you need to install the following dependencies:
```bash
# For awq quantization:
# The versions of autoawq and cuda are related, please choose according to `https://github.com/casper-hansen/AutoAWQ`
# If there is a torch dependency conflict, please add the `--no-deps` option.
pip install autoawq -U

# For gptq quantization:
# The versions of auto_gptq and cuda are related, please choose according to `https://github.com/PanQiWei/AutoGPTQ#quick-installation`
pip install auto_gptq optimum -U

# For bnb quantization:
pip install bitsandbytes -U

# For hqq quantization:
# pip install transformers>=4.41
pip install hqq

# For eetq quantization:
# pip install transformers>=4.41
# Reference https://github.com/NetEase-FuXi/EETQ
git clone https://github.com/NetEase-FuXi/EETQ.git
cd EETQ/
git submodule update --init --recursive
pip install .
```

## Quantization Command

Please refer to the [examples](https://github.com/modelscope/ms-swift/tree/main/examples/export) for quantization commands.

To perform model quantization after training with a training set, use the `--model` and `--adapters` options to specify the checkpoint directory. The checkpoint directory contains the parameter files from training:

```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --adapters 'output/some-model/vx-xxx/checkpoint-xxx' \
    --quant_bits 4 \
    --load_data_args true \
    --quant_method gptq
```

Once quantized, the model can be used for inference or deployment directly with `--model`, for example:
```shell
swift infer --model /xxx/quantize-output-folder
swift deploy --model /xxx/quantize-output-folder
```

### bnb, hqq, eetq

For bnb, hqq, and eetq, you can use `swift infer` for quick quantization and training inference:
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

**Note**
- hqq supports more custom parameters, such as specifying different quantization configurations for various network layers. Please refer to [command line parameters](Commend-line-parameters.md).
- eetq quantization is 8bit quantization; you do not need to specify quantization bits. Currently, bf16 is not supported; set dtype to fp16.
- Currently, qlora speed is relatively slow for eetq. It is recommended to use hqq instead. See [issue](https://github.com/NetEase-FuXi/EETQ/issues/17).
