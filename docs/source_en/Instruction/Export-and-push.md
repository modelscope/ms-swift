# Export and Push

## Merge LoRA

- See [here](https://github.com/modelscope/ms-swift/blob/main/examples/export/merge_lora.sh).

## Quantization

SWIFT supports quantization exports for AWQ, GPTQ, FP8, and BNB models. AWQ and GPTQ require a calibration dataset, which yields better quantization performance but takes longer to quantize. On the other hand, FP8 and BNB does not require a calibration dataset and is quicker to quantize.

| Quantization Technique | Multimodal | Inference Acceleration | Continued Training |
| ---------------------- | ---------- | ---------------------- | ------------------ |
| FP8     | ✅      | ✅        | ✅        |
| GPTQ           | ✅          | ✅                      | ✅                  |
| AWQ                    | Partial     | ✅                      | ✅                  |
| BNB                    | ❌          | ✅                      | ✅                  |

In addition to the SWIFT installation, the following additional dependencies need to be installed:

```shell
# For AWQ quantization (from the source checkout):
pip install -e '.[awq]'

# Once a published ms-swift release provides this extra:
# pip install 'ms-swift[awq]'

# For GPTQ quantization:
pip install gptqmodel optimum -U

# For GPTQ v2 quantization:
pip install gptqmodel optimum -U

# For BNB quantization:
pip install bitsandbytes -U
```

`awq` is an independent optional extra; it is not included in the base installation or the `all` extra. Check the environment's dependency constraints before installing and run `python -m pip check` afterward. The following observations apply to the specific versions tested, not to all releases:

| Component | Tested version or known constraint |
| --- | --- |
| AWQ export test environment | Python 3.12, PyTorch 2.11.0+cu130, Transformers 5.14.1, GPTQModel 7.3.2 |
| GPTQModel 7.3.2 | Requires Python >=3.10; on Python 3.12, requires NumPy 2.2.6 and protobuf >=7.34.0 |
| SwanLab 0.9.2 | Requires protobuf <7, conflicting with GPTQModel 7.3.2 |
| ms-opencompass 0.1.6 | Requires NumPy <2, conflicting with GPTQModel 7.3.2 |

If AWQ, SwanLab, or ms-opencompass are needed in the same environment, first confirm that the chosen versions' declared requirements have a common solution; changing the installation order cannot resolve the conflicts above. If no common solution is available, use separate environments for quantization, training, and evaluation. Do not treat `--no-deps` as a resolution of `pip check` conflicts. Requirements may change by package and Python version; also verify PyTorch, CUDA, Transformers, and other installed components. The test image has other dependency conflicts, so successful AWQ export does not establish compatibility of the entire image.

We provide a series of scripts to demonstrate SWIFT's quantization export capabilities:

- Supports [AWQ](https://github.com/modelscope/ms-swift/blob/main/examples/export/quantize/awq.sh)/[GPTQ](https://github.com/modelscope/ms-swift/blob/main/examples/export/quantize/gptq.sh)/[GPTQ v2](https://github.com/modelscope/ms-swift/blob/main/examples/export/quantize/gptq_v2.sh)/[BNB](https://github.com/modelscope/ms-swift/blob/main/examples/export/quantize/bnb.sh) quantization exports.
- Multimodal quantization: GPTQ supports multimodal models; AWQ currently supports image calibration for Qwen2-VL only. Refer to [here](https://github.com/modelscope/ms-swift/tree/main/examples/export/quantize/mllm).
- Support for more model series: Supports quantization exports for [BERT](https://github.com/modelscope/ms-swift/tree/main/examples/export/quantize/bert) and [Reward Model](https://github.com/modelscope/ms-swift/tree/main/examples/export/quantize/reward_model).
- Models exported with SWIFT's quantization support inference acceleration using vllm/sglang/lmdeploy; they also support further SFT/RLHF using QLoRA.


AWQ uses GPTQModel to produce GEMM weights with zero points. It supports text and Qwen2-VL image calibration, safetensors, and unquantized source models; merge LoRA adapters first using `--merge_lora true`. Adding special tokens, setting `--device_map`, or passing custom model loading arguments during AWQ export are not supported. Qwen2-VL image calibration uses GPTQModel's processor and model chat template, does not support custom `--template`, `--system`, or `--max_pixels`, and requires `--truncation_strategy delete`. The model architecture must be supported by GPTQModel. Check compatibility with the existing PyTorch, Transformers, and other dependencies when installing it.

GPTQ backends may save symmetric weights in GPTQ v1 format even when GPTQ v2 is used during quantization. The exported configuration records the saved format so the model reloads correctly.

## Push Model

SWIFT supports re-pushing trained/quantized models to ModelScope/Hugging Face. By default, it pushes to ModelScope, but you can specify `--use_hf true` to push to Hugging Face.

```shell
swift export \
    --model output/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id '<model-id>' \
    --hub_token '<sdk-token>' \
    --use_hf false
```

Tips:

- You can use `--model <checkpoint-dir>` or `--adapters <checkpoint-dir>` to specify the checkpoint directory to be pushed. There is no difference between these two methods in the model pushing scenario.
- When pushing to ModelScope, you need to make sure you have registered for a ModelScope account. Your SDK token can be obtained from [this page](https://www.modelscope.cn/my/myaccesstoken). Ensure that the account associated with the SDK token has edit permissions for the organization corresponding to the model_id. The model pushing process will automatically create a model repository corresponding to the model_id (if it does not already exist), and you can use `--hub_private_repo true` to automatically create a private model repository.
