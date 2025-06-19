# Export and Push

## Merge LoRA

- See [here](https://github.com/modelscope/ms-swift/blob/main/examples/export/merge_lora.sh).

## Quantization

SWIFT supports quantization exports for AWQ, GPTQ, and BNB models. AWQ and GPTQ require a calibration dataset, which yields better quantization performance but takes longer to quantize. On the other hand, BNB does not require a calibration dataset and is quicker to quantize.

| Quantization Technique | Multimodal | Inference Acceleration | Continued Training |
| ---------------------- | ---------- | ---------------------- | ------------------ |
| GPTQ                   | ✅          | ✅                      | ✅                  |
| AWQ                    | ✅          | ✅                      | ✅                  |
| BNB                    | ❌          | ✅                      | ✅                  |

In addition to the SWIFT installation, the following additional dependencies need to be installed:

```shell
# For AWQ quantization:
# The versions of autoawq and CUDA are correlated; please choose the version according to `https://github.com/casper-hansen/AutoAWQ`.
# If there are dependency conflicts with torch, please add the `--no-deps` option.
pip install autoawq -U

# For GPTQ quantization:
# The versions of auto_gptq and CUDA are correlated; please choose the version according to `https://github.com/PanQiWei/AutoGPTQ#quick-installation`.
pip install auto_gptq optimum -U

# For BNB quantization:
pip install bitsandbytes -U
```

We provide a series of scripts to demonstrate SWIFT's quantization export capabilities:

- Supports [AWQ](https://github.com/modelscope/ms-swift/blob/main/examples/export/quantize/awq.sh)/[GPTQ](https://github.com/modelscope/ms-swift/blob/main/examples/export/quantize/gptq.sh)/[BNB](https://github.com/modelscope/ms-swift/blob/main/examples/export/quantize/bnb.sh) quantization exports.
- Multimodal quantization: Supports quantizing multimodal models using GPTQ and AWQ, with limited multimodal models supported by AWQ. Refer to [here](https://github.com/modelscope/ms-swift/tree/main/examples/export/quantize/mllm).
- Support for more model series: Supports quantization exports for [BERT](https://github.com/modelscope/ms-swift/tree/main/examples/export/quantize/bert) and [Reward Model](https://github.com/modelscope/ms-swift/tree/main/examples/export/quantize/reward_model).
- Models exported with SWIFT's quantization support inference acceleration using vllm/sglang/lmdeploy; they also support further SFT/RLHF using QLoRA.


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
