# 导出与推送


## Merge LoRA

- 查看[这里](https://github.com/modelscope/ms-swift/blob/main/examples/export/merge_lora.sh)。

## 量化

SWIFT支持AWQ、GPTQ、FP8、BNB模型的量化导出。其中使用AWQ、GPTQ需使用校准数据集，量化性能较好但量化耗时较长；而FP8、BNB无需校准数据集，量化耗时较短。

| 量化技术 | 多模态 | 推理加速 | 继续训练 |
| -------- | ------ | -------- | -------- |
| FP8     | ✅      | ✅        | ✅        |
| GPTQ     | ✅      | ✅        | ✅        |
| AWQ      | ❌      | ✅        | ✅        |
| BNB      | ❌      | ✅        | ✅        |


除SWIFT安装外，需要安装以下额外依赖：
```shell
# 使用 AWQ 量化（从源码目录安装）:
pip install -e '.[awq]'

# 已发布的 ms-swift 包提供该选项后，也可以使用:
# pip install 'ms-swift[awq]'

# 使用gptq量化:
pip install gptqmodel optimum -U

# 使用gptq v2量化:
pip install gptqmodel optimum -U

# 使用bnb量化：
pip install bitsandbytes -U
```

`awq` 是独立的可选依赖，不包含在基础安装或 `all` 选项中。安装前请检查环境中的依赖约束，安装后运行 `python -m pip check`。以下是已验证的特定版本，不能推断其他版本也兼容：

| 组件 | 测试版本或已知约束 |
| --- | --- |
| AWQ 导出测试环境 | Python 3.12、PyTorch 2.11.0+cu130、Transformers 5.14.1、GPTQModel 7.3.2 |
| GPTQModel 7.3.2 | 要求 Python >=3.10；Python 3.12 下要求 NumPy 2.2.6 和 protobuf >=7.34.0 |
| SwanLab 0.9.2 | 要求 protobuf <7，与 GPTQModel 7.3.2 的要求冲突 |
| ms-opencompass 0.1.6 | 要求 NumPy <2，与 GPTQModel 7.3.2 的要求冲突 |

如果同一环境需要 AWQ、SwanLab 或 ms-opencompass，先确认所选版本的依赖声明存在共同解；上述冲突版本不能通过调整安装顺序解决。若暂时没有共同解，可以将量化、训练和评测放在各自的环境中。不要使用 `--no-deps` 将 `pip check` 中的冲突视为已解决。实际依赖可能随版本和 Python 版本变化，还应验证 PyTorch、CUDA、Transformers 和其他已安装组件。上述测试镜像自身仍有其他依赖冲突，AWQ 导出测试通过不代表整个镜像兼容。

我们提供了一系列脚本展现SWIFT的量化导出能力：
- 支持[AWQ](https://github.com/modelscope/ms-swift/blob/main/examples/export/quantize/awq.sh)/[GPTQ](https://github.com/modelscope/ms-swift/blob/main/examples/export/quantize/gptq.sh)/[GPTQ v2](https://github.com/modelscope/ms-swift/blob/main/examples/export/quantize/gptq_v2.sh)/[BNB](https://github.com/modelscope/ms-swift/blob/main/examples/export/quantize/bnb.sh)量化导出。
- 多模态量化: GPTQ支持多模态模型；AWQ目前不支持多模态校准。参考[这里](https://github.com/modelscope/ms-swift/tree/main/examples/export/quantize/mllm)。
- 更多系列模型的支持: 支持[Bert](https://github.com/modelscope/ms-swift/tree/main/examples/export/quantize/bert)，[Reward Model](https://github.com/modelscope/ms-swift/tree/main/examples/export/quantize/reward_model)的量化导出。
- 使用SWIFT量化导出的模型支持使用vllm/sglang/lmdeploy进行推理加速；也支持使用QLoRA继续进行SFT/RLHF。


AWQ 使用 GPTQModel 生成带零点的 GEMM 格式权重。目前支持文本校准，使用 safetensors 和未量化的模型；LoRA 需要先使用 `--merge_lora true` 合并。AWQ 不支持在导出时添加特殊 token、指定 `--device_map` 或设置自定义模型加载参数。模型架构须受 GPTQModel 支持，安装时请检查与现有 PyTorch、Transformers 和其他依赖的兼容性。

使用 GPTQ v2 量化时，后端可能将对称量化权重保存为 GPTQ v1 格式。导出配置会记录实际保存的格式，以便正确重新加载模型。

## 推送模型

SWIFT支持将训练/量化的模型重新推送到ModelScope/HuggingFace。默认推送到ModelScope，你可以指定`--use_hf true`推送到HuggingFace。
```shell
swift export \
    --model output/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id '<model-id>' \
    --hub_token '<sdk-token>' \
    --use_hf false
```

小贴士：
- 你可以使用`--model <checkpoint-dir>`或者`--adapters <checkpoint-dir>`指定需要推送的checkpoint目录，这两种写法在推送模型场景没有差异。
- 推送到ModelScope时，你需要确保你已经注册了魔搭账号，你的SDK token可以在[该页面](https://www.modelscope.cn/my/myaccesstoken)中获取。推送模型需确保sdk token的账号具有model_id对应组织的编辑权限。推送模型将自动创建对应model_id的模型仓库（如果该模型仓库不存在），你可以使用`--hub_private_repo true`来自动创建私有的模型仓库。
