# 快速开始

🍲 ms-swift是魔搭社区提供的大模型与多模态大模型微调部署框架，现已支持600+纯文本大模型与300+多模态大模型的训练（预训练、微调、人类对齐）、推理、评测、量化与部署。其中大模型包括：Qwen3、Qwen3-Next、InternLM3、GLM4.5、Mistral、DeepSeek-R1、Llama4等模型，多模态大模型包括：Qwen3-VL、Qwen3-Omni、Llava、InternVL3.5、MiniCPM-V-4、Ovis2.5、GLM4.5-V、DeepSeek-VL2等模型。

🍔 除此之外，ms-swift汇集了最新的训练技术，包括集成Megatron并行技术，包括TP、PP、CP、EP等为训练提供加速，以及众多GRPO算法族强化学习的算法，包括：GRPO、DAPO、GSPO、SAPO、CISPO、RLOO、Reinforce++等提升模型智能。ms-swift支持广泛的训练任务，包括DPO、KTO、RM、CPO、SimPO、ORPO等偏好学习算法，以及Embedding、Reranker、序列分类任务。ms-swift提供了大模型训练全链路的支持，包括使用vLLM、SGLang和LMDeploy对推理、评测、部署模块提供加速，以及使用GPTQ、AWQ、BNB、FP8技术对大模型进行量化。

**为什么选择ms-swift？**
- 🍎 **模型类型**：支持600+纯文本大模型、**300+多模态大模型**以及All-to-All全模态模型训练到部署全流程，热门模型Day0支持。
- **数据集类型**：内置150+预训练、微调、人类对齐、多模态等各种任务数据集，并支持自定义数据集，用户只需准备数据集即可一键训练。
- **硬件支持**：支持A10/A100/H100、RTX系列、T4/V100、CPU、MPS以及国产硬件Ascend NPU等。
- **轻量训练**：支持了LoRA、QLoRA、DoRA、LoRA+、LLaMAPro、LongLoRA、LoRA-GA、ReFT、RS-LoRA、Adapter、LISA等轻量微调方式。
- **量化训练**：支持对BNB、AWQ、GPTQ、AQLM、HQQ、EETQ量化模型进行训练，7B模型训练只需9GB训练资源。
- **显存优化**: GaLore、Q-Galore、UnSloth、Liger-Kernel、Flash-Attention 2/3 以及 **Ulysses和Ring-Attention序列并行技术**支持，降低长文本训练显存占用。
- **分布式训练**：支持分布式数据并行（DDP）、device_map简易模型并行、DeepSpeed ZeRO2 ZeRO3、FSDP/FSDP2以及Megatron等分布式训练技术。
- 🍓 **多模态训练**：支持多模态packing技术提升训练速度100%+，支持文本、图像、视频和语音混合模态数据训练，支持vit/aligner/llm单独控制。
- **Agent训练**：支持Agent template，准备一套数据集可用于不同模型的训练。
- 🍊 **训练任务**：支持预训练和指令微调，以及DPO、GKD、KTO、RM、CPO、SimPO、ORPO等训练任务，支持**Embedding/Reranker**和序列分类任务。
- 🥥 **Megatron并行技术**：提供TP/PP/SP/CP/ETP/EP/VPP并行策略，**MoE模型加速可达10倍**。支持250+纯文本大模型和100+多模态大模型的全参数和LoRA训练方法。支持CPT/SFT/GRPO/DPO/KTO/RM训练任务。
- 🍉 **强化学习**：内置**丰富GRPO族算法**，包括GRPO、DAPO、GSPO、SAPO、CISPO、CHORD、RLOO、Reinforce++等，支持同步和异步vLLM引擎推理加速，可使用插件拓展奖励函数、多轮推理调度器以及环境等。
- **全链路能力**：覆盖训练、推理、评测、量化和部署全流程。
- **界面训练**：提供使用Web-UI界面的方式进行训练、推理、评测、量化，完成大模型的全链路。
- **推理加速**：支持PyTorch、vLLM、SGLang和LmDeploy推理加速引擎，并提供OpenAI接口，为推理、部署和评测模块提供加速。
- **模型评测**：以EvalScope作为评测后端，支持100+评测数据集对纯文本和多模态模型进行评测。
- **模型量化**：支持AWQ、GPTQ、FP8和BNB的量化导出，导出的模型支持使用vLLM/SGLang/LmDeploy推理加速。


## 安装

ms-swift的安装请参考[安装文档](./SWIFT-installation.md)。

## 使用样例

10分钟在单卡3090上对Qwen2.5-7B-Instruct进行自我认知微调：
```shell
# 22GB
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
```

小贴士：
- 如果要使用自定义数据集进行训练，你可以参考[这里](../Customization/Custom-dataset.md)组织数据集格式，并指定`--dataset <dataset_path>`。
- `--model_author`和`--model_name`参数只有当数据集中包含`swift/self-cognition`时才生效。
- 如果要使用其他模型进行训练，你只需要修改`--model <model_id/model_path>`即可。
- 默认使用ModelScope进行模型和数据集的下载。如果要使用HuggingFace，指定`--use_hf true`即可。

训练完成后，使用以下命令对训练后的权重进行推理：
- 这里的`--adapters`需要替换成训练生成的last checkpoint文件夹。由于adapters文件夹中包含了训练的参数文件`args.json`，因此不需要额外指定`--model`，`--system`，swift会自动读取这些参数。如果要关闭此行为，可以设置`--load_args false`。

```shell
# 使用交互式命令行进行推理
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048

# merge-lora并使用vLLM进行推理加速
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --merge_lora true \
    --infer_backend vllm \
    --vllm_max_model_len 8192 \
    --temperature 0 \
    --max_new_tokens 2048
```

最后，使用以下命令将模型推送到ModelScope：
```shell
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --adapters output/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id '<your-model-id>' \
    --hub_token '<your-sdk-token>' \
    --use_hf false
```

## 了解更多

- 更多Shell脚本：[https://github.com/modelscope/ms-swift/tree/main/examples](https://github.com/modelscope/ms-swift/tree/main/examples)
- 使用Python：[https://github.com/modelscope/ms-swift/blob/main/examples/notebook/qwen2_5-self-cognition/self-cognition-sft.ipynb](https://github.com/modelscope/ms-swift/blob/main/examples/notebook/qwen2_5-self-cognition/self-cognition-sft.ipynb)
