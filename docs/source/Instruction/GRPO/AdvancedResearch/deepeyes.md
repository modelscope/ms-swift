# DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning

## 原理介绍

[DeepEyes论文](https://arxiv.org/abs/2505.14362) 提出了一种方法使用强化学习模型学会“think with images”能力，
论文通过端到端强化学习自发涌现模型能力，无需单独的SFT过程。模型内建图像定位能力，体现在“图像放大工具”的主动调用上：模型在推理过程中自动选取图片具体区域并进行放大、裁剪，将处理后的区域信息与推理过程串联融合，实现视觉与文本的链式思考

![DeepEyes Overview](../../../../resources/deepeyes.png)

## 最佳实践

**数据集下载与注册**

下载 DeepEyes 官方训练数据集到本地
```bash
# modelscope
modelscope download --dataset Lixiang/ChenShawn-DeepEyes-Datasets-47k

# huggingface
huggingface-cli download ChenShawn/DeepEyes-Datasets-47k --repo-type=dataset
```

数据集内有三个parquet文件，`swift/swift/llm/dataset/data/dataset_info.json` 文件中分别对其注册，将数据集中的 `prompt` 列重命名为 `messages`

```json
    {
        "ms_dataset_id": "path/to/data_0.1.2_visual_toolbox_v2.parquet",
        "columns": {
            "prompt": "messages"
        }
    },
    {
        "ms_dataset_id": "path/to/data/data_thinklite_reasoning_acc.parquet",
        "columns": {
            "prompt": "messages"
        }
    },
    {
        "ms_dataset_id": "path/to/data/data_v0.8_visual_toolbox_v2.parquet",
        "columns": {
            "prompt": "messages"
        }
    }
```

在本地注册论文中所用到的奖励函数和工具调用逻辑，参考文件https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/plugin/deepeyes.py

**部署验证模型**

Deepeyes 的奖励函数依赖生成式奖励模型对模型生成结果与标准答案进行对比评估，为了加速这一环节，推荐对模型进行部署。
假设使用 Qwen2.5-VL-72B-Instruct 模型进行评估，参考以下部署命令
```bash
# 4*80G
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift deploy \
    --model Qwen/Qwen2.5-VL-72B-Instruct \
    --infer_backend vllm \
    --vllm_tensor_parallel_size 4 \
```

在 plugin 文件中，根据部署结果修改 client 的 base_url（默认为http://127.0.0.1:8000/v1）

**工具调用**
根据论文，我们需要提供模型一个工具可以提取图像的部分区域，并将



完整的训练脚本参考 https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/plugin/deepeyes.sh
