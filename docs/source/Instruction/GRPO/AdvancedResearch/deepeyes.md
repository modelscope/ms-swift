

## 原理介绍


## 最佳实践

**数据集下载与注册**

首先下载训练数据集到本地
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
