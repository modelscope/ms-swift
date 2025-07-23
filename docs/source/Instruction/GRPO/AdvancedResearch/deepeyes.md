

## 原理介绍


## 数据集

下载数据集到本地
```bash
# modelscope
modelscope download --dataset Lixiang/ChenShawn-DeepEyes-Datasets-47k

# huggingface
huggingface-cli download ChenShawn/DeepEyes-Datasets-47k --repo-type=dataset
```

在 swift/swift/llm/dataset/data/dataset_info.json 文件中进行数据集注册, 将数据集中的 `prompt` 列重命名为 `messages`

```json
    {
        "ms_dataset_id": "../data/data_0.1.2_visual_toolbox_v2.parquet",
        "columns": {
            "prompt": "messages"
        }
    },
    {
        "ms_dataset_id": "../data/data_thinklite_reasoning_acc.parquet",
        "columns": {
            "prompt": "messages"
        }
    },
    {
        "ms_dataset_id": "../data/data_v0.8_visual_toolbox_v2.parquet",
        "columns": {
            "prompt": "messages"
        }
    }
```

## 插件注册
注册论文中所用到的奖励函数和工具调用逻辑，参考文件swift/examples/train/grpo/plugin/deepeyes.py


Deepeyes 的奖励函数依赖生成式奖励模型对模型生成结果与标准答案进行对比评估，为了加速这一环节，推荐对模型进行部署。
假设使用Qwen2.5-VL-72B-Instruct模型进行评估，参考以下部署命令
```bash
# 4*80G
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift deploy \
    --model Qwen/Qwen2.5-VL-72B-Instruct \
    --infer_backend vllm \
    --vllm_tensor_parallel_size 4 \
```

在plugin文件中，根据部署结果修改client的base_url（默认为http://127.0.0.1:8000/v1）
