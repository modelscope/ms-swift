

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
