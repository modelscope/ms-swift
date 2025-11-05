# DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning

**版本依赖**：ms-swift>=3.7

## 原理介绍

[DeepEyes论文](https://arxiv.org/abs/2505.14362) 提出了一种利用强化学习使模型具备“think with images”（以图辅助思考）能力的方法。该方法通过端到端的强化学习，模型能力自发涌现，无需额外的 SFT（监督微调）过程。模型内置图像定位能力，能够主动调用“图像放大工具”：在推理过程中，模型会自动选取图片中的具体区域进行放大和裁剪，将处理后的区域信息进行进一步推理，实现视觉与文本的链式推理。

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

数据集内有三个parquet文件，`swift/swift/llm/dataset/data/dataset_info.json` 文件中分别进行注册，将数据集中的 `prompt` 列重命名为 `messages`

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

在本地注册论文中所用到的奖励函数和工具调用逻辑，实现可以参考[DeepEyes实现示例](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/plugin/deepeyes/deepeyes_plugin.py)

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

在 plugin 文件中，使用OpenAI接口进行调用，参考[奖励模型文档](../DeveloperGuide/奖励模型.md#外部部署)


训练参考该[脚本](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/plugin/deepeyes/deepeyes.sh)

## 实现细节

[DeepEyes实现示例](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/plugin/deepeyes/deepeyes_plugin.py)参考[官方实现](https://github.com/Visual-Agent/DeepEyes/blob/main/verl/utils/reward_score/vl_agent.py) 给出了 DeepEyes 训练插件的样例代码，涵盖了奖励函数与多轮交互调用的相关逻辑。

**数据集数据**如下

| 数据集文件名                             | data_source           | 对应评分函数                         | 工具调用         |
|------------------------------------------|-----------------------|----------------------------------|------------------|
| data_v0.8_visual_toolbox_v2.parquet      | chart                 | vl_agent.compute_score           | True (image_zoom_in_tool)  |
| data_0.1.2_visual_toolbox_v2.parquet     | vstar                 | vl_agent.compute_score           | True (image_zoom_in_tool)  |
| data_thinklite_reasoning_acc.parquet     | thinklite_eureka      | vl_agent.compute_score_math      | False           |


**注意**：多模态大模型在处理图像输入时，可能会对图像进行预处理（例如受 max_pixels 参数限制的裁剪或缩放等操作）。当调用图像放大工具 image_zoom_in_tool 时，模型会根据输入图像输出裁剪后的 bbox。因此，在调用图像放大工具时，需要确保输入的是经过预处理后的图像。示例代码展示了 Qwen2.5-VL 系列模型的实现方式：

```python
from qwen_vl_utils import fetch_image
# 这里的images尚未经过图像处理
infer_request.images
# 通过加载为PIL.Image格式，进行裁剪（使用环境变量MAX_PIXELS时的处理）
img = fetch_image({'image': load_pil_image(infer_request.images[0])})
```

**工具奖励**

论文中指出当最终答案正确，且轨迹至少使用一个工具时给予工具奖励。为了避免模型生成的工具调用是无效的，我们通过图像数量而不是`<tool_call>` 等token进行判断。
```python
tool_reward = 1.0 if num_image > 1 and acc_reward > 0.5 else 0.0
```
