# Qwen3.5 最佳实践

ms-swift 4.0支持使用transformers/Megatron后端对[Qwen3.5](https://github.com/QwenLM/Qwen3.5) Dense/Moe模型进行训练。Qwen3.5 属于混合思考的多模态模型，结合了linear attention和full attention。本文将介绍如何对Qwen3.5 Dense/Moe模型进行推理、指令微调以及强化学习。（强化学习部分将在后续补充）


## 环境设置
```shell
pip install -U ms-swift
pip install -U "transformers>=5.2.0" "qwen_vl_utils>=0.0.14" peft liger-kernel

# flash-linear-attention
# 请安装fla main分支，若出现训练缓慢的问题请参考：https://github.com/fla-org/flash-linear-attention/issues/758
pip install -U git+https://github.com/fla-org/flash-linear-attention

# deepspeed训练
pip install deepspeed

# vllm (torch2.10) for inference/deployment/RL
pip install uv
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
# RL训练，需覆盖vllm默认安装版本
# 训练报错参考这个issue: https://github.com/modelscope/ms-swift/issues/8188
pip install -U "transformers>=5.2.0"
```

- Qwen3.5 视频数据训练卡住：使用decord后端读取视频可能导致卡住问题，参考[这个issue](https://github.com/dmlc/decord/issues/269)。你可以使用torchcodec后端，具体参考[qwen_vl_utils](https://github.com/QwenLM/Qwen3-VL/blob/50068df2334f309979ff05d75f1078c8309c63ed/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L390-L400)库。


## 推理

使用 ms-swift 的 `TransformersEngine` 进行推理：

- 其中特定模型参数，例如 `VIDEO_MAX_TOKEN_NUM` 等环境变量的含义与Qwen3-VL相同，参考[命令行参数文档](../Instruction/Command-line-parameters.md#qwen3_vl,qwen3_5)。

```python
import os
# os.environ['SWIFT_DEBUG'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['IMAGE_MAX_TOKEN_NUM'] = '1024'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '128'
os.environ['FPS_MAX_FRAMES'] = '16'

from swift import get_model_processor, get_template
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig

model, processor = get_model_processor('Qwen/Qwen3.5-4B')  # attn_impl='flash_attention_2'
template = get_template(processor, enable_thinking=False)
engine = TransformersEngine(model, template=template)
infer_request = InferRequest(messages=[{
    "role": "user",
    "content": '<video>Describe this video.',
}], videos=['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4'])
request_config = RequestConfig(max_tokens=128, temperature=0)
resp_list = engine.infer([infer_request], request_config=request_config)
response = resp_list[0].choices[0].message.content
print(response)

# use stream
request_config = RequestConfig(max_tokens=128, temperature=0, stream=True)
gen_list = engine.infer([infer_request], request_config=request_config)
for chunk in gen_list[0]:
    if chunk is None:
        continue
    print(chunk.choices[0].delta.content, end='', flush=True)
print()
```

使用命令行进行推理：

```shell
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen3.5-4B \
    --enable_thinking false \
    --stream true
```


## 微调

本章将介绍如何使用 ms-swift 与 Megatron-SWIFT 训练 Qwen3.5。推荐 Dense 模型使用 ms-swift（即 transformers 后端，更加方便简单），而 Moe 模型使用 Megatron-SWIFT（即 megatron 后端，更快的训练速度）

如果您需要自定义数据集微调模型，你可以将数据准备成以下格式，并在命令行中设置`--dataset train.jsonl --val_dataset val.jsonl`，其中验证集为可选。更多介绍请参考[多模态数据集文档](../Customization/Custom-dataset.md#多模态)。

```jsonl
{"messages": [{"role": "user", "content": "浙江的省会在哪？"}, {"role": "assistant", "content": "浙江的省会在杭州。"}]}
{"messages": [{"role": "user", "content": "<image><image>两张图片有什么区别"}, {"role": "assistant", "content": "前一张是小猫，后一张是小狗"}], "images": ["/xxx/x.jpg", "/xxx/x.png"]}
{"messages": [{"role": "system", "content": "你是个有用无害的助手"}, {"role": "user", "content": "<image>图片中是什么，<video>视频中是什么"}, {"role": "assistant", "content": "图片中是一个大象，视频中是一只小狗在草地上奔跑"}], "images": ["/xxx/x.jpg"], "videos": ["/xxx/x.mp4"]}
```

Qwen3.5的bbox输出采用归一化1000的相对坐标。你可以使用 ms-swift 提供的 grounding 数据集格式，其中"bbox"中的坐标为绝对坐标，ms-swift 会自动将绝对坐标转为归一化1000的相对坐标。更多信息请参考[grounding数据集格式文档](../Customization/Custom-dataset.md#grounding)。

```jsonl
{"messages": [{"role": "user", "content": "<image>找到图像中的<ref-object>"}, {"role": "assistant", "content": "[\n\t{\"bbox_2d\": <bbox>, \"label\": \"<ref-object>\"},\n\t{\"bbox_2d\": <bbox>, \"label\": \"<ref-object>\"}\n]"}], "images": ["cat.png"], "objects": {"ref": ["羊", "羊", "羊"], "bbox": [[90.9, 160.8, 135, 212.8], [360.9, 480.8, 495, 532.8]]}}
```


### Dense模型

以下提供对Qwen3.5-4B模型的微调脚本，该示例脚本仅作为演示用途。训练显存为 4 * 20GiB，训练时间为12分钟。由于GatedDeltaNet不支持packing/padding_free，因此我们使用group_by_length参数来加速训练，保证DP的负载均衡并减少micro batch中的零填充，但这会导致loss曲线跳动（因数据随机不充分），当然你也可以去掉此参数。
对模型进行微调的脚本如下：

```shell
# 4 * 20GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen3.5-4B \
    --tuner_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
              'AI-ModelScope/LaTeX_OCR:human_handwrite#2000' \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --group_by_length true \
    --output_dir output/Qwen3.5-4B \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --deepspeed zero2 \
    --model_author swift \
    --model_name swift-robot
```

训练结束后，使用以下脚本对验证集进行推理：

```shell
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift infer \
    --adapters output/Qwen3.5-4B/vx-xxx/checkpoint-xxx \
    --stream true \
    --enable_thinking false \
    --max_new_tokens 512 \
    --load_data_args true
```


```text
[QUERY] 你好，你是谁？
[RESPONSE] <think>

</think>

你好，我是由swift开发的人工智能语言模型，我的名字叫swift-robot。很高兴能与你交流。
--------------------------------------------------
[QUERY] Using LaTeX to perform OCR on the image.
[LABELS] e = \sum _ { k = 0 } ^ { \infty } \frac { 1 } { k ! }
[RESPONSE] <think>

</think>

e = \sum _ { k = 0 } ^ { \infty } \frac { 1 } { k ! }
```

使用python进行推理：
```python
import os
# os.environ['SWIFT_DEBUG'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['IMAGE_MAX_TOKEN_NUM'] = '1024'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '128'
os.environ['FPS_MAX_FRAMES'] = '16'

from peft import PeftModel
from swift import get_model_processor, get_template
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig

adapter_dir = 'output/Qwen3.5-4B/vx-xxx/checkpoint-xxx'
enable_thinking = False

model, processor = get_model_processor('Qwen/Qwen3.5-4B')  # attn_impl='flash_attention_2'
model = PeftModel.from_pretrained(model, adapter_dir)
template = get_template(processor, enable_thinking=enable_thinking)
engine = TransformersEngine(model, template=template)
infer_request = InferRequest(messages=[{
    "role": "user",
    "content": 'who are you?',
}])
request_config = RequestConfig(max_tokens=128, temperature=0)
resp_list = engine.infer([infer_request], request_config=request_config)
response = resp_list[0].choices[0].message.content
print(response)

# use stream
request_config = RequestConfig(max_tokens=128, temperature=0, stream=True)
gen_list = engine.infer([infer_request], request_config=request_config)
for chunk in gen_list[0]:
    if chunk is None:
        continue
    print(chunk.choices[0].delta.content, end='', flush=True)
print()
# I am an artificial intelligence assistant named swift-robot, trained by swift. I am designed to understand and generate natural language text in order to provide information, answer questions, and engage in conversation with humans. How can I assist you?
```


使用transformers后端训练MoE的例子参考：https://github.com/modelscope/ms-swift/blob/main/examples/models/qwen3_5/transformers.sh

### Moe模型
Qwen3.5-35B-A3B Megatron训练，环境的准备请参考[Megatron-SWIFT快速开始文档](../Megatron-SWIFT/Quick-start.md)。你可以在15分钟内跑完以下案例：

```shell
# 4 * 40GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
megatron sft \
    --model Qwen/Qwen3.5-35B-A3B \
    --save_safetensors true \
    --merge_lora true \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
              'AI-ModelScope/LaTeX_OCR:human_handwrite#2000' \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0.01 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --expert_model_parallel_size 4 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --micro_batch_size 4 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 1 \
    --group_by_length true \
    --finetune true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --output_dir megatron_output/Qwen3.5-35B-A3B \
    --eval_steps 200 \
    --save_steps 200 \
    --max_length 2048 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --padding_free false \
    --model_author swift \
    --model_name swift-robot
```
训练结束后，使用以下脚本对验证集进行推理：

```shell
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift infer \
    --model megatron_output/Qwen3.5-35B-A3B/vx-xxx/checkpoint-xxx-merged \
    --stream true \
    --enable_thinking false \
    --max_new_tokens 512 \
    --load_data_args true
```

## 强化学习（RL）

敬请期待
