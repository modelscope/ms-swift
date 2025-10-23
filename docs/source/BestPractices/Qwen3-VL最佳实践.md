
# Qwen3-VL最佳实践

## 环境准备

在开始推理和训练之前，请确保您的环境已准备就绪。

```shell
pip install "transformers>=4.57" "qwen_vl_utils>=0.0.14"

pip install "ms-swift>=3.9.1"
# pip install "vllm>=0.11.0"  # 若使用vllm推理后端进行推理
```


## 推理

使用 transformers 推理：
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from modelscope import snapshot_download
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model_dir = snapshot_download('Qwen/Qwen3-VL-4B-Instruct')

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_dir, dtype="auto", device_map="auto",
    # attn_implementation='flash_attention_2',
)

processor = AutoProcessor.from_pretrained(model_dir)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4",
                "max_pixels": 128*32*32,
                "max_frames": 16,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True,
                                                                image_patch_size= 16,
                                                                return_video_metadata=True)
if video_inputs is not None:
    video_inputs, video_metadatas = zip(*video_inputs)
    video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
else:
    video_metadatas = None
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, video_metadata=video_metadatas, **video_kwargs, do_resize=False, return_tensors="pt")
inputs = inputs.to('cuda')

generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
# 'A baby wearing glasses sits on a bed, engrossed in reading a book. The baby turns the pages with both hands, occasionally looking up and smiling. The room is cozy, with a crib in the background and clothes scattered around. The baby’s focus and curiosity are evident as they explore the book, creating a heartwarming scene of early learning and discovery.'
```

使用 ms-swift 的 `PtEngine` 进行推理：
```python
import os
# os.environ['SWIFT_DEBUG'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '128'
os.environ['FPS_MAX_FRAMES'] = '16'


from swift.llm import PtEngine, InferRequest, RequestConfig
engine = PtEngine('Qwen/Qwen3-VL-4B-Instruct', attn_impl='flash_attention_2')
infer_request = InferRequest(messages=[{
    "role": "user",
    "content": '<video>Describe this video.',
}], videos=['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4'])
request_config = RequestConfig(max_tokens=128, temperature=0)
resp_list = engine.infer([infer_request], request_config=request_config)
response = resp_list[0].choices[0].message.content
# 'A baby wearing glasses sits on a bed, engrossed in reading a book. The baby turns the pages with both hands, occasionally looking up and smiling. The room is cozy, with a crib in the background and clothes scattered around. The baby’s focus and curiosity are evident as they explore the book, creating a heartwarming scene of early learning and discovery.'

# use stream
request_config = RequestConfig(max_tokens=128, temperature=0, stream=True)
gen_list = engine.infer([infer_request], request_config=request_config)
for chunk in gen_list[0]:
    if chunk is None:
        continue
    print(chunk.choices[0].delta.content, end='', flush=True)
print()
```

使用命令行推理：
```shell
CUDA_VISIBLE_DEVICES=0 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
swift infer \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --stream true
```

```
<<< who are you?
Hello! I'm Qwen, a large-scale language model independently developed by the Tongyi Lab under Alibaba Group. My main functions include answering questions, creating text such as stories, official documents, emails, scripts, and more, as well as performing logical reasoning, programming, and other tasks. If you have any questions or need assistance, feel free to let me know anytime, and I'll do my best to help!
--------------------------------------------------
<<< <image>describe the image.
Input an image path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
This is a beautifully detailed, close-up portrait of an adorable tabby kitten, rendered with a soft, painterly effect that gives it a gentle, dreamy quality.

Here’s a breakdown of the image:

- **The Kitten:** The subject is a young, fluffy kitten with a classic tabby pattern. Its fur is a mix of white and soft grayish-brown stripes, with a prominent dark stripe running down the center of its forehead and over its nose. The kitten’s face is predominantly white, with delicate markings around its eyes and cheeks.

- **The Eyes:** Its most captivating feature is its large, round, and expressive eyes. They are a striking shade of bright blue-gray, with dark pupils that give it an intense, curious, and slightly innocent gaze. The eyes are wide open, suggesting the kitten is alert and attentive.

- **The Expression:** The kitten’s expression is sweet and innocent. Its small pink nose and slightly parted mouth give it a gentle, almost pleading look. Its whiskers are long and white, standing out against its fur.

- **The Style:** The image has a soft-focus, artistic quality, reminiscent of impressionist painting. The edges of the kitten’s fur are slightly blurred, creating a halo effect that draws attention to its face. The background is softly blurred with muted tones of green and gray, which helps the kitten stand out as the clear focal point.

- **Overall Impression:** The image evokes feelings of warmth, cuteness, and tenderness. The kitten appears to be looking directly at the viewer, creating a sense of connection and affection.

This is a lovely and charming depiction of a young kitten, capturing its innocence and charm in a visually appealing and emotionally engaging way.
--------------------------------------------------
<<< <video>describe the video.
Input a video path or URL <<< https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4
This video captures a charming and adorable moment of a young child, likely a toddler, sitting on a bed and pretending to read a book. The child is wearing glasses, which adds a humorous and endearing touch to the scene — as if they’re a little scholar or librarian.

Here’s a breakdown of what unfolds:

- The child is seated cross-legged on a bed with a patterned quilt. Behind them, a crib and some household items are visible, suggesting a cozy bedroom setting.

- The child holds an open book and appears to be turning the pages with focused attention, mimicking the behavior of a real reader.

- At one point, the child looks up, smiles, or seems to react with delight — perhaps amused by something in the book or just enjoying the activity.

- The child’s movements are gentle and deliberate, showing a sense of concentration and curiosity. They turn pages, sometimes with one hand, and occasionally lift the book slightly as if to examine it more closely.

- The video has a warm, candid feel — it’s not staged, and the child’s natural behavior makes it feel authentic and heartwarming.

Overall, this is a sweet, lighthearted video that showcases the innocence and imagination of early childhood. The child’s engagement with the book, combined with their glasses and playful demeanor, creates a delightful and memorable scene.
```

- 其中特定模型参数，例如 `VIDEO_MAX_TOKEN_NUM` 等环境变量的含义参考[命令行参数文档](../Instruction/命令行参数.md#qwen3_vl)。


## 训练

本文档将介绍如何使用 ms-swift 与 Megatron-SWIFT 训练 Qwen3-VL。推荐 Dense 模型使用 ms-swift（即 transformers 后端，更加方便简单），而 Moe 模型使用 Megatron-SWIFT（即 megatron 后端，更快的训练速度，benchmark查看[这里](../Megatron-SWIFT/快速开始.md#benchmark)）。

如果您需要自定义数据集微调模型，你可以将数据准备成以下格式，并在命令行中设置`--dataset train.jsonl --val_dataset val.jsonl`，其中验证集为可选。更多介绍请参考[多模态数据集文档](../Customization/自定义数据集.md#多模态)。
```jsonl
{"messages": [{"role": "user", "content": "浙江的省会在哪？"}, {"role": "assistant", "content": "浙江的省会在杭州。"}]}
{"messages": [{"role": "user", "content": "<image><image>两张图片有什么区别"}, {"role": "assistant", "content": "前一张是小猫，后一张是小狗"}], "images": ["/xxx/x.jpg", "/xxx/x.png"]}
{"messages": [{"role": "system", "content": "你是个有用无害的助手"}, {"role": "user", "content": "<image>图片中是什么，<video>视频中是什么"}, {"role": "assistant", "content": "图片中是一个大象，视频中是一只小狗在草地上奔跑"}], "images": ["/xxx/x.jpg"], "videos": ["/xxx/x.mp4"]}
```

Qwen3-VL的bbox输出采用归一化1000的相对坐标。你可以使用 ms-swift 提供的 grounding 数据集格式，其中"bbox"中的坐标为绝对坐标，ms-swift 会自动将绝对坐标转为归一化1000的相对坐标。更多信息请参考[grounding数据集格式文档](../Customization/自定义数据集.md#grounding)。
```jsonl
{"messages": [{"role": "user", "content": "<image>找到图像中的<ref-object>"}, {"role": "assistant", "content": "[\n\t{\"bbox_2d\": <bbox>, \"label\": \"<ref-object>\"}\n\t{\"bbox_2d\": <bbox>, \"label\": \"<ref-object>\"}\n]"}], "images": ["cat.png"], "objects": {"ref": ["羊", "羊", "羊"], "bbox": [[90.9, 160.8, 135, 212.8], [360.9, 480.8, 495, 532.8]]}}
```

### Dense模型

以下提供对`Qwen3-VL-4B-Instruct`模型的微调脚本，我们使用混合模态数据作为Demo数据集，该示例脚本仅作为演示用途。训练显存为2 * 21GiB，训练时间为12分钟。

```shell
# 2 * 21GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#10000' \
              'AI-ModelScope/LaTeX_OCR:human_handwrite#5000' \
              'swift/VideoChatGPT:Generic#2000' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --attn_impl flash_attn \
    --padding_free true \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --packing true \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --gradient_accumulation_steps 2 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --deepspeed zero2 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4
```

训练结束后，我们使用以下脚本对验证集进行推理：
```shell
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --max_new_tokens 2048 \
    --load_data_args true
```

```
--------------------------------------------------
[QUERY] Using LaTeX to perform OCR on the image.
[LABELS] 1 + \frac { 1 } { 1 ! } + \frac { 1 } { 2 ! } + \frac { 1 } { 3 ! } + \frac { 1 } { 4 ! }
[RESPONSE] 1 + \frac { 1 } { 1 ! } + \frac { 1 } { 2 ! } + \frac { 1 } { 3 ! } + \frac { 1 } { 4 ! }
--------------------------------------------------
[QUERY] What color suit is the man wearing while playing the saxophone on stage?
[LABELS] The man is wearing a black suit and white shirt while playing the saxophone on the red-floored stage.
[RESPONSE] The man is wearing a black suit while playing the saxophone on stage.
--------------------------------------------------
...
```

### Moe模型


以下提供对`Qwen3-VL-30B-A3B-Instruct`模型的微调脚本，我们使用 Megatron-SWIFT 进行单机全参数训练。我们同样采用混合数据进行训练，该示例脚本仅作为演示用途。训练所需显存资源为8 * 80GiB，训练时间为20分钟。

关于 Megatron-SWIFT 的环境安装，请参考[Megatron-SWIFT文档](../Megatron-SWIFT/快速开始.md)。Megatron-SWIFT 与 ms-swift 共用 template 和 dataset 模块，因此前面介绍的自定义数据集格式和模型特有环境变量依旧生效。

HF格式权重转为Megatron格式：
```shell
CUDA_VISIBLE_DEVICES=0,1 \
swift export \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir Qwen3-VL-30B-A3B-Instruct-mcore
```

微调脚本如下，训练技巧与并行技术的调整参考[Megatron-SWIFT文档](../Megatron-SWIFT/快速开始.md#训练技巧)。
```shell
# 8 * 80GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
OMP_NUM_THREADS=14 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
megatron sft \
    --load Qwen3-VL-30B-A3B-Instruct-mcore \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#10000' \
              'AI-ModelScope/LaTeX_OCR:human_handwrite#5000' \
              'swift/VideoChatGPT:Generic#2000' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --moe_permute_fusion true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 8 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save megatron_output/Qwen3-VL-30B-A3B-Instruct \
    --eval_interval 500 \
    --save_interval 500 \
    --max_length 4096 \
    --packing true \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --moe_expert_capacity_factor 2 \
    --optimizer_cpu_offload true \
    --use_precision_aware_optimizer true \
    --optimizer_offload_fraction 0.2 \
    --attention_backend flash
```

Megatron格式权重转为Hf格式：
```shell
CUDA_VISIBLE_DEVICES=0,1 \
swift export \
    --mcore_model megatron_output/Qwen3-VL-30B-A3B-Instruct/vx-xxx \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir megatron_output/Qwen3-VL-30B-A3B-Instruct/vx-xxx-hf
```
- 若要调整使用对应迭代次数（iter）的权重，请修改`megatron_output/Qwen3-VL-30B-A3B-Instruct/vx-xxx`目录下的`latest_checkpointed_iteration.txt`文件。


训练结束后，我们使用以下脚本对验证集进行推理：
```shell
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
swift infer \
    --model megatron_output/Qwen3-VL-30B-A3B-Instruct/vx-xxx-hf \
    --stream true \
    --max_new_tokens 2048 \
    --load_data_args true
```


使用以下命令将训练权重推送到 Modelscope：
```shell
swift export \
    --model output/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id '<your-model-id>' \
    --hub_token '<your-sdk-token>'
```
