# AnimateDiff的微调和推理

SWIFT已经支持了AnimateDiff的微调和推理，目前支持两种方式：全参数微调和LoRA微调。

首先需要clone并安装SWIFT：

```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install ".[aigc]"
```

## 全参数训练

### 训练效果

全参数微调可以复现[官方提供的模型animatediff-motion-adapter-v1-5-2](https://www.modelscope.cn/models/Shanghai_AI_Laboratory/animatediff-motion-adapter-v1-5-2/summary)的效果，需要的短视频数量较多，魔搭官方复现使用了官方数据集的subset版本：[WebVid 2.5M](https://maxbain.com/webvid-dataset/)。训练效果如下：

```text
Prompt:masterpiece, bestquality, highlydetailed, ultradetailed, girl, walking, on the street, flowers
```



![image.png](../../resources/1.gif)

```text
Prompt: masterpiece, bestquality, highlydetailed, ultradetailed, beautiful house, mountain, snow top
```

![image.png](../../resources/2.gif)

2.5M子数据集训练的生成效果仍存在效果不稳定的情况，开发者使用10M数据集效果会更稳定。

### 运行命令

```shell
# 该文件在swift/examples/pytorch/animatediff/scripts/full中
# Experimental environment: A100 * 4
# 200GB GPU memory totally
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 animatediff_sft.py \
  --model_id_or_path wyj123456/Realistic_Vision_V5.1_noVAE \
  --csv_path /mnt/workspace/yzhao/tastelikefeet/webvid/results_2M_train.csv \
  --video_folder /mnt/workspace/yzhao/tastelikefeet/webvid/videos2 \
  --sft_type full \
  --lr_scheduler_type constant \
  --trainable_modules .*motion_modules.* \
  --batch_size 4 \
  --eval_steps 100 \
  --gradient_accumulation_steps 16
```

我们使用了A100 * 4进行训练，共需要200GB显存，训练时长约40小时。数据格式如下：

```text
--csv_path 传入一个csv文件，该csv文件应包含如下格式：
name,contentUrl
Travel blogger shoot a story on top of mountains. young man holds camera in forest.,stock-footage-travel-blogger-shoot-a-story-on-top-of-mountains-young-man-holds-camera-in-forest.mp4
```

name字段代表该短视频的prompt，contentUrl代表该视频文件的名称

```text
--video_folder 传入一个视频目录，该目录中包含了csv文件中，contentUrl指代的所有视频文件
```

使用全参数进行推理方式如下：

```shell
# 该文件在swift/examples/pytorch/animatediff/scripts/full中
# Experimental environment: A100
# 18GB GPU memory
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python animatediff_infer.py \
  --model_id_or_path wyj123456/Realistic_Vision_V5.1_noVAE \
  --sft_type full \
  --ckpt_dir /output/path/like/checkpoints/iter-xxx \
  --eval_human true
```

其中的--ckpt_dir 传入训练时输出的文件夹即可。

## LoRA训练

### 运行命令

全参数训练会从0开始训练整个Motion-Adapter结构，用户可以使用现有的模型使用少量视频进行微调，只需要运行下面的命令：

```shell
# 该文件在swift/examples/pytorch/animatediff/scripts/lora中
# Experimental environment: A100
# 20GB GPU memory
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python animatediff_sft.py \
  --model_id_or_path wyj123456/Realistic_Vision_V5.1_noVAE \
  --csv_path /mnt/workspace/yzhao/tastelikefeet/webvid/results_2M_train.csv \
  --video_folder /mnt/workspace/yzhao/tastelikefeet/webvid/videos2 \
  --motion_adapter_id_or_path Shanghai_AI_Laboratory/animatediff-motion-adapter-v1-5-2 \
  --sft_type lora \
  --lr_scheduler_type constant \
  --trainable_modules .*motion_modules.* \
  --batch_size 1 \
  --eval_steps 200 \
  --dataset_sample_size 10000 \
  --gradient_accumulation_steps 16
```

视频数据参数同上。

推理命令如下：

```shell
# 该文件在swift/examples/pytorch/animatediff/scripts/lora中
# Experimental environment: A100
# 18GB GPU memory
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python animatediff_infer.py \
  --model_id_or_path wyj123456/Realistic_Vision_V5.1_noVAE \
  --motion_adapter_id_or_path Shanghai_AI_Laboratory/animatediff-motion-adapter-v1-5-2 \
  --sft_type lora \
  --ckpt_dir /output/path/like/checkpoints/iter-xxx \
  --eval_human true
```

其中的--ckpt_dir 传入训练时输出的文件夹即可。

## 参数列表

下面给出训练和推理分别支持的参数列表及其含义：

### 训练参数

```text
motion_adapter_id_or_path: Optional[str] = None # motion adapter的模型id或模型路径，指定这个参数可以基于现有的官方模型效果继续训练
motion_adapter_revision: Optional[str] = None # motion adapter的模型revision，仅在motion_adapter_id_or_path是模型id时有用

model_id_or_path: str = None # sd基模型的模型id或模型路径
model_revision: str = None # sd基模型的revision，仅在model_id_or_path是模型id时有用

dataset_sample_size: int = None # 数据集训练条数，默认代表全量训练

sft_type: str = field(
    default='lora', metadata={'choices': ['lora', 'full']}) # 训练方式，支持lora和全参数

output_dir: str = 'output' # 输出文件夹
ddp_backend: str = field(
    default='nccl', metadata={'choices': ['nccl', 'gloo', 'mpi', 'ccl']}) # 如使用ddp训练，ddp backend

seed: int = 42 # 随机种子

lora_rank: int = 8 # lora 参数
lora_alpha: int = 32 # lora 参数
lora_dropout_p: float = 0.05 # lora 参数
lora_dtype: str = 'fp32' # lora模块dtype类型，如果为`AUTO`则跟随原始模块的dtype设定

gradient_checkpointing: bool = False # 是否开启gc，默认不开启。注：当前版本diffusers有问题，不支持该参数为True
batch_size: int = 1 # batchsize
num_train_epochs: int = 1 # epoch数
# if max_steps >= 0, override num_train_epochs
learning_rate: Optional[float] = None # 学习率
weight_decay: float = 0.01 # adamw参数
gradient_accumulation_steps: int = 16 # ga大小
max_grad_norm: float = 1. # grad norm大小
lr_scheduler_type: str = 'cosine' # lr_scheduler的类型
warmup_ratio: float = 0.05 # 是否warmup及warmup占比

eval_steps: int = 50 # eval step间隔
save_steps: Optional[int] = None # save step间隔
dataloader_num_workers: int = 1 # dataloader workers数量

push_to_hub: bool = False # 是否推送到modelhub
# 'user_name/repo_name' or 'repo_name'
hub_model_id: Optional[str] = None # modelhub id
hub_private_repo: bool = False
push_hub_strategy: str = field( # 推送策略，推送最后一个还是每个都推送
    default='push_best',
    metadata={'choices': ['push_last', 'all_checkpoints']})
# None: use env var `MODELSCOPE_API_TOKEN`
hub_token: Optional[str] = field( # modelhub的token
    default=None,
    metadata={
        'help':
        'SDK token can be found in https://modelscope.cn/my/myaccesstoken'
    })

ignore_args_error: bool = False  # True: notebook compatibility

text_dropout_rate: float = 0.1 # drop一定比例的文本保证模型鲁棒性

validation_prompts_path: str = field( # 评测过程使用的prompt文件目录，默认使用swift/aigc/configs/validation.txt
    default=None,
    metadata={
        'help':
        'The validation prompts file path, use aigc/configs/validation.txt is None'
    })

trainable_modules: str = field( # 可训练模块，建议使用默认值
    default='.*motion_modules.*',
    metadata={
        'help':
        'The trainable modules, by default, the .*motion_modules.* will be trained'
    })

mixed_precision: bool = True # 混合精度训练

enable_xformers_memory_efficient_attention: bool = True # 使用xformers

num_inference_steps: int = 25 #
guidance_scale: float = 8.
sample_size: int = 256
sample_stride: int = 4 # 训练视频最大长度秒数
sample_n_frames: int = 16 # 每秒帧数

csv_path: str = None # 输入数据集
video_folder: str = None # 输入数据集

motion_num_attention_heads: int = 8 # motion adapter参数
motion_max_seq_length: int = 32 # motion adapter参数
num_train_timesteps: int = 1000 # 推理pipeline参数
beta_start: int = 0.00085 # 推理pipeline参数
beta_end: int = 0.012 # 推理pipeline参数
beta_schedule: str = 'linear' # 推理pipeline参数
steps_offset: int = 1 # 推理pipeline参数
clip_sample: bool = False # 推理pipeline参数

use_wandb: bool = False # 是否使用wandb
```

### 推理参数

```text
motion_adapter_id_or_path: Optional[str] = None # motion adapter的模型id或模型路径，指定这个参数可以基于现有的官方模型效果继续训练
motion_adapter_revision: Optional[str] = None # motion adapter的模型revision，仅在motion_adapter_id_or_path是模型id时有用

model_id_or_path: str = None # sd基模型的模型id或模型路径
model_revision: str = None # sd基模型的revision，仅在model_id_or_path是模型id时有用

sft_type: str = field(
    default='lora', metadata={'choices': ['lora', 'full']}) # 训练方式，支持lora和全参数

ckpt_dir: Optional[str] = field(
    default=None, metadata={'help': '/path/to/your/vx-xxx/checkpoint-xxx'}) # 训练的输出文件夹
eval_human: bool = False  # False: eval val_dataset # 是否使用人工输入评测

seed: int = 42 # 随机种子

merge_lora: bool = False # Merge lora into the MotionAdapter and save the model.
replace_if_exists: bool = False # Replace the files if the output merged dir exists when `merge_lora` is True.

# other
ignore_args_error: bool = False  # True: notebook compatibility

validation_prompts_path: str = None # 用于validation的文件，eval_human=False时使用，每一行一个prompt

output_path: str = './generated' # 输出gif的目录

enable_xformers_memory_efficient_attention: bool = True # 使用xformers

num_inference_steps: int = 25 #
guidance_scale: float = 8.
sample_size: int = 256
sample_stride: int = 4 # 训练视频最大长度秒数
sample_n_frames: int = 16 # 每秒帧数

motion_num_attention_heads: int = 8 # motion adapter参数
motion_max_seq_length: int = 32 # motion adapter参数
num_train_timesteps: int = 1000 # 推理pipeline参数
beta_start: int = 0.00085 # 推理pipeline参数
beta_end: int = 0.012 # 推理pipeline参数
beta_schedule: str = 'linear' # 推理pipeline参数
steps_offset: int = 1 # 推理pipeline参数
clip_sample: bool = False # 推理pipeline参数

```
