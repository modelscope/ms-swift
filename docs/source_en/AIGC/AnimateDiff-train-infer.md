# AnimateDiff Fine-tuning and Inference

SWIFT supports fine-tuning and inference of AnimateDiff of full parameter and LoRA fine-tuning.

First, you need to clone and install SWIFT:

```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install ".[aigc]"
```

## Full Parameter Training

### Training Effect

Full parameter fine-tuning can reproduce the effect of the [officially provided model animatediff-motion-adapter-v1-5-2](https://www.modelscope.cn/models/Shanghai_AI_Laboratory/animatediff-motion-adapter-v1-5-2/summary), requiring a large number of short videos. The official reproduction used a subset version of the official dataset: [WebVid 2.5M](https://maxbain.com/webvid-dataset/). The training effect is as follows:

```text
Prompt:masterpiece, bestquality, highlydetailed, ultradetailed, girl, walking, on the street, flowers
```

![image.png](../../resources/1.gif)

```text
Prompt: masterpiece, bestquality, highlydetailed, ultradetailed, beautiful house, mountain, snow top```
```

![image.png](../../resources/2.gif)

The generation effect of training with the 2.5M subset still has unstable results. Developers using the 10M dataset will have more stable effects.

### Running Command

```shell
# This file is in swift/examples/pytorch/animatediff/scripts/full
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

We used A100 * 4 for training, requiring a total of 200GB GPU memory, and the training time is about 40 hours. The data format is as follows:


```text
--csv_path # Pass in a csv file, which should contain the following format:
name,contentUrl
Travel blogger shoot a story on top of mountains. young man holds camera in forest.,stock-footage-travel-blogger-shoot-a-story-on-top-of-mountains-young-man-holds-camera-in-forest.mp4
```

The name field represents the prompt of the short video, and contentUrl represents the name of the video file.

```text
--video_folder Pass in a video directory containing all the video files referenced by contentUrl in the csv file.
```

To perform inference using full parameters:
```shell
# This file is in swift/examples/pytorch/animatediff/scripts/full
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

The --ckpt_dir should be the output folder from training.

## LoRA Training

### Running Command

Full parameter training will train the entire Motion-Adapter structure from scratch. Users can use an existing model and a small number of videos for fine-tuning by running the following command:
```shell
# This file is in swift/examples/pytorch/animatediff/scripts/lora
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

Video data parameters are the same as above.

The inference command is as follows:
```shell
# This file is in swift/examples/pytorch/animatediff/scripts/lora
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

The --ckpt_dir should be the output folder from training.

## Parameter List

Below are the supported parameter lists and their meanings for training and inference respectively:

### Training Parameters
```text
motion_adapter_id_or_path: Optional[str] = None # The model ID or model path of the motion adapter. Specifying this parameter allows for continued training based on the effect of existing official models.
motion_adapter_revision: Optional[str] = None # The model revision of the motion adapter, only useful when motion_adapter_id_or_path is the model ID.

model_id_or_path: str = None # The model ID or model path of the SD base model.
model_revision: str = None # The revision of the SD base model, only useful when model_id_or_path is the model ID.

dataset_sample_size: int = None # The number of training samples in the dataset. Default represents full training.

sft_type: str = field(
    default='lora', metadata={'choices': ['lora', 'full']}) # Training method, supporting lora and full parameters.

output_dir: str = 'output' # Output folder.
ddp_backend: str = field(
    default='nccl', metadata={'choices': ['nccl', 'gloo', 'mpi', 'ccl']}) # If using ddp training, ddp backend.

seed: int = 42 # Random seed.

lora_rank: int = 8 # lora parameter.
lora_alpha: int = 32 # lora parameter.
lora_dropout_p: float = 0.05 # lora parameter.
lora_dtype: str = 'fp32' # lora module dtype type. If `AUTO`, it follows the dtype setting of the original module.

gradient_checkpointing: bool = False # Whether to enable gc, disabled by default. Note: The current version of diffusers has a problem and does not support this parameter being True.
batch_size: int = 1 # batchsize.
num_train_epochs: int = 1 # Number of epochs.
# if max_steps >= 0, override num_train_epochs
learning_rate: Optional[float] = None # Learning rate.
weight_decay: float = 0.01 # adamw parameter.
gradient_accumulation_steps: int = 16 # ga size.
max_grad_norm: float = 1. # grad norm size.
lr_scheduler_type: str = 'cosine' # Type of lr_scheduler.
warmup_ratio: float = 0.05 # Whether to warmup and the proportion of warmup.

eval_steps: int = 50 # eval step interval.
save_steps: Optional[int] = None # save step interval.
dataloader_num_workers: int = 1 # Number of dataloader workers.

push_to_hub: bool = False # Whether to push to modelhub.
# 'user_name/repo_name' or 'repo_name'
hub_model_id: Optional[str] = None # modelhub id.
hub_private_repo: bool = False
push_hub_strategy: str = field( # Push strategy, push the last one or push each one.
    default='push_best',
    metadata={'choices': ['push_last', 'all_checkpoints']})
# None: use env var `MODELSCOPE_API_TOKEN`
hub_token: Optional[str] = field( # modelhub token.
    default=None,
    metadata={
        'help':
        'SDK token can be found in https://modelscope.cn/my/myaccesstoken'
    })

ignore_args_error: bool = False  # True: notebook compatibility.

text_dropout_rate: float = 0.1 # Drop a certain proportion of text to ensure model robustness.

validation_prompts_path: str = field( # The prompt file directory used in the evaluation process. By default, swift/aigc/configs/validation.txt is used.
    default=None,
    metadata={
        'help':
        'The validation prompts file path, use aigc/configs/validation.txt is None'
    })

trainable_modules: str = field( # Trainable modules, recommended to use the default value.
    default='.*motion_modules.*',
    metadata={
        'help':
        'The trainable modules, by default, the .*motion_modules.* will be trained'
    })

mixed_precision: bool = True # Mixed precision training.

enable_xformers_memory_efficient_attention: bool = True # Use xformers.

num_inference_steps: int = 25 #
guidance_scale: float = 8.
sample_size: int = 256
sample_stride: int = 4 # Maximum length of training videos in seconds.
sample_n_frames: int = 16 # Frames per second.

csv_path: str = None # Input dataset.
video_folder: str = None # Input dataset.

motion_num_attention_heads: int = 8 # motion adapter parameter.
motion_max_seq_length: int = 32 # motion adapter parameter.
num_train_timesteps: int = 1000 # Inference pipeline parameter.
beta_start: int = 0.00085 # Inference pipeline parameter.
beta_end: int = 0.012 # Inference pipeline parameter.
beta_schedule: str = 'linear' # Inference pipeline parameter.
steps_offset: int = 1 # Inference pipeline parameter.
clip_sample: bool = False # Inference pipeline parameter.

use_wandb: bool = False # Whether to use wandb.
```

### Inference Parameters
```text
motion_adapter_id_or_path: Optional[str] = None # The model ID or model path of the motion adapter. Specifying this parameter allows for continued training based on the effect of existing official models.
motion_adapter_revision: Optional[str] = None # The model revision of the motion adapter, only useful when motion_adapter_id_or_path is the model ID.

model_id_or_path: str = None # The model ID or model path of the SD base model.
model_revision: str = None # The revision of the SD base model, only useful when model_id_or_path is the model ID.

sft_type: str = field(
    default='lora', metadata={'choices': ['lora', 'full']}) # Training method, supporting lora and full parameters.

ckpt_dir: Optional[str] = field(
    default=None, metadata={'help': '/path/to/your/vx-xxx/checkpoint-xxx'}) # The output folder of training.
eval_human: bool = False  # False: eval val_dataset # Whether to use manual input evaluation.

seed: int = 42 # Random seed.

merge_lora: bool = False # Merge lora into the MotionAdapter and save the model.
replace_if_exists: bool = False # Replace the files if the output merged dir exists when `merge_lora` is True.

# other
ignore_args_error: bool = False  # True: notebook compatibility.

validation_prompts_path: str = None # The file used for validation. When eval_human=False, each line is a prompt.

output_path: str = './generated' # The output directory for gifs.

enable_xformers_memory_efficient_attention: bool = True # Use xformers.

num_inference_steps: int = 25 #
guidance_scale: float = 8.
sample_size: int = 256
sample_stride: int = 4 # Maximum length of training videos in seconds.
sample_n_frames: int = 16 # Frames per second.

motion_num_attention_heads: int = 8 # motion adapter parameter.
motion_max_seq_length: int = 32 # motion adapter parameter.
num_train_timesteps: int = 1000 # Inference pipeline parameter.
beta_start: int = 0.00085 # Inference pipeline parameter.
beta_end: int = 0.012 # Inference pipeline parameter.
beta_schedule: str = 'linear' # Inference pipeline parameter.
steps_offset: int = 1 # Inference pipeline parameter.
clip_sample: bool = False # Inference pipeline parameter.
```
