# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import torch
import torch.distributed as dist

from swift import get_logger
from swift.utils import broadcast_string, get_dist_setting, is_dist

logger = get_logger()


@dataclass
class AnimateDiffArguments:
    motion_adapter_id_or_path: Optional[str] = None
    motion_adapter_revision: Optional[str] = None

    model_id_or_path: str = None
    model_revision: str = None

    dataset_sample_size: int = None

    sft_type: str = field(default='lora', metadata={'choices': ['lora', 'full']})

    output_dir: str = 'output'
    ddp_backend: str = field(default='nccl', metadata={'choices': ['nccl', 'gloo', 'mpi', 'ccl', 'hccl']})

    seed: int = 42

    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout_p: float = 0.05
    lora_dtype: Literal['fp16', 'bf16', 'fp32', 'AUTO'] = 'fp32'

    gradient_checkpointing: bool = False
    batch_size: int = 1
    num_train_epochs: int = 1
    # if max_steps >= 0, override num_train_epochs
    max_steps: int = -1
    learning_rate: Optional[float] = None
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 16
    max_grad_norm: float = 1.
    lr_scheduler_type: str = 'cosine'
    warmup_ratio: float = 0.05

    eval_steps: int = 50
    save_steps: Optional[int] = None
    dataloader_num_workers: int = 1

    push_to_hub: bool = False
    # 'user_name/repo_name' or 'repo_name'
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = False
    push_hub_strategy: str = field(default='push_best', metadata={'choices': ['push_last', 'all_checkpoints']})
    # None: use env var `MODELSCOPE_API_TOKEN`
    hub_token: Optional[str] = field(
        default=None, metadata={'help': 'SDK token can be found in https://modelscope.cn/my/myaccesstoken'})

    ignore_args_error: bool = False  # True: notebook compatibility

    text_dropout_rate: float = 0.1

    validation_prompts_path: str = field(
        default=None, metadata={'help': 'The validation prompts file path, use llm/configs/ad_validation.txt is None'})

    trainable_modules: str = field(
        default='.*motion_modules.*',
        metadata={'help': 'The trainable modules, by default, the .*motion_modules.* will be trained'})

    mixed_precision: bool = True

    enable_xformers_memory_efficient_attention: bool = True

    num_inference_steps: int = 25
    guidance_scale: float = 8.
    sample_size: int = 256
    sample_stride: int = 4
    sample_n_frames: int = 16

    csv_path: str = None
    video_folder: str = None

    motion_num_attention_heads: int = 8
    motion_max_seq_length: int = 32
    num_train_timesteps: int = 1000
    beta_start: int = 0.00085
    beta_end: int = 0.012
    beta_schedule: str = 'linear'
    steps_offset: int = 1
    clip_sample: bool = False

    use_wandb: bool = False

    def __post_init__(self) -> None:
        handle_compatibility(self)

        current_dir = os.path.dirname(__file__)
        if self.validation_prompts_path is None:
            self.validation_prompts_path = os.path.join(current_dir, 'configs/animatediff', 'validation.txt')
        if self.learning_rate is None:
            self.learning_rate = 1e-4
        if self.save_steps is None:
            self.save_steps = self.eval_steps

        if is_dist():
            rank, local_rank, _, _ = get_dist_setting()
            torch.cuda.set_device(local_rank)
            self.seed += rank  # Avoid the same dropout
            # Initialize in advance
            if not dist.is_initialized():
                dist.init_process_group(backend=self.ddp_backend)
            # Make sure to set the same output_dir when using DDP.
            self.output_dir = broadcast_string(self.output_dir)


@dataclass
class AnimateDiffInferArguments:

    motion_adapter_id_or_path: Optional[str] = None
    motion_adapter_revision: Optional[str] = None

    model_id_or_path: str = None
    model_revision: str = None

    sft_type: str = field(default='lora', metadata={'choices': ['lora', 'full']})

    ckpt_dir: Optional[str] = field(default=None, metadata={'help': '/path/to/your/vx-xxx/checkpoint-xxx'})
    eval_human: bool = False  # False: eval val_dataset

    seed: int = 42

    # other
    ignore_args_error: bool = False  # True: notebook compatibility

    validation_prompts_path: str = None

    output_path: str = './generated'

    enable_xformers_memory_efficient_attention: bool = True

    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    sample_size: int = 256
    sample_stride: int = 4
    sample_n_frames: int = 16

    motion_num_attention_heads: int = 8
    motion_max_seq_length: int = 32
    num_train_timesteps: int = 1000
    beta_start: int = 0.00085
    beta_end: int = 0.012
    beta_schedule: str = 'linear'
    steps_offset: int = 1
    clip_sample: bool = False

    merge_lora: bool = False
    replace_if_exists: bool = False

    # compatibility. (Deprecated)
    merge_lora_and_save: Optional[bool] = None

    def __post_init__(self) -> None:
        handle_compatibility(self)


def handle_compatibility(args: Union[AnimateDiffArguments, AnimateDiffInferArguments]) -> None:
    if isinstance(args, AnimateDiffInferArguments):
        if args.merge_lora_and_save is not None:
            args.merge_lora = args.merge_lora_and_save
