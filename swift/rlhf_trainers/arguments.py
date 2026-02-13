# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass
from typing import Optional

from transformers.utils.versions import require_version
from trl import CPOConfig as HfCPOConfig
from trl import DPOConfig as HfDPOConfig
from trl import GKDConfig as HfGKDConfig
from trl import GRPOConfig as HfGRPOConfig
from trl import KTOConfig as HfKTOConfig
from trl import ORPOConfig as HfORPOConfig
from trl import PPOConfig as HfPPOConfig
from trl import RewardConfig as HfRewardConfig

from swift.trainers import TrainArgumentsMixin
from .args_mixin import GRPOArgumentsMixin, RolloutTrainerArgumentsMixin


@dataclass
class DPOConfig(TrainArgumentsMixin, HfDPOConfig):
    ld_alpha: Optional[float] = None  # compat trl==0.15

    def __post_init__(self):
        TrainArgumentsMixin.__post_init__(self)
        HfDPOConfig.__post_init__(self)


@dataclass
class CPOConfig(TrainArgumentsMixin, HfCPOConfig):

    def __post_init__(self):
        TrainArgumentsMixin.__post_init__(self)
        HfCPOConfig.__post_init__(self)


@dataclass
class ORPOConfig(TrainArgumentsMixin, HfORPOConfig):

    def __post_init__(self):
        TrainArgumentsMixin.__post_init__(self)
        HfORPOConfig.__post_init__(self)


@dataclass
class KTOConfig(TrainArgumentsMixin, HfKTOConfig):

    def __post_init__(self):
        TrainArgumentsMixin.__post_init__(self)
        HfKTOConfig.__post_init__(self)


@dataclass
class RewardConfig(TrainArgumentsMixin, HfRewardConfig):

    def __post_init__(self):
        TrainArgumentsMixin.__post_init__(self)
        HfRewardConfig.__post_init__(self)


@dataclass
class PPOConfig(TrainArgumentsMixin, HfPPOConfig):

    def __post_init__(self):
        TrainArgumentsMixin.__post_init__(self)
        HfPPOConfig.__post_init__(self)


@dataclass
class GKDConfig(RolloutTrainerArgumentsMixin, TrainArgumentsMixin, HfGKDConfig):
    sft_alpha: float = 0

    offload_teacher_model: bool = False
    max_completion_length: int = 512
    log_completions: bool = False

    def __post_init__(self):
        RolloutTrainerArgumentsMixin.__post_init__(self)
        TrainArgumentsMixin.__post_init__(self)
        HfGKDConfig.__post_init__(self)


@dataclass
class GRPOConfig(GRPOArgumentsMixin, TrainArgumentsMixin, HfGRPOConfig):

    def __post_init__(self):
        require_version('trl>=0.20')
        GRPOArgumentsMixin.__post_init__(self)
        TrainArgumentsMixin.__post_init__(self)
        HfGRPOConfig.__post_init__(self)
        if self.vllm_reasoning_parser is not None:
            raise ValueError('vllm_reasoning_parser is not supported for GRPO Training, please unset it.')

        if self.cosine_max_len is None:
            self.cosine_max_len = self.max_completion_length

        if self.deepspeed and 'zero_optimization' in self.deepspeed and self.deepspeed['zero_optimization'][
                'stage'] == 3:
            # https://github.com/modelscope/ms-swift/issues/3237
            self.deepspeed['zero_optimization']['stage3_prefetch_bucket_size'] = 0
            self.deepspeed_plugin.hf_ds_config.config['zero_optimization']['stage3_prefetch_bucket_size'] = 0

        # https://github.com/modelscope/ms-swift/issues/3863
        self.dataloader_drop_last = True

        self.check_num_generations()

    def check_num_generations(self):
        # check num_generations for trl < 0.18
        num_processes = self.world_size

        # Check if the effective batch size can be divided by the number of generations
        if self.num_generations < 2:
            raise ValueError(
                'GRPO requires at least 2 generations per prompt to calculate the advantages. You provided '
                f'{self.num_generations}, which is less than the minimum required.')
        possible_values = [
            n_gen for n_gen in range(2, self.generation_batch_size + 1) if (self.generation_batch_size) % n_gen == 0
        ]

        if self.num_generations not in possible_values:
            raise ValueError(
                f'The effective train batch size ({num_processes} x {self.per_device_train_batch_size} x '
                f'{self.steps_per_generation}) must be evenly divisible by the number of generations per '
                f'prompt ({self.num_generations}). Given the current effective train batch size, the valid values for '
                f'the number of generations are: {possible_values}.')
        if self.eval_strategy != 'no':
            # Use num_generations_eval if set, otherwise fall back to num_generations
            num_generations_eval = self.num_generations_eval or self.num_generations
            global_eval_batch_size = self.per_device_eval_batch_size * num_processes
            possible_values = [
                n_gen for n_gen in range(1, global_eval_batch_size + 1) if (global_eval_batch_size) % n_gen == 0
            ]
            if num_generations_eval not in possible_values:
                raise ValueError(
                    f'The global eval batch size ({num_processes} x {self.per_device_eval_batch_size}) must be '
                    f'evenly divisible by the number of generations for eval ({num_generations_eval}). Given the '
                    'current global eval batch size, the valid values for the number of generations are: '
                    f'{possible_values}.')
