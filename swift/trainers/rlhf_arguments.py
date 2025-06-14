from dataclasses import dataclass, field
from typing import List

from trl import CPOConfig as HfCPOConfig
from trl import DPOConfig as HfDPOConfig
from trl import GKDConfig as HfGKDConfig
from trl import GRPOConfig as HfGRPOConfig
from trl import KTOConfig as HfKTOConfig
from trl import ORPOConfig as HfORPOConfig
from trl import PPOConfig as HfPPOConfig
from trl import RewardConfig as HfRewardConfig

from .arguments import GRPOArgumentsMixin, SwiftArgumentsMixin


@dataclass
class DPOConfig(SwiftArgumentsMixin, HfDPOConfig):
    pass


@dataclass
class CPOConfig(SwiftArgumentsMixin, HfCPOConfig):
    pass


@dataclass
class ORPOConfig(SwiftArgumentsMixin, HfORPOConfig):
    pass


@dataclass
class KTOConfig(SwiftArgumentsMixin, HfKTOConfig):
    pass


@dataclass
class RewardConfig(SwiftArgumentsMixin, HfRewardConfig):
    pass


@dataclass
class PPOConfig(SwiftArgumentsMixin, HfPPOConfig):
    pass


@dataclass
class GKDConfig(SwiftArgumentsMixin, HfGKDConfig):
    pass


@dataclass
class GRPOConfig(GRPOArgumentsMixin, SwiftArgumentsMixin, HfGRPOConfig):
    stop_words: List[str] = field(default_factory=list)

    def __post_init__(self):
        from swift.llm.argument.base_args.model_args import ModelArguments
        super().__post_init__()
        if self.cosine_max_len is None:
            self.cosine_max_len = self.max_completion_length
        self.vllm_limit_mm_per_prompt = ModelArguments.parse_to_dict(self.vllm_limit_mm_per_prompt)

        if self.deepspeed and 'zero_optimization' in self.deepspeed and self.deepspeed['zero_optimization'][
                'stage'] == 3:
            # https://github.com/modelscope/ms-swift/issues/3237
            self.deepspeed['zero_optimization']['stage3_prefetch_bucket_size'] = 0
            self.deepspeed_plugin.hf_ds_config.config['zero_optimization']['stage3_prefetch_bucket_size'] = 0

        # https://github.com/modelscope/ms-swift/issues/3863
        self.dataloader_drop_last = True

        num_processes = self.world_size
        if self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
        if self.generation_batch_size is None:
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation

        self.check_num_generations()

    def check_num_generations(self):
        # check num_generations for trl < 0.18
        num_processes = self.world_size

        if self.generation_batch_size % self.per_device_train_batch_size * num_processes != 0:
            raise ValueError(
                f'generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size '
                f'({self.per_device_train_batch_size * num_processes}).')

        self.steps_per_generation = self.generation_batch_size // (self.per_device_train_batch_size * num_processes)

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
            global_eval_batch_size = self.per_device_eval_batch_size * num_processes
            possible_values = [
                n_gen for n_gen in range(2, global_eval_batch_size + 1) if (global_eval_batch_size) % n_gen == 0
            ]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f'The global eval batch size ({num_processes} x {self.per_device_eval_batch_size}) must be '
                    f'evenly divisible by the number of generations per prompt ({self.num_generations}). Given the '
                    'current global eval batch size, the valid values for the number of generations are: '
                    f'{possible_values}.')
