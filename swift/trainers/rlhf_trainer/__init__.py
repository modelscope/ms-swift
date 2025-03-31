# Copyright (c) Alibaba, Inc. and its affiliates.
from .cpo_trainer import CPOTrainer
from .dpo_trainer import DPOTrainer
from .grpo_trainer import GRPOTrainer
from .kto_trainer import KTOTrainer
from .orpo_trainer import ORPOTrainer
from .ppo_trainer import PPOTrainer
from .reward_trainer import RewardTrainer
from .rlhf_mixin import RLHFTrainerMixin
from .utils import patch_lora_merge, patch_lora_unmerge, round_robin
