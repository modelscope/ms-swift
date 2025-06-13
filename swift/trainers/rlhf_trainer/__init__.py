# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .cpo_trainer import CPOTrainer
    from .dpo_trainer import DPOTrainer
    from .grpo_trainer import GRPOTrainer
    from .kto_trainer import KTOTrainer
    from .orpo_trainer import ORPOTrainer
    from .ppo_trainer import PPOTrainer
    from .reward_trainer import RewardTrainer
    from .gkd_trainer import GKDTrainer
    from .rlhf_mixin import RLHFTrainerMixin
    from .utils import patch_lora_merge, patch_lora_unmerge, round_robin, _ForwardRedirection
else:
    _import_structure = {
        'cpo_trainer': ['CPOTrainer'],
        'dpo_trainer': ['DPOTrainer'],
        'grpo_trainer': ['GRPOTrainer'],
        'kto_trainer': ['KTOTrainer'],
        'orpo_trainer': ['ORPOTrainer'],
        'ppo_trainer': ['PPOTrainer'],
        'reward_trainer': ['RewardTrainer'],
        'gkd_trainer': ['GKDTrainer'],
        'rlhf_mixin': ['RLHFTrainerMixin'],
        'utils': ['patch_lora_merge', 'patch_lora_unmerge', 'round_robin', '_ForwardRedirection'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
