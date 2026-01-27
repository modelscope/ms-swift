# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .dpo_trainer import MegatronDPOTrainer
    from .gkd_trainer import MegatronGKDTrainer
    from .grpo_trainer import MegatronGRPOTrainer
    from .kto_trainer import MegatronKTOTrainer
    from .reward_trainer import MegatronRewardTrainer
    from .rollout_mixin import MegatronRolloutMixin
    from .embedding_trainer import MegatronEmbeddingTrainer
    from .reranker_trainer import MegatronRerankerTrainer
    from .trainer import MegatronTrainer
else:
    _import_structure = {
        'dpo_trainer': ['MegatronDPOTrainer'],
        'gkd_trainer': ['MegatronGKDTrainer'],
        'grpo_trainer': ['MegatronGRPOTrainer'],
        'kto_trainer': ['MegatronKTOTrainer'],
        'reward_trainer': ['MegatronRewardTrainer'],
        'rollout_mixin': ['MegatronRolloutMixin'],
        'embedding_trainer': ['MegatronEmbeddingTrainer'],
        'reranker_trainer': ['MegatronRerankerTrainer'],
        'trainer': ['MegatronTrainer'],
    }
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
