# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from torch.optim.lr_scheduler import LambdaLR


class _HFSchedulerAdapter(LambdaLR):
    """Build ``hf_name``'s schedule via HF's get_scheduler and drive it as a plain LambdaLR.

    HF ships factory FUNCTIONS, and neither a function nor an instance can cross set_lr_scheduler
    (the optimizer lives on the Ray worker); module-level classes pickle by reference into it.
    """

    hf_name: str = ''

    def __init__(self, optimizer, num_warmup_steps: int, num_training_steps: int, **scheduler_specific_kwargs):
        from transformers import get_scheduler
        inner = get_scheduler(
            self.hf_name,
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=scheduler_specific_kwargs or None)
        # Every schedule below is LambdaLR-based, so adopting the lambda reproduces HF exactly;
        # re-running LambdaLR.__init__ on the same optimizer is safe (initial_lr stored first pass).
        super().__init__(optimizer, inner.lr_lambdas[0])


class CosineWithMinLRScheduler(_HFSchedulerAdapter):
    """Cosine decay to a floor: ``--lr_scheduler cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr": ...}'``."""

    hf_name = 'cosine_with_min_lr'


class ConstantWithWarmupScheduler(_HFSchedulerAdapter):
    """Warm up, then hold (twinkle's LinearWarmupScheduler decays to 0 -- the opposite)."""

    hf_name = 'constant_with_warmup'


class CosineWithRestartsScheduler(_HFSchedulerAdapter):
    hf_name = 'cosine_with_restarts'


class PolynomialDecayScheduler(_HFSchedulerAdapter):
    hf_name = 'polynomial'


class InverseSqrtScheduler(_HFSchedulerAdapter):
    """Megatron spells this 'inverse-square-root' and has it natively; this is the HF-side twin."""

    hf_name = 'inverse_sqrt'
