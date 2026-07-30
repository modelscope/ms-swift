"""HF lr_schedulers exposed as classes, so twinkle can construct them.

twinkle's contract is torch's ``LRScheduler`` -- nothing twinkle-specific -- and HF's schedulers
already satisfy it (``get_scheduler`` returns a ``LambdaLR``). What HF does not ship is a
constructible CLASS: ``transformers.optimization`` only exposes factory FUNCTIONS, and neither a
function nor an instance can be handed to ``set_lr_scheduler``:

  - an instance would have to be built against the optimizer, which lives on the Ray worker
    (``set_lr_scheduler`` is a ``@remote_function``), not on the driver that calls it;
  - a function falls through ``construct_class``'s last branch (``else: return func``), so it would
    be assigned as the scheduler and silently never step.

Hence these thin ``LambdaLR`` subclasses. The lr math is still HF's -- each one asks
``get_scheduler`` for the schedule and adopts its lambda -- so there is no second implementation to
drift. They must stay module-level classes (not generated in a factory) to survive being pickled by
reference into a Ray worker.

Only the Transformers backend uses these. Megatron accepts nothing but its own
``OptimizerParamScheduler`` (which also drives the weight-decay schedule), so its schedule names are
mapped to ``lr_decay_style`` instead; see naming._MEGATRON_DECAY_STYLE_MAP.
"""
from __future__ import annotations

from torch.optim.lr_scheduler import LambdaLR


class _HFSchedulerAdapter(LambdaLR):
    """Base: build ``hf_name``'s schedule via HF and drive it as a plain LambdaLR.

    ``**scheduler_specific_kwargs`` carries the per-schedule extras HF expects (e.g. min_lr for
    cosine_with_min_lr); dev fills it from TrainConfig.lr_scheduler_kwargs, matching legacy swift.
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
        # Every schedule below is LambdaLR-based, so adopting the lambda reproduces HF exactly.
        # Re-running LambdaLR.__init__ on the same optimizer is safe: the first pass stores
        # initial_lr on each param group, and the second reads that stored value rather than the
        # already-scaled lr.
        super().__init__(optimizer, inner.lr_lambdas[0])


class CosineWithMinLRScheduler(_HFSchedulerAdapter):
    """Cosine decay to a floor. The combination swift's own docs recommend:
    ``--lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr": 1e-6}'``."""

    hf_name = 'cosine_with_min_lr'


class ConstantWithWarmupScheduler(_HFSchedulerAdapter):
    """Warm up, then hold. Previously mis-mapped to twinkle's LinearWarmupScheduler, which decays
    to 0 after warmup -- the opposite of what the name promises."""

    hf_name = 'constant_with_warmup'


class CosineWithRestartsScheduler(_HFSchedulerAdapter):
    hf_name = 'cosine_with_restarts'


class PolynomialDecayScheduler(_HFSchedulerAdapter):
    hf_name = 'polynomial'


class InverseSqrtScheduler(_HFSchedulerAdapter):
    """Megatron spells this 'inverse-square-root' and has it natively; this is the HF-side twin."""

    hf_name = 'inverse_sqrt'
