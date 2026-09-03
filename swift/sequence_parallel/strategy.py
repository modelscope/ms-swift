# Copyright (c) ModelScope Contributors. All rights reserved.
"""Centralized sequence-parallel strategy (facade) for ms-swift.

``SPStrategy`` is the single entry point for all sequence-parallel wiring:
``initialize`` (mesh setup + model patching), ``preprocess_inputs`` (input
splitting before forward), ``gather_loss_tensors`` (loss aggregation) and
``postprocess_outputs`` (metrics gather). The actual algorithms
(pad/split/gather/ulysses) live in the ``sequence_parallel`` singleton; with
``device_mesh=None`` every method is byte-for-byte the legacy behavior.

An external mesh object may be passed to derive the (data, ring, sequence)
topology; it only needs three duck-typed attributes: ``ulysses_size``,
``data_world_size`` and optionally ``cp_world_size``.
"""
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

from swift.utils import get_current_device, get_logger
from .sequence_parallel import sequence_parallel
from .utils import GatherLoss

logger = get_logger()


@dataclass(frozen=True)
class SPConfig:
    """Sequence-parallel configuration.

    Attributes:
        enabled: Whether sequence parallelism is enabled.
        ulysses_size: Ulysses SP size. When an external mesh object is
            provided to ``SPStrategy``, this is derived from its
            ``ulysses_size`` attribute instead of being set manually.
        gather_logits: Whether to gather full-sequence logits across the SP
            group in ``postprocess_outputs`` (metrics/eval path).
    """
    enabled: bool = True
    ulysses_size: Optional[int] = None
    gather_logits: bool = True


def _derive_ulysses_size(device_mesh, sp_config: SPConfig) -> Optional[int]:
    """Resolve the effective ulysses size: the external mesh object wins over config."""
    if device_mesh is not None:
        mesh_sp = getattr(device_mesh, 'ulysses_size', None)
        if mesh_sp is not None:
            if sp_config.ulysses_size is not None and sp_config.ulysses_size != mesh_sp:
                logger.warning(f'SPConfig.ulysses_size={sp_config.ulysses_size} differs from '
                               f'device_mesh.ulysses_size={mesh_sp}; using the external mesh value.')
            return mesh_sp
    return sp_config.ulysses_size


class SPStrategy:
    """Facade that centralizes every sequence-parallel wiring point.

    Holds references only; the actual algorithm still lives in the
    ``sequence_parallel`` singleton (``swift.sequence_parallel.sequence_parallel``).
    The facade exposes the four wiring points used by the training loop:

    - ``initialize``: mesh setup + model patching, once per model.
    - ``preprocess_inputs``: input splitting before ``forward``.
    - ``gather_loss_tensors``: per-token loss aggregation after ``forward``.
    - ``postprocess_outputs``: preds/labels gather for metrics.
    """

    def __init__(self, device_mesh=None, sp_config: Optional[SPConfig] = None, model=None, tokenizer=None):
        self.device_mesh = device_mesh
        self.sp_config = sp_config or SPConfig()
        self.model = model
        self.tokenizer = tokenizer
        self.ulysses_size = _derive_ulysses_size(device_mesh, self.sp_config)
        self._initialized = False

    def initialize(self, sp_size: Optional[int] = None, model=None, tokenizer=None, padding_free: bool = False) -> bool:
        """Wire up SP: build the (data, ring, sequence) mesh and patch the model.

        Used by ``sft.py`` / ``rlhf.py`` model preparation. With
        ``device_mesh=None`` this is exactly the legacy
        ``sequence_parallel.prepare(sp_size, ...)`` path; with an external
        mesh object the (data, ring, sequence) sizes are derived from it
        instead. May be called once per model object (e.g. RLHF policy + ref)
        — the singleton internally guards global patches with
        ``_global_inited`` while hooks are registered per model. Returns True
        when SP was activated.
        """
        if not self.sp_config.enabled:
            return False
        model = model if model is not None else self.model
        tokenizer = tokenizer if tokenizer is not None else self.tokenizer
        if self.ulysses_size is not None:
            if sp_size is not None and sp_size != self.ulysses_size:
                logger.warning(f'sp_size={sp_size} differs from the strategy ulysses_size={self.ulysses_size}; '
                               'using the strategy value.')
            ulysses_size = self.ulysses_size
        else:
            ulysses_size = sp_size
        if ulysses_size is None or ulysses_size <= 1:
            return False
        sequence_parallel.prepare(
            ulysses_size, model=model, tokenizer=tokenizer, padding_free=padding_free, device_mesh=self.device_mesh)
        self._initialized = True
        return True

    def _check_initialized(self, method: str):
        if not self._initialized:
            raise RuntimeError(f'SPStrategy.{method}() called before initialize(); '
                               'SP is not wired up on this strategy instance.')

    @property
    def real_position_ids(self):
        """Full packed position ids cached by the last ``prepare_inputs`` run (read-only)."""
        return getattr(sequence_parallel, 'real_position_ids', None)

    @property
    def dp_rank(self):
        """Rank inside the data-parallel group (read-only)."""
        return getattr(sequence_parallel, 'dp_rank', None)

    @property
    def dp_group(self):
        """Data-parallel process group, or None when SP is not wired up (read-only)."""
        return getattr(sequence_parallel, 'dp_group', None)

    @property
    def world_size(self):
        """Total sequence-parallel world size (sp x ring) (read-only)."""
        return getattr(sequence_parallel, 'world_size', None)

    @property
    def rp_world_size(self):
        """Ring-parallel world size (read-only)."""
        return getattr(sequence_parallel, 'rp_world_size', None)

    def gather(self, tensor, dim: int, position_ids=None):
        """Gather a tensor across the SP group (reverse of split)."""
        self._check_initialized('gather')
        return sequence_parallel.gather(tensor, dim=dim, position_ids=position_ids)

    def pad(self, *args, **kwargs):
        """Pad a tensor to a multiple of the SP size; forwards to the singleton."""
        self._check_initialized('pad')
        return sequence_parallel.pad(*args, **kwargs)

    def pad_and_split_inputs(self, *args, **kwargs):
        """Pad and split input tensors across SP ranks; forwards to the singleton."""
        self._check_initialized('pad_and_split_inputs')
        return sequence_parallel.pad_and_split_inputs(*args, **kwargs)

    def gather_object_dp(self, input_data):
        """Gather a list of objects across the data-parallel group and flatten one level."""
        self._check_initialized('gather_object_dp')
        return sequence_parallel._gather_object_dp(input_data)

    def create_sp_sampler(self, dataset, shuffle: bool = True, seed=None, round_up: bool = True):
        """Build a ``SequenceParallelSampler`` from the current dp topology.

        The sampler only receives the data-parallel group, not the SP engine
        instance.
        """
        self._check_initialized('create_sp_sampler')
        from .utils import SequenceParallelSampler
        return SequenceParallelSampler(
            dataset,
            dp_group=sequence_parallel.device_mesh['data'].get_group(),
            shuffle=shuffle,
            seed=seed,
            round_up=round_up)

    def create_sp_dispatcher(self, dataloader, device=None, skip_batches: int = 0):
        """Build a ``SequenceParallelDispatcher`` from the current dp topology.

        The dispatcher only receives dp rank/world_size/group, not the SP
        engine instance.
        """
        self._check_initialized('create_sp_dispatcher')
        from .utils import SequenceParallelDispatcher
        return SequenceParallelDispatcher(
            dataloader,
            dp_rank=sequence_parallel.dp_rank,
            dp_world_size=sequence_parallel.dp_world_size,
            dp_group=sequence_parallel.dp_group,
            device=device,
            skip_batches=skip_batches)

    def preprocess_inputs(self, inputs: dict) -> dict:
        """Set extra_kwargs and split labels before forward.

        Used by ``trainer.py`` / ``seq2seq_trainer.py`` ``_prepare_inputs``.
        """
        self._check_initialized('preprocess_inputs')
        sequence_parallel.prepare_inputs(inputs)
        return inputs

    def postprocess_outputs(self, preds, labels):
        """Gather preds/labels across the SP group for metrics (used by
        ``mixin.py`` ``compute_metrics``), then roll labels back to fit
        ``compute_acc``."""
        self._check_initialized('postprocess_outputs')
        if not self.sp_config.gather_logits:
            return preds, labels
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds).to(get_current_device())
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).to(get_current_device())
        assert labels.shape[1] == preds.shape[1]

        if sequence_parallel.rp_world_size > 1:
            position_ids = sequence_parallel.real_position_ids
            position_ids = sequence_parallel.pad(position_ids, padding_value=-1, position_ids=position_ids)
        else:
            position_ids = None
        preds_output = sequence_parallel.gather(preds, dim=1, position_ids=position_ids)
        labels_output = sequence_parallel.gather(labels, dim=1, position_ids=position_ids)
        # Roll labels back (+1) to cancel the shift (-1) applied upstream, so that
        # callers comparing preds[i] against labels[i] (e.g. SwiftMixin._compute_acc)
        # see position-aligned tensors.
        labels_output = torch.roll(labels_output, shifts=1, dims=1)
        return preds_output, labels_output.int()

    def gather_loss_tensors(self, loss, labels, batch_size: int):
        """Gather per-token loss across the SP group.

        Used by the ``per_token_loss_func_sp`` tail in ``trainers/utils.py``.
        """
        self._check_initialized('gather_loss_tensors')
        position_ids = sequence_parallel.real_position_ids
        if position_ids is not None:
            position_ids = sequence_parallel.pad(position_ids, padding_value=-1, position_ids=position_ids)
        loss, labels = GatherLoss.apply(loss.reshape(batch_size, -1), labels.reshape(batch_size, -1), 1, position_ids)
        if position_ids is not None and position_ids.min() == -1:
            pos_mask = position_ids >= 0
            loss = loss[pos_mask].contiguous()
        return loss

    def wrap_model(self, model, optimizer=None):
        """Lifecycle hook for wrapping the model; reserved for later milestones."""
        raise NotImplementedError('SPStrategy.wrap_model is reserved for later milestones')

    def save_optimizer_checkpoint(self, *args, **kwargs):
        raise NotImplementedError('SPStrategy.save_optimizer_checkpoint is reserved for later milestones')

    def load_optimizer_checkpoint(self, *args, **kwargs):
        raise NotImplementedError('SPStrategy.load_optimizer_checkpoint is reserved for later milestones')


_sp_strategy: Optional[SPStrategy] = None


def get_sp_strategy() -> SPStrategy:
    """Process-wide lazy ``SPStrategy`` for the legacy CLI path.

    All mutable state lives in the ``sequence_parallel`` singleton, so this
    facade instance only carries config/mesh references; a process-wide
    instance is sufficient. Callers holding an external mesh object should
    construct their own ``SPStrategy(device_mesh=...)`` instead.
    """
    global _sp_strategy
    if _sp_strategy is None:
        _sp_strategy = SPStrategy()
    return _sp_strategy
