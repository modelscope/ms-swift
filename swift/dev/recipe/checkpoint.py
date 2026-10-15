"""The checkpoint half of a flat offline recipe: twinkle-native write, hub push, resume, args.json.

``CheckpointSaver`` is the thin re-home of the persistence the old ``OfflineLoop.save`` / ``TrainLoop.save``
/ ``resume`` and ``TrainAssembly.save_final`` carried: one twinkle-native write (optimizer.pt /
scheduler.pt / rng_state.pt / trainer_state.json), the ``on_save`` / ``on_push_begin`` callback fan-out,
the optional hub upload, and the resume that re-seeds the recorder counters + skips the dataloader to its
consumed offset. It owns no loop -- the recipe calls :meth:`save` at the ``save_steps`` cadence and
:meth:`save_final` at train end, and calls :meth:`resume` once before the loop when resuming.

The self-describing ``args.json`` ``swift infer`` reads back is written by the recipe's ``write_args_json``
callable (built from ``TrainAssembly.write_ckpt_args_json``), so this class stays free of config knowledge.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.model import TrainableModel
    from swift.dev.recipe.step_runner import StepRunner

logger = get_logger()

__all__ = ['CheckpointSaver']


class CheckpointSaver:
    """Persist + resume one offline recipe's checkpoints, and fire the save / hub callbacks."""

    def __init__(self,
                 model: 'TrainableModel',
                 recorder: 'StepRunner',
                 dataloader: Any,
                 *,
                 output_dir: str = 'output',
                 save_steps: Optional[int] = None,
                 no_save_optim: bool = False,
                 no_save_rng: bool = False,
                 safe_serialization: bool = True,
                 max_shard_size: str = '5GB',
                 save_total_limit: Optional[int] = None,
                 ignore_data_skip: bool = False,
                 hub_pusher: Optional[Callable[[str], None]] = None,
                 hub_strategy: str = 'every_save',
                 consumed_train_samples_fn: Optional[Callable[[], int]] = None):
        self.model = model
        self.recorder = recorder
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.no_save_optim = no_save_optim
        self.no_save_rng = no_save_rng
        self.safe_serialization = safe_serialization
        self.max_shard_size = max_shard_size
        self.save_total_limit = save_total_limit
        # On resume, skip the dataloader to the consumed offset (False) or start a fresh pass while keeping
        # checkpoint progress / optimizer state (True).
        self.ignore_data_skip = ignore_data_skip
        # An optional callable the assembly injects to upload one checkpoint dir (it owns the hub
        # credentials / target / revision). hub_strategy='end' defers the push to the final checkpoint only.
        self.hub_pusher = hub_pusher
        self.hub_strategy = hub_strategy
        #: An online-RL recipe rolls a prompt stream, not a dataloader, so it has no ``get_state()`` consumed
        #: count to read; it injects this closure (the recorder's optimizer-step count) instead. ``None`` keeps
        #: the offline behaviour of reading the count off the dataloader (see :meth:`_consumed_train_samples`).
        self.consumed_train_samples_fn = consumed_train_samples_fn

    # --- write ------------------------------------------------------------------------------------
    def save(self, name: str = 'checkpoint-final', *, is_final: bool = False) -> str:
        """Persist the model + full training state, then fire the save / hub callbacks.

        The twinkle-native write (optimizer.pt / scheduler.pt / scaler.pt / rng_state.pt / trainer_state.json
        with schema cur_step / gradient_accumulation_steps / consumed_train_samples) is ``save_training_checkpoint``;
        dev does NOT hand-roll trainer_state -- the resume position comes from the dataloader's own consumed
        count (see :meth:`_consumed_train_samples`). ``is_final`` gates the hub-push 'end' strategy.
        """
        from swift.dev.recipe._loop_utils import save_training_checkpoint

        result = save_training_checkpoint(
            self.model,
            name,
            output_dir=self.output_dir,
            consumed_train_samples=self._consumed_train_samples(),
            no_save_optim=self.no_save_optim,
            no_save_rng=self.no_save_rng,
            safe_serialization=self.safe_serialization,
            max_shard_size=self.max_shard_size,
            save_total_limit=self.save_total_limit)
        # on_save reports the resolved dir rather than ``result``: in Ray (Megatron) mode save() returns a
        # deferred handle, not a path, so recompute the same way save_final does.
        self.recorder.publish_state()
        ckpt_dir = os.path.join(self.output_dir, name)
        self.recorder.handler.on_save(checkpoint_dir=ckpt_dir)
        # Hub push: on_push_begin fires before the upload so a callback can veto/annotate. 'end' pushes only
        # the final checkpoint; the push-on-save strategies push every one. Main-process only: under torchrun
        # every rank runs this loop, so an unguarded push would upload once per rank.
        if (self.hub_pusher is not None and (self.hub_strategy != 'end' or is_final) and self._is_main_process()):
            self.recorder.handler.on_push_begin(checkpoint_dir=ckpt_dir)
            self.hub_pusher(ckpt_dir)
        return result

    def save_final(self, name: str = 'checkpoint-final', *,
                   write_args_json: Optional[Callable[[str], None]] = None) -> str:
        """Persist the final checkpoint plus the ``args.json`` ``swift infer`` reads back.

        The path is recomputed rather than taken from :meth:`save`: in Ray (Megatron) mode save() returns a
        deferred handle, not a path. ``write_args_json`` (built from ``TrainAssembly.write_ckpt_args_json``)
        writes the self-describing args into the checkpoint dir; None skips it.
        """
        self.save(name, is_final=True)
        ckpt_dir = os.path.join(self.output_dir, name)
        if write_args_json is not None:
            write_args_json(ckpt_dir)
        return ckpt_dir

    # --- resume -----------------------------------------------------------------------------------
    def resume(self, state: dict) -> int:
        """Seed the recorder counters + dataloader position from a restored twinkle trainer_state.

        ``state`` = ``{'cur_step', 'consumed_train_samples', 'gradient_accumulation_steps'}`` (twinkle
        schema; single counting source). Weights / optimizer / scheduler / RNG are already restored by
        ``model.resume_from_checkpoint`` BEFORE this call. Here we only align the recorder's own counters and
        the dataloader's skip position so training + GA phase continue exactly. Returns the epoch the recipe
        should start its loop at (``0`` on ``ignore_data_skip``, else the dataloader's resume epoch so a
        cross-epoch resume does not replay already-consumed epochs).
        """
        from swift.dev.recipe._loop_utils import num_optimizer_steps

        cur_step = int(state['cur_step'])
        ga = self.recorder.gradient_accumulation_steps
        # micro_step must equal twinkle's cur_step so do_grad_sync phase stays aligned (twinkle's
        # optimizer_config.cur_step was already restored to cur_step).
        self.recorder.micro_step = cur_step
        # global_step (optimizer steps taken) derived from cur_step + ga (single source).
        self.recorder.global_step = num_optimizer_steps(cur_step, ga)
        # dataloader skip: reproduce the exact epoch/offset unless the user explicitly requested a fresh
        # pass over the data while retaining checkpoint progress / optimizer state.
        if self.ignore_data_skip:
            return 0
        consumed = int(state['consumed_train_samples'])
        if hasattr(self.dataloader, 'skip_consumed_samples'):
            self.dataloader.skip_consumed_samples(consumed)
        # Start the epoch loop at the resume epoch so cross-epoch resume does not replay already-consumed
        # epochs (the dataloader only skips the offset within its resume epoch).
        return int(self._dataloader_state().get('resume_epoch', 0))

    def _consumed_train_samples(self) -> int:
        """The resume counter written into a checkpoint: the dataloader's consumed count, not the step count.

        Read through ``get_state()`` rather than off an attribute: the dataloader is a twinkle
        ``remote_class``, so in ray mode the driver holds a handle whose attributes live in the worker and
        only its remote_functions answer. Same call the twinkle cookbooks use. An online-RL recipe has no
        dataloader and injects ``consumed_train_samples_fn`` instead.
        """
        if self.consumed_train_samples_fn is not None:
            return int(self.consumed_train_samples_fn())
        return self._dataloader_state().get('consumed_train_samples', 0)

    def _dataloader_state(self) -> dict:
        """The train dataloader's ``{consumed_train_samples, resume_epoch}``, or ``{}`` if it has none."""
        get_state = getattr(self.dataloader, 'get_state', None)
        return get_state() if get_state is not None else {}

    def _is_main_process(self) -> bool:
        """True on the rank that owns single-instance side effects (the hub upload).

        Under torchrun every rank drives its own copy of the loop, so an unguarded upload would run once per
        rank; under ray the loop runs only on the driver, where ``dist`` is not initialized and this is
        trivially the single owner.
        """
        import torch.distributed as dist
        return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
