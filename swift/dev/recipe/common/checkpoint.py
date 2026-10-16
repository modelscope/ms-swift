"""The checkpoint half of a flat offline recipe: twinkle-native write, hub push, resume, args.json.

``CheckpointSaver`` is the thin re-home of the persistence the old ``OfflineLoop.save`` / ``TrainLoop.save``
/ ``resume`` and ``TrainAssembly.save_final`` carried: one twinkle-native write (optimizer.pt /
scheduler.pt / rng_state.pt / trainer_state.json), the ``on_save`` / ``on_push_begin`` callback fan-out,
the optional hub upload, and the resume that re-seeds the recorder counters + skips the data to its
consumed offset. It owns no loop -- the recipe calls :meth:`save` at the ``save_steps`` cadence and
:meth:`save_final` at train end, and calls :meth:`resume` once before the loop when resuming.

The self-describing ``args.json`` ``swift infer`` reads back is written by the recipe's ``write_args_json``
callable (built from ``TrainAssembly.write_ckpt_args_json``), so this class stays free of config knowledge.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.model import TrainableModel
    from swift.dev.recipe.common.step_runner import StepRunner

logger = get_logger()

__all__ = ['CheckpointSaver']


class CheckpointSaver:
    """Persist training state and delegate component state semantics to registered components.

    Every registered component implements ``get_state()`` and ``load_state(state)``. This class only stores
    the resulting ``{namespace: state}`` mapping and returns each namespace to the same component on resume.
    The positional ``dataloader`` remains the default offline registration for existing recipe callers.
    """

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
                 components: Optional[Dict[str, Any]] = None):
        self.model = model
        self.recorder = recorder
        self.components = dict(components or {})
        if dataloader is not None:
            if 'dataloader' in self.components:
                raise ValueError('Register the dataloader either positionally or in components, not both.')
            self.components['dataloader'] = dataloader
        self.dataloader = self.components.get('dataloader')
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

    # --- write ------------------------------------------------------------------------------------
    def save(self, name: str = 'checkpoint-final', *, is_final: bool = False) -> str:
        """Persist the model + full training state, then fire the save / hub callbacks.

        The twinkle-native write (optimizer.pt / scheduler.pt / scaler.pt / rng_state.pt / trainer_state.json)
        is ``save_training_checkpoint``; dev passes the generic ``component_states`` mapping without interpreting
        component-specific fields. ``is_final`` gates the hub-push 'end' strategy.
        """
        from swift.dev.recipe.common.loop_utils import save_training_checkpoint

        component_states = self._component_states()
        result = save_training_checkpoint(
            self.model,
            name,
            output_dir=self.output_dir,
            consumed_train_samples=self._consumed_train_samples(component_states),
            component_states=component_states,
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

        Weights / optimizer / scheduler / RNG are already restored by ``model.resume_from_checkpoint`` before
        this call. Here we align recorder counters and return each ``component_states`` namespace to its owner.
        Checkpoints without that field use the top-level ``consumed_train_samples`` dataloader fallback. The
        return value is the dataloader epoch for the offline loop, or 0 when no dataloader is registered.
        """
        from swift.dev.recipe.common.loop_utils import num_optimizer_steps

        cur_step = int(state['cur_step'])
        ga = self.recorder.gradient_accumulation_steps
        # micro_step must equal twinkle's cur_step so do_grad_sync phase stays aligned (twinkle's
        # optimizer_config.cur_step was already restored to cur_step).
        self.recorder.micro_step = cur_step
        # global_step (optimizer steps taken) derived from cur_step + ga (single source).
        self.recorder.global_step = num_optimizer_steps(cur_step, ga)
        if 'component_states' in state:
            component_states = state['component_states']
            if not isinstance(component_states, dict):
                raise TypeError('checkpoint component_states must be a dict.')
            self._load_component_states(component_states)
        elif not self.ignore_data_skip and self.dataloader is not None:
            # Compatibility with checkpoints written before component_states existed.
            skip = getattr(self.dataloader, 'skip_consumed_samples', None)
            if skip is not None:
                skip(int(state.get('consumed_train_samples', 0)))
        if self.ignore_data_skip or self.dataloader is None:
            return 0
        return int(self._dataloader_state().get('epoch', 0))

    def _consumed_train_samples(self, component_states: Optional[Dict[str, dict]] = None) -> int:
        """The resume counter written into a checkpoint: the data position, not the optimizer-step count.

        Read through ``get_state()`` rather than off an attribute: the dataloader is a twinkle
        ``remote_class``, so in ray mode the driver holds a handle whose attributes live in the worker and
        only its remote_functions answer. Same call the twinkle cookbooks use.

        A step count would be the wrong thing to store even for a recipe that has no dataloader: the two
        numbers are unrelated, so restoring one as the other replays data the run already trained on.
        """
        states = component_states if component_states is not None else self._component_states()
        return int(states.get('dataloader', {}).get('consumed_train_samples', 0))

    def _component_states(self) -> Dict[str, dict]:
        states: Dict[str, dict] = {}
        for namespace, component in self.components.items():
            get_state = getattr(component, 'get_state', None)
            if get_state is None:
                raise TypeError(f'Checkpoint component {namespace!r} does not implement get_state().')
            component_state = get_state()
            if not isinstance(component_state, dict):
                raise TypeError(f'Checkpoint component {namespace!r} returned non-dict state.')
            states[namespace] = component_state
        return states

    def _load_component_states(self, states: Dict[str, dict]) -> None:
        for namespace, component in self.components.items():
            if namespace not in states or (namespace == 'dataloader' and self.ignore_data_skip):
                continue
            load_state = getattr(component, 'load_state', None)
            if load_state is None:
                raise TypeError(f'Checkpoint component {namespace!r} does not implement load_state().')
            load_state(states[namespace])

    def _dataloader_state(self) -> dict:
        """Return the registered dataloader state, or an empty state when no dataloader is registered."""
        if self.dataloader is None:
            return {}
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
