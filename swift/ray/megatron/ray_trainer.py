# Copyright (c) ModelScope Contributors. All rights reserved.
"""Algorithm-specific Ray training loops.

Class hierarchy::

    RayTrainer                  # default loop (SFT / KTO)
      ├─ DPORayTrainer          # DPO with optional remote ref
      └─ GRPORayTrainer         # GRPO with vLLM rollout

The driver owns the global dataset and dataloader. Workers never
create their own train data_iterators — they receive per-DP-rank
batches dispatched from the driver each step.
"""
import ray
import torch
from functools import partial
from typing import Any, Dict, List, Optional

from swift.utils import get_logger
from .worker_group import DPDispatchedDict

logger = get_logger()

# ======================================================================
# Driver-side dataset builder
# ======================================================================


def build_dataset_from_argv(argv: List[str], rlhf_type: str = 'dpo'):
    """Build dataset + collator on the driver (no Megatron/NCCL init)."""
    import os

    from swift.megatron.arguments.sft_args import MegatronSftArguments
    from swift.megatron.utils import get_padding_to
    from swift.utils import to_abspath

    saved = MegatronSftArguments._init_megatron_args
    MegatronSftArguments._init_megatron_args = lambda self: None

    try:
        from swift.megatron.pipelines.train.rlhf import MegatronRLHF
        from swift.pipelines.base import SwiftPipeline

        class _DriverPipeline(MegatronRLHF):

            def __init__(self, _argv, _rlhf_type):
                self.train_msg = {}
                SwiftPipeline.__init__(self, _argv)
                args = self.args
                args.rlhf_type = _rlhf_type
                if args.output_dir is None:
                    args.output_dir = f'megatron_output/{args.model_suffix}'
                args.output_dir = to_abspath(args.output_dir)
                os.makedirs(args.output_dir, exist_ok=True)
                with torch.device('meta'):
                    self.model, self.processor = args.get_model_processor(
                        load_model=False, download_model=args.mcore_model is None)
                self._prepare_template()
                self.template.use_megatron = True

        pipeline = _DriverPipeline(argv, rlhf_type)
        train_dataset, val_dataset = pipeline._prepare_dataset()

        args = pipeline.args
        if args.rlhf_type in ('grpo', 'gkd'):
            from swift.rlhf_trainers.utils import identity_data_collator
            data_collator = identity_data_collator
            collator_fn = identity_data_collator
        else:
            collator_fn = pipeline.template.data_collator
            padding_to = get_padding_to(pipeline.args)
            data_collator = partial(collator_fn, padding_to=padding_to)

        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'data_collator': data_collator,
            '_collator_fn': collator_fn,
            'micro_batch_size': args.micro_batch_size,
            'global_batch_size': args.global_batch_size,
            'padding_free': args.padding_free,
            'num_train_epochs': args.num_train_epochs,
            'train_iters': args.train_iters,
            'save_strategy': args.save_strategy,
            'eval_iters': args.eval_iters,
            'rlhf_type': args.rlhf_type,
            'num_generations': getattr(args, 'num_generations', 1),
            'template': pipeline.template,
        }
    finally:
        MegatronSftArguments._init_megatron_args = saved


def compute_iter_params(data_info: Dict[str, Any], dp_size: int) -> Dict[str, Any]:
    """Compute train_iters / eval_iters / save_steps on the driver."""
    mbs = data_info['micro_batch_size']
    gbs = data_info['global_batch_size']
    step_batch_size = mbs * dp_size
    num_gen = data_info.get('num_generations', 1)
    if data_info.get('rlhf_type') not in ('grpo', 'gkd'):
        num_gen = 1

    train_ds = data_info.get('train_dataset')
    val_ds = data_info.get('val_dataset')
    train_len = len(train_ds) if train_ds is not None and hasattr(train_ds, '__len__') else 0
    val_len = len(val_ds) if val_ds is not None and hasattr(val_ds, '__len__') else 0

    result: Dict[str, Any] = {}

    if data_info.get('save_strategy') == 'epoch' and train_len > 0:
        ds_sample = train_len // step_batch_size * step_batch_size * num_gen
        result['save_steps'] = ds_sample // gbs
        result['eval_steps'] = result['save_steps']

    train_iters = data_info.get('train_iters')
    if data_info.get('num_train_epochs') is not None and train_len > 0:
        ds_sample = train_len // step_batch_size * step_batch_size * num_gen
        train_iters = ds_sample * data_info['num_train_epochs'] // gbs
    result['train_iters'] = train_iters

    eval_iters = data_info.get('eval_iters', -1)
    if eval_iters is not None and eval_iters < 0:
        if val_len == 0:
            eval_iters = 0
        else:
            ds_sample = val_len // step_batch_size * step_batch_size * num_gen
            eval_iters = max(ds_sample // gbs, 1)
    if val_len > 0 and val_len < step_batch_size:
        eval_iters = 0
    result['eval_iters'] = eval_iters or 0

    return result


# ======================================================================
# RayTrainer
# ======================================================================


class RayTrainer:
    """Base class.  Subclasses override ``_train_loop``."""

    def __init__(self, worker_groups: Dict[str, Any]):
        self.worker_groups = worker_groups
        self._dataloader = None
        self._data_iter = None

    @property
    def train_group(self):
        return self.worker_groups['train']

    def get_train_init_kwargs(self, group_names=None) -> dict:
        return {}

    def set_data_info(self, data_info: Dict[str, Any]):
        self._data_info = data_info

    def fit(self) -> Any:
        tg = self.train_group
        self._build_dataloader(tg)

        iter_params = compute_iter_params(self._data_info, tg.dp_size)
        meta = tg.setup(iter_params)
        train_iters = meta['train_iters']
        iteration = meta['iteration']

        try:
            iteration = self._train_loop(tg, train_iters, iteration)
        except Exception:
            logger.exception('Failed at iteration %d', iteration)
            raise
        finally:
            results = tg.finalize()

        return results[0] if results else None

    def _build_dataloader(self, tg):
        info = self._data_info
        dataset = info['train_dataset']
        collator = info['data_collator']
        self._padding_free = info.get('padding_free', False)

        if self._padding_free:
            batch_size = info['micro_batch_size']
        else:
            batch_size = info['global_batch_size']

        self._dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            drop_last=True,
            pin_memory=True,
        )
        self._data_iter = self._cyclic_iter(self._dataloader)
        self._dp_size = tg.dp_size
        self._num_microbatches = info['global_batch_size'] // (info['micro_batch_size'] * tg.dp_size)
        logger.info('Driver dataloader: dataset=%d, batch_size=%d, dp_size=%d, '
                    'padding_free=%s, num_microbatches=%d', len(dataset), batch_size, tg.dp_size, self._padding_free,
                    self._num_microbatches)

    @staticmethod
    def _cyclic_iter(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    def _next_batch(self) -> Dict[str, Any]:
        if not self._padding_free:
            return next(self._data_iter)
        dp_size = self._dp_size
        n_micro = self._num_microbatches
        dp_batches = DPDispatchedDict({r: [] for r in range(dp_size)})
        for dp_r in range(dp_size):
            for _ in range(n_micro):
                dp_batches[dp_r].append(next(self._data_iter))
        return dp_batches

    def _train_loop(self, tg, train_iters, iteration):
        dispatch = 'dp' if self._padding_free else None
        while iteration < train_iters:
            batch = self._next_batch()
            results = tg.train_step(batch, dispatch=dispatch)
            iteration = self._extract_iteration(results)
        return iteration

    def _extract_iteration(self, step_results):
        for r in step_results:
            if r and 'iteration' in r:
                return r['iteration']
        return 0


# ======================================================================
# DPORayTrainer
# ======================================================================


class DPORayTrainer(RayTrainer):
    """DPO training with optional remote ref group.

    Supports both co-located and separated ref models:
      - Co-located: train and ref share the same HybridWorker (same group).
        ref_forward is called on the train group.
      - Separated: ref is an independent MegatronWorker group.
        forward is called on the ref group.
    """

    REMOTE_REF_TRAINER_CLS = ('swift.ray.megatron.ray_megatron_trainer.RayMegatronDPOTrainer')

    @property
    def ref_group(self) -> Optional[Any]:
        return self.worker_groups.get('ref')

    @property
    def has_remote_ref(self) -> bool:
        return self.ref_group is not None

    @property
    def is_ref_colocated(self) -> bool:
        """True if ref and train share the same WorkerGroup (HybridWorker)."""
        return self.ref_group is not None and self.ref_group is self.train_group

    def get_train_init_kwargs(self, group_names=None) -> dict:
        has_ref = (group_names is not None and 'ref' in group_names) or self.has_remote_ref
        if has_ref and not self.is_ref_colocated:
            return {'trainer_cls_path': self.REMOTE_REF_TRAINER_CLS}
        return {}

    def _train_loop(self, tg, train_iters, iteration):
        ref = self.ref_group
        dispatch = 'dp' if self._padding_free else None

        while iteration < train_iters:
            global_batch = self._next_batch()

            if ref is not None:
                if self.is_ref_colocated:
                    ref_results = ref.ref_forward(global_batch, dispatch=dispatch)
                else:
                    ref_results = ref.forward(global_batch, dispatch=dispatch)
                self._merge_ref_logps(global_batch, ref_results)

            results = tg.train_step(global_batch, dispatch=dispatch)
            iteration = self._extract_iteration(results)

        return iteration

    def _merge_ref_logps(self, global_batch, ref_results):
        """Merge ref log-probs into the global batch for training.

        Handles both padding-free (DPDispatchedDict keyed by dp_rank)
        and regular (single dict) batch formats.
        """
        if not ref_results:
            return

        from .worker_group import _is_dp_dispatched

        if _is_dp_dispatched(global_batch):
            for dp_r, micro_batches in global_batch.items():
                r = ref_results.get(dp_r)
                if r is not None and 'ref_logps' in r:
                    logps = r['ref_logps']
                    n_micro = len(micro_batches)
                    if n_micro > 1 and logps.shape[0] >= n_micro:
                        parts = logps.chunk(n_micro)
                        for mb, lp in zip(micro_batches, parts):
                            mb['ref_logps'] = lp
                    else:
                        for mb in micro_batches:
                            mb['ref_logps'] = logps
        else:
            sorted_ranks = sorted(ref_results.keys())
            logps_parts = []
            for dp_r in sorted_ranks:
                r = ref_results[dp_r]
                if r is not None and 'ref_logps' in r:
                    logps_parts.append(r['ref_logps'])
            if logps_parts:
                global_batch['ref_logps'] = torch.cat(logps_parts, dim=0)


# ======================================================================
# GRPORayTrainer
# ======================================================================


class GRPORayTrainer(RayTrainer):
    """GRPO training with vLLM rollout group.

    The GRPO training loop:
    1. Fetch a batch of prompts from the driver dataloader
    2. Generate N completions per prompt via rollout group
    3. Score completions with reward functions on the train group
    4. Compute advantages on the driver
    5. Feed encoded batches with advantages to the Megatron trainer

    Supports both co-located and separated modes:
      - Co-located: train and rollout share the same HybridWorker.
        The Megatron trainer's ``_replace_data_iterator`` handles the
        full rollout->reward->advantage pipeline internally.
      - Separated: rollout is an independent VllmWorker group.
        The driver coordinates weight sync, generation, and feeds
        the raw prompt batches (with completions) to the train group,
        which still uses ``_replace_data_iterator`` internally.
    """

    _sync_interval: int = 1

    @property
    def rollout_group(self):
        return self.worker_groups.get('rollout')

    @property
    def is_colocated(self) -> bool:
        """True if rollout and train share the same WorkerGroup."""
        return self.rollout_group is not None and self.rollout_group is self.train_group

    def _build_dataloader(self, tg):
        info = self._data_info
        is_grpo = info.get('rlhf_type') in ('grpo', 'gkd')
        if is_grpo:
            dataset = info['train_dataset']
            self._padding_free = False
            batch_size = info['global_batch_size']
            self._dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=info['data_collator'],
                drop_last=True,
            )
            self._data_iter = self._cyclic_iter(self._dataloader)
            self._dp_size = tg.dp_size
            self._num_microbatches = info['global_batch_size'] // (info['micro_batch_size'] * tg.dp_size)
            logger.info('GRPO driver dataloader: dataset=%d, batch_size=%d', len(dataset), batch_size)
        else:
            super()._build_dataloader(tg)

    def fit(self) -> Any:
        tg = self.train_group
        self._build_dataloader(tg)

        iter_params = compute_iter_params(self._data_info, tg.dp_size)
        meta = tg.setup(iter_params)
        train_iters = meta['train_iters']
        iteration = meta['iteration']

        rollout = self.rollout_group
        if rollout is not None and self.is_colocated:
            logger.info('GRPO co-located mode: train and rollout share HybridWorker')
        elif rollout is not None:
            logger.info('GRPO separated mode: train and rollout are separate groups')

        try:
            iteration = self._train_loop(tg, train_iters, iteration)
        except Exception:
            logger.exception('Failed at iteration %d', iteration)
            raise
        finally:
            results = tg.finalize()

        return results[0] if results else None

    def _train_loop(self, tg, train_iters, iteration):
        """GRPO training loop.

        Co-located mode:
            The Megatron trainer's ``_replace_data_iterator`` handles
            the full rollout->reward->advantage pipeline internally.
            The driver simply dispatches the raw prompt batch.

        Separated mode:
            The driver coordinates: (1) weight sync from train to rollout,
            (2) generation via rollout group, (3) merging completions into
            the prompt batch, and (4) dispatching to the train group which
            handles reward scoring, advantage computation, and training
            via ``_replace_data_iterator`` (with generation already done
            and vllm_mode='server' pointing to the separated rollout).
        """
        rollout = self.rollout_group

        while iteration < train_iters:
            batch = self._next_batch()

            if rollout is not None and not self.is_colocated:
                if iteration == 0 or iteration % self._sync_interval == 0:
                    self._do_weight_sync(tg, rollout)

                completions = self._generate_on_rollout(rollout, batch)
                self._merge_completions(batch, completions)

            dispatch = 'dp' if self._padding_free else None
            results = tg.train_step(batch, dispatch=dispatch)
            iteration = self._extract_iteration(results)

        return iteration

    def _generate_on_rollout(self, rollout, batch):
        """Generate completions via separated rollout group.

        Args:
            rollout: The rollout WorkerGroup (VllmWorker instances).
            batch: The prompt batch (list of dicts with 'messages', etc.).

        Returns:
            List of generation output dicts from the rollout group.
        """
        return rollout.generate(batch)

    def _merge_completions(self, batch, completions):
        """Merge rollout completions into the prompt batch.

        For separated mode, the completions from the rollout group
        are attached to each prompt so that the Megatron trainer's
        ``_replace_data_iterator`` can use them without re-generating.

        Args:
            batch: The prompt batch (list of dicts).
            completions: Generation output from rollout group.
        """
        if completions is None:
            return
        if isinstance(batch, list) and isinstance(completions, list):
            for item, comp in zip(batch, completions):
                if comp is not None:
                    item['_rollout_completion'] = comp

    def _do_weight_sync(self, tg, rollout):
        """Sync weights from training model to rollout engine.

        - Co-located: direct intra-process call (zero-copy via bridge).
        - Separated: export weights to CPU, send via driver to VllmWorker.
        """
        if self.is_colocated:
            tg.sync_weights_to_rollout()
            logger.info('Co-located weight sync done')
        else:
            weights = tg.export_weights()
            if weights:
                rollout.update_weights(weights)
                logger.info('Separated weight sync done')
