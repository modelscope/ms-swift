# Copyright (c) ModelScope Contributors. All rights reserved.
"""Trainer mixin for Expert Parallel optimizer integration.

When EP is active and ``offload_expert_optimizer`` is set, the expert
optimizer state (Adam m/v, master weights) is kept on CPU, reducing GPU
memory.  This mixin integrates the CPU-offloaded optimizer into the
HuggingFace trainer lifecycle: creation, checkpoint save/load, and
cross-mesh param group handling.
"""
import os

import torch

from swift.expert_parallel.grad_norm import get_param_mesh
from swift.utils import get_logger

logger = get_logger()


class ExpertParallelMixin:
    """Mixin for SwiftMixin: expert parallel optimizer integration.

    Provides methods that carve EP expert parameters out of the main HF
    optimizer, handle CPU-offloaded optimizer state checkpointing, and
    split param groups by DTensor mesh to avoid cross-mesh fused Adam
    failures.
    """

    expert_optimizer = None

    def _maybe_create_expert_cpu_optimizer(self):
        """Carve EP expert params out of the HF optimizer into a CPU-offloaded optimizer.

        When `offload_expert_optimizer` is set under expert parallel, the expert params
        (the modules `expert_parallel` excluded from FSDP) are removed from the main HF
        optimizer and handed to an `ExpertCPUOptimizer`. This keeps the HF optimizer
        single-mesh / single-device so the FSDP2 checkpoint save+resume path stays valid,
        and moves the expert Adam state + gradients off the GPU. The expert optimizer is
        driven from `ExpertOptimizerCallback` (after grad clipping, at sync steps only).
        """
        self.expert_optimizer = None
        args = self.args
        # NOTE: gate only on `offload_expert_optimizer` — `expert_parallel_size` lives on the
        # base/template args and is NOT propagated onto the trainer's `training_args`, so it
        # cannot be read here. Whether EP is actually active is decided authoritatively below
        # by `expert_parallel.ignored_modules` (non-empty only after EP shards the experts).
        if not getattr(args, 'offload_expert_optimizer', False):
            return
        from swift.expert_parallel import expert_parallel, ExpertCPUOptimizer

        # The authoritative set of expert params is exactly what EP excluded from FSDP.
        expert_param_ids = set()
        for module in getattr(expert_parallel, 'ignored_modules', []):
            for p in module.parameters(recurse=True):
                expert_param_ids.add(id(p))
        if not expert_param_ids:
            logger.warning('offload_expert_optimizer set but no expert (ignored) modules found '
                           '(expert parallel not active?); skipping CPU offload of expert optimizer.')
            return

        # Pull expert params out of every HF optimizer group, remembering the hyper-params
        # of the group they came from (lr/betas/eps/weight_decay).
        expert_params = []
        src_hparams = None
        for pg in self.optimizer.param_groups:
            kept, taken = [], []
            for p in pg['params']:
                (taken if id(p) in expert_param_ids else kept).append(p)
            if taken:
                expert_params.extend(taken)
                if src_hparams is None:
                    src_hparams = {
                        'lr': pg.get('lr', args.learning_rate),
                        'betas': pg.get('betas', (args.adam_beta1, args.adam_beta2)),
                        'eps': pg.get('eps', args.adam_epsilon),
                        'weight_decay': pg.get('weight_decay', 0.0),
                    }
            pg['params'] = kept
        # Drop groups emptied by the carve-out (mirrors the filter in create_optimizer).
        self.optimizer.param_groups = [pg for pg in self.optimizer.param_groups if len(pg['params']) > 0]

        if not expert_params:
            return
        self.expert_optimizer = ExpertCPUOptimizer(
            expert_params,
            list(self.model.named_parameters()),
            master_dtype=getattr(args, 'expert_optimizer_dtype', 'bf16'),
            backend=getattr(args, 'expert_optimizer_backend', 'torch'),
            pin_memory=getattr(args, 'expert_optimizer_pin_memory', False),
            num_threads=getattr(args, 'expert_optimizer_num_threads', None),
            **src_hparams,
        )
        from swift.expert_parallel.expert_optimizer import ExpertOptimizerCallback
        self.add_callback(ExpertOptimizerCallback(self))
        logger.info(f'Moved {len(expert_params)} expert params to CPU-offloaded optimizer '
                    f'(hparams={src_hparams}); HF optimizer now has '
                    f'{len(self.optimizer.param_groups)} group(s).')

    def _expert_optimizer_path(self, ckpt_dir):
        # Expert optimizer state is rank-local (experts are EP-disjoint), so it is saved
        # per-rank rather than rank0-only.
        return os.path.join(ckpt_dir, f'expert_optimizer_{self.args.process_index}.bin')

    def _save_expert_optimizer(self, output_dir):
        expert_opt = getattr(self, 'expert_optimizer', None)
        if expert_opt is None or output_dir is None:
            return
        path = self._expert_optimizer_path(output_dir)
        torch.save(expert_opt.state_dict(), path)
        logger.info(f'Saved expert optimizer state to {path}')

    def _load_expert_optimizer(self, checkpoint):
        expert_opt = getattr(self, 'expert_optimizer', None)
        if expert_opt is None or checkpoint is None:
            return
        path = self._expert_optimizer_path(checkpoint)
        if not os.path.isfile(path):
            logger.warning(f'expert optimizer state not found at {path}; keeping freshly initialised state.')
            return
        expert_opt.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
        logger.info(f'Loaded expert optimizer state from {path}')

    def _disable_fused_for_cross_mesh_params(self):
        """When expert parallel is enabled, expert params live on the ep mesh
        while FSDP params live on the global mesh. Fused/foreach Adam batches
        params of different meshes into one kernel and fails on pointwise
        propagation. Split param groups by mesh to avoid this.
        """
        all_meshes = set()
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                all_meshes.add(get_param_mesh(param))
        # Only one mesh present (or no DTensor at all): nothing to do.
        if len(all_meshes) <= 1:
            return

        new_param_groups = []
        for param_group in self.optimizer.param_groups:
            mesh_to_params = {}
            for param in param_group['params']:
                mesh_to_params.setdefault(get_param_mesh(param), []).append(param)
            if len(mesh_to_params) <= 1:
                new_param_groups.append(param_group)
                continue
            # Split this group — one sub-group per mesh.
            # Disable fused/foreach to prevent cross-mesh kernel batching.
            for mesh, params in mesh_to_params.items():
                new_group = {k: v for k, v in param_group.items() if k != 'params'}
                new_group['params'] = params
                new_group['fused'] = False
                new_group['foreach'] = False
                new_param_groups.append(new_group)
        self.optimizer.param_groups = new_param_groups
        logger.info(f'Split optimizer param groups by device mesh for expert parallel '
                    f'({len(new_param_groups)} groups)')