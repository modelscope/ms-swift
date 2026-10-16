# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU-offloaded optimizer for Expert-Parallel (EP) expert weights.

When EP is enabled (`swift/expert_parallel/ep.py`), each rank owns a distinct slice of
the routed experts, stored as `nn.Parameter(DTensor)` sharded `Shard(0)` on a dedicated
`ep` device-mesh and *excluded* from FSDP (`ignored_modules`). Their weights, Adam state
and gradients would otherwise stay permanently on the GPU and dominate memory.

This optimizer keeps the expert **optimizer state (and the master copy) on CPU** and runs
the AdamW update on CPU. Per global step it:
  1. copies the (already grad-clipped) GPU expert gradient down to a pinned CPU buffer,
  2. runs a CPU AdamW step on the fp32 master + m/v (or bf16 m/v),
  3. copies the updated weight back up into the GPU param's local shard.

It is intentionally NOT a `torch.optim.Optimizer` subclass and is driven manually from a
trainer callback (after grad clipping, at sync steps only) so that the Hugging Face
optimizer it is carved out of stays single-mesh / single-device — which keeps the FSDP2
checkpoint save+resume path (`get_optimizer_state_dict` / `set_optimizer_state_dict`)
working. See the plan / `swift/trainers/mixin.py` for the integration points.
"""
import os

import torch
from torch.distributed.tensor import DTensor
from transformers import TrainerCallback

from swift.utils import get_logger

logger = get_logger()


class ExpertCPUOptimizer:
    """AdamW for EP expert params with optimizer state offloaded to CPU.

    Args:
        params: the expert parameters carved out of the main optimizer. These are EP
            `DTensor`s (placements `[Shard(0)]`) living on the `ep` mesh; their *local*
            shard is the GPU tensor we actually update.
        named_parameters: iterable of `(name, param)` over the full model, used to build
            a stable FQN map for checkpointing (expert state is keyed by FQN, not by the
            python object identity, so it survives a process restart).
        lr, betas, eps, weight_decay: AdamW hyper-parameters, copied from the param group
            removed from the main optimizer.
        master_dtype: 'fp32' keeps an fp32 master weight + fp32 m/v on CPU and writes the
            update back to the (bf16) GPU weight; 'bf16' keeps bf16 m/v with no master,
            matching the on-GPU `adamw_torch_fused` baseline.
        backend: 'torch' uses the functional `torch.optim.adamw`; 'deepspeed' uses
            `DeepSpeedCPUAdam` (a faster multithreaded CPU kernel) if installed.
    """

    def __init__(self,
                 params,
                 named_parameters,
                 lr,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0.0,
                 master_dtype='bf16',
                 backend='torch',
                 pin_memory=False,
                 num_threads=None):
        self.params = [p for p in params if p is not None]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        # 'fp32' is an alias of 'mixed' (fp32 master + fp32 m/v). 'bf16' = bf16 m/v, no master.
        assert master_dtype in ('fp32', 'mixed', 'bf16'), f'invalid master_dtype: {master_dtype}'
        self.master_dtype = master_dtype
        self.use_master = master_dtype in ('fp32', 'mixed')
        self._state_dtype = torch.float32 if self.use_master else torch.bfloat16
        # dtype of the actual (GPU) weight shard we write back into.
        self.weight_dtype = self.params[0].dtype if self.params else torch.bfloat16

        # Stable FQN for every expert param, for checkpointing.
        id2name = {id(p): n for n, p in named_parameters}
        self._param_fqn = {}
        for p in self.params:
            fqn = id2name.get(id(p))
            assert fqn is not None, 'expert param not found in model.named_parameters()'
            self._param_fqn[p] = fqn

        # Per-param CPU state: pinned m/v (+ master). `step` is a plain python int
        # (advanced once per *global* step) for bias correction.
        # NOTE: the D2H grad buffer and H2D write-back buffer are NOT kept per-param —
        # `step()` processes one param at a time, so two SHARED scratch buffers sized to
        # the largest single expert param suffice. Keeping them per-param would roughly
        # double resident CPU memory (an extra fp32 grad + bf16 weight copy per param).
        self.state = {}
        # Pinned (page-locked) memory speeds H2D/D2H but is non-swappable and counts against
        # the cgroup limit — dangerous for hundreds of GB of state. Off by default.
        self._pin = bool(pin_memory) and torch.cuda.is_available()
        # Threads for the CPU AdamW step. torch defaults to 1 inside the training process
        # (set by the launcher / OMP_NUM_THREADS), but the per-rank expert step is a big
        # single-threaded bottleneck (~48s -> ~4.5s at 64 threads for DeepSeek-V4 EP8).
        # We bump the thread count ONLY during step() and restore it after, so the rest of
        # the training loop (dataloader, other CPU ops) is unaffected.
        self._num_threads = self._resolve_num_threads(num_threads)
        max_numel = 0
        for p in self.params:
            local = p.to_local() if isinstance(p, DTensor) else p
            max_numel = max(max_numel, local.numel())
            self.state[p] = {
                'step': 0,
                'exp_avg': self._empty_like(local, self._state_dtype),
                'exp_avg_sq': self._empty_like(local, self._state_dtype),
            }
            if self.use_master:
                # fp32 master initialised from the (bf16) weight's current value
                self.state[p]['master'] = local.detach().to('cpu', torch.float32).clone()

        # Shared, flat scratch buffers (allocated once, sliced+reshaped per param).
        # Grad scratch dtype matches the optimizer math: fp32 when there is an fp32 master,
        # bf16 in the no-master path (the GPU grad is bf16 and consumed as bf16 — avoids both
        # an fp32 over-allocation and a per-step cast).
        self._grad_dtype = torch.float32 if self.use_master else torch.bfloat16
        self._scratch_grad = self._empty_flat(max_numel, self._grad_dtype)
        self._scratch_wb = self._empty_flat(max_numel, self.weight_dtype)

        self._backend = backend
        self._ds_opt = None
        if backend == 'deepspeed':
            self._init_deepspeed()

        logger.info(f'ExpertCPUOptimizer initialised: {len(self.params)} expert params, '
                    f'master_dtype={master_dtype}, backend={self._backend}, '
                    f'num_threads={self._num_threads}, pin_memory={self._pin}, lr={lr}')

    @staticmethod
    def _resolve_num_threads(num_threads):
        """Threads for the CPU AdamW step.

        Inside a torchrun worker `os.cpu_count()`/`torch.get_num_threads()` is often 1
        (the launcher sets OMP_NUM_THREADS=1), so we read the cgroup CPU quota to find how
        many cores the container actually has. CRUCIAL: all `local_world_size` ranks on a
        node run their expert step concurrently, so we divide the cores by that count —
        otherwise 8 ranks x 64 threads = 512 threads oversubscribe ~101 cores (~5x) and the
        step barely speeds up. Cap at 64 (sweet spot; beyond that memory bandwidth saturates).
        Set `num_threads` explicitly (or env SWIFT_EXPERT_OPTIMIZER_THREADS) to override; an
        explicit value is used as-is and NOT divided.
        """
        if num_threads is None:
            num_threads = os.environ.get('SWIFT_EXPERT_OPTIMIZER_THREADS')
        if num_threads is not None:
            return max(1, int(num_threads))
        cores = None
        # cgroup v2
        try:
            with open('/sys/fs/cgroup/cpu.max') as f:
                quota, period = f.read().split()
                if quota != 'max':
                    cores = int(int(quota) / int(period))
        except Exception:  # noqa
            pass
        # cgroup v1
        if cores is None:
            try:
                with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us') as f:
                    quota = int(f.read())
                with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us') as f:
                    period = int(f.read())
                if quota > 0:
                    cores = int(quota / period)
            except Exception:  # noqa
                pass
        if cores is None:
            cores = os.cpu_count() or 1
        # Divide by ranks-per-node so concurrent expert steps don't oversubscribe the cores.
        from swift.utils import get_dist_setting
        local_world_size = max(1, get_dist_setting()[3])
        per_rank = max(1, cores // local_world_size)
        return max(1, min(64, per_rank))

    def _empty_like(self, ref: torch.Tensor, dtype) -> torch.Tensor:
        t = torch.zeros(ref.shape, dtype=dtype, device='cpu')
        if self._pin:
            t = t.pin_memory()
        return t

    def _empty_flat(self, numel: int, dtype) -> torch.Tensor:
        t = torch.zeros(numel, dtype=dtype, device='cpu')
        if self._pin:
            t = t.pin_memory()
        return t

    def _grad_buf(self, ref):
        """A view of the shared grad scratch shaped like `ref` (matches optimizer math dtype)."""
        return self._scratch_grad[:ref.numel()].view(ref.shape)

    def _wb_buf(self, ref):
        """A view of the shared write-back scratch shaped like `ref`."""
        return self._scratch_wb[:ref.numel()].view(ref.shape)

    def _d2h_grad(self, grad_local):
        """D2H copy the GPU gradient into the shared CPU scratch buffer, then synchronize."""
        grad_cpu = self._grad_buf(grad_local)
        grad_cpu.copy_(grad_local, non_blocking=True)
        if grad_local.is_cuda:
            torch.cuda.current_stream().synchronize()
        return grad_cpu

    def _h2d_weight(self, updated, weight_local):
        """H2D write-back the updated weight into the GPU param's local shard (blocking)."""
        wb = self._wb_buf(weight_local)
        if updated is not wb:
            wb.copy_(updated.to(wb.dtype))
        weight_local.copy_(wb, non_blocking=False)

    def _init_deepspeed(self):
        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
        except Exception as e:  # noqa
            logger.warning(f'DeepSpeedCPUAdam unavailable ({e}); falling back to torch CPU AdamW.')
            self._backend = 'torch'
            return
        # DeepSpeedCPUAdam owns its own fp32 master + state internally; we feed it CPU
        # params whose .grad we fill each step, then copy the result back to the GPU.
        self._ds_params = []
        self._ds_param_for = {}
        for p in self.params:
            local = p.to_local() if isinstance(p, DTensor) else p
            cpu_p = torch.nn.Parameter(local.detach().to('cpu', torch.float32).clone())
            self._ds_params.append(cpu_p)
            self._ds_param_for[p] = cpu_p
        self._ds_opt = DeepSpeedCPUAdam(
            self._ds_params, lr=self.lr, betas=(self.beta1, self.beta2), eps=self.eps,
            weight_decay=self.weight_decay, adamw_mode=True)

    def set_lr(self, lr):
        """Mirror the main scheduler's current LR onto the expert optimizer."""
        self.lr = lr
        if self._ds_opt is not None:
            for g in self._ds_opt.param_groups:
                g['lr'] = lr

    @staticmethod
    def _local(t):
        return t.to_local() if isinstance(t, DTensor) else t

    def _has_any_grad(self) -> bool:
        return any(p.grad is not None for p in self.params)

    @torch.no_grad()
    def step(self):
        """Run one CPU AdamW step over all expert params with a pending gradient.

        Must be called at a *sync* step, after grad clipping, while the GPU grads are
        still populated. If every expert grad is None (e.g. the global grad-norm was NaN
        and `_fix_grad_norm_nan` zeroed all grads) this is a no-op, matching the main
        optimizer's skip.
        """
        if not self._has_any_grad():
            return
        # Bump CPU threads only for the (otherwise single-threaded) AdamW math, then restore
        # so the rest of the training loop is unaffected.
        prev_threads = torch.get_num_threads()
        if self._num_threads != prev_threads:
            torch.set_num_threads(self._num_threads)
        try:
            if self._backend == 'deepspeed' and self._ds_opt is not None:
                self._step_deepspeed()
            else:
                self._step_torch()
        finally:
            if self._num_threads != prev_threads:
                torch.set_num_threads(prev_threads)

    def _step_torch(self):
        from torch.optim.adamw import adamw

        for p in self.params:
            if p.grad is None:
                continue
            st = self.state[p]
            grad_local = self._local(p.grad)
            weight_local = self._local(p)  # GPU bf16 shard (updated in place)

            # 1. D2H the clipped gradient into the shared scratch (sliced to this param).
            grad_cpu = self._d2h_grad(grad_local)

            if self.use_master:
                # fp32 master IS the optimizer param; grad already fp32 (D2H upcast).
                params_list = [st['master']]
                grads_list = [grad_cpu]
            else:
                # bf16 path: update a bf16 copy of the weight on CPU (shared scratch).
                # grad_cpu is already bf16 here (no per-step cast needed).
                wb = self._wb_buf(weight_local)
                wb.copy_(weight_local, non_blocking=True)
                if weight_local.is_cuda:
                    torch.cuda.current_stream().synchronize()
                params_list = [wb]
                grads_list = [grad_cpu]

            st['step'] += 1
            step_t = torch.tensor(float(st['step']))
            adamw(
                params_list,
                grads_list,
                [st['exp_avg']],
                [st['exp_avg_sq']],
                [],
                [step_t],
                amsgrad=False,
                beta1=self.beta1,
                beta2=self.beta2,
                lr=self.lr,
                weight_decay=self.weight_decay,
                eps=self.eps,
                maximize=False,
                foreach=True,
                capturable=False,
                differentiable=False,
                fused=False,
                grad_scale=None,
                found_inf=None,
                has_complex=False,
            )

            # 2. H2D the updated weight back into the GPU param's local shard.
            #    The write-back source is a SHARED scratch buffer reused by the
            #    next param, so the H2D must complete before the loop overwrites it.
            self._h2d_weight(params_list[0], weight_local)
        # make sure all weight write-backs complete before the next forward reads them
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()

    def _step_deepspeed(self):
        for p in self.params:
            if p.grad is None:
                continue
            cpu_p = self._ds_param_for[p]
            grad_local = self._local(p.grad)
            grad_cpu = self._d2h_grad(grad_local)
            # DeepSpeedCPUAdam reads cpu_p.grad in place; give it a private (cloned) grad
            # since the shared scratch is reused across params.
            cpu_p.grad = grad_cpu.clone()
        self._ds_opt.step()
        for p in self.params:
            if p.grad is None:
                continue
            cpu_p = self._ds_param_for[p]
            cpu_p.grad = None
            weight_local = self._local(p)
            self._h2d_weight(cpu_p.detach(), weight_local)
            self.state[p]['step'] += 1
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    # ------------------------------------------------------------------ #
    # Checkpointing — keyed by FQN so it survives a process restart.
    # State is rank-local (experts are EP-disjoint) and saved per-rank by the trainer.
    # ------------------------------------------------------------------ #
    def state_dict(self):
        sd = {
            'lr': self.lr,
            'betas': (self.beta1, self.beta2),
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'master_dtype': self.master_dtype,
            'backend': self._backend,
            'state': {},
        }
        if self._backend == 'deepspeed' and self._ds_opt is not None:
            # delegate the heavy state (m/v/master) to DeepSpeed, remap pid -> FQN
            ds_sd = self._ds_opt.state_dict()
            pid_for = {id(cpu_p): i for i, cpu_p in enumerate(self._ds_params)}
            fqn_for_idx = {pid_for[id(self._ds_param_for[p])]: self._param_fqn[p] for p in self.params}
            sd['deepspeed'] = {'raw': ds_sd, 'idx2fqn': fqn_for_idx}
            sd['state'] = {self._param_fqn[p]: {'step': self.state[p]['step']} for p in self.params}
            return sd
        for p in self.params:
            fqn = self._param_fqn[p]
            st = self.state[p]
            entry = {
                'step': st['step'],
                'exp_avg': st['exp_avg'].detach().cpu(),
                'exp_avg_sq': st['exp_avg_sq'].detach().cpu(),
            }
            if self.use_master:
                entry['master'] = st['master'].detach().cpu()
            sd['state'][fqn] = entry
        return sd

    def load_state_dict(self, sd):
        self.lr = sd.get('lr', self.lr)
        if 'betas' in sd:
            self.beta1, self.beta2 = sd['betas']
        self.eps = sd.get('eps', self.eps)
        self.weight_decay = sd.get('weight_decay', self.weight_decay)
        if sd.get('master_dtype', self.master_dtype) != self.master_dtype:
            logger.warning(f"expert optimizer master_dtype changed "
                           f"({sd.get('master_dtype')} -> {self.master_dtype}); state may be reinitialised.")
        if self._backend == 'deepspeed' and self._ds_opt is not None and 'deepspeed' in sd:
            self._ds_opt.load_state_dict(sd['deepspeed']['raw'])
            for p in self.params:
                fqn = self._param_fqn[p]
                if fqn in sd['state']:
                    self.state[p]['step'] = sd['state'][fqn]['step']
            return
        loaded = sd.get('state', {})
        for p in self.params:
            fqn = self._param_fqn[p]
            if fqn not in loaded:
                logger.warning(f'expert optimizer state for {fqn} missing in checkpoint; keeping init state.')
                continue
            entry = loaded[fqn]
            st = self.state[p]
            st['step'] = int(entry['step'])
            st['exp_avg'].copy_(entry['exp_avg'].to(st['exp_avg'].dtype))
            st['exp_avg_sq'].copy_(entry['exp_avg_sq'].to(st['exp_avg_sq'].dtype))
            if self.use_master and 'master' in entry:
                st['master'].copy_(entry['master'].to(st['master'].dtype))
            elif self.use_master:
                logger.warning(f'no master weight for {fqn} in checkpoint; re-deriving from current weight.')


class ExpertOptimizerCallback(TrainerCallback):
    """Drives `trainer.expert_optimizer` once per global (sync) step.

    Registered only when `trainer.expert_optimizer` exists. `on_pre_optimizer_step` fires
    after grad clipping and before `trainer.optimizer.step()` / `model.zero_grad()`
    (transformers `_run_epoch`), and only on sync steps — so the expert grads are fully
    accumulated and already clipped, and we never step mid-accumulation. The LR is mirrored
    from the main optimizer (whose scheduler steps *after* this hook), so both optimizers
    use the same LR this iteration.
    """

    def __init__(self, trainer):
        self.trainer = trainer

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        expert_opt = getattr(self.trainer, 'expert_optimizer', None)
        if expert_opt is None:
            return control
        # Mirror the LR the main optimizer is about to use (first non-empty group).
        for pg in self.trainer.optimizer.param_groups:
            if len(pg['params']) > 0:
                expert_opt.set_lr(pg['lr'])
                break
        # If all expert grads are None (e.g. _fix_grad_norm_nan zeroed grads on NaN),
        # step() is a no-op, matching the main optimizer being effectively skipped.
        expert_opt.step()
        return control