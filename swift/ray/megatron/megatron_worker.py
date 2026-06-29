# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import os
import torch
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from swift.rl_core.data import GRPOBatch
from swift.utils import gc_collect, get_current_device, get_logger
from .checkpoint_engine import CheckpointEngineMixin
from .worker_group import dispatch_collect

if TYPE_CHECKING:
    from swift.rlhf_trainers.gkd_loss import TeacherOutput

logger = get_logger()

# --- worker-side data-flow types ------------------------------------------
# Since collation moved to the driver, the worker consumes already-collated
# micro-batches and returns per-sample logps:
#
# ModelInputs: a collated micro-batch (driver-side ``collate_to_grpo_micro_batch``)
#   fed to ``model(**model_inputs)``. Carries the model-forward tensors plus
#   ``grpo_batch`` (GRPOBatch) and optional ``teacher_output`` / ``teacher_model_inputs``
#   / ``data_source``.
# LogpsRow: one per-sample logps result returned worker→driver (collected via
#   ``collect='dp_flat'``), e.g. ``{'per_token_logps': ..., 'completion_mask': ...}``.
ModelInputs = Dict[str, Any]
LogpsRow = Dict[str, torch.Tensor]


def _import_class(dotted_path: str):
    """Import a class from a dotted module path like ``'a.b.ClassName'``."""
    import importlib
    mod_path, cls_name = dotted_path.rsplit('.', 1)
    return getattr(importlib.import_module(mod_path), cls_name)


def _make_lifecycle_trainer(args, template):
    from swift.megatron.trainers.rlhf_mixin import MegatronRLHFTrainer

    class _LifecycleTrainer(MegatronRLHFTrainer):

        def forward_step(self, data_iterator, model):
            return None

    return _LifecycleTrainer(args, template)


class MegatronWorker(CheckpointEngineMixin):

    def __init__(self):
        self._megatron = None
        self._loss_fn = None
        self._args = None
        self._pipeline = None
        self.rollout = None
        self._checkpoint_engine = None
        self._bucket_size: int = 3072 << 20
        self.actor = None  # TrainableModelWorker
        self.ref = None  # MegatronModelWorker (explicit ref for full fine-tune)
        self.teacher = None  # MegatronModelWorker (colocated teacher)

    def init_actor(
        self,
        cfg: Dict[str, Any],
        loss_cls_path: Optional[str] = None,
        rollout_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise the training (actor) model, optimizer, and optionally the rollout adapter.

        This only sets up the actor model for training. Ref and teacher models
        are initialized separately (ref in _init_trainable, teacher via
        init_teacher_model).

        Args:
            cfg: Merged config dict (shared + group overrides).
            rollout_config: When provided, creates an internal RolloutAdapter.
        """
        from swift.megatron.arguments import MegatronRLHFArguments
        from swift.megatron.pipelines.train.rlhf import MegatronRLHF
        from .driver_utils import parse_args_from_dict

        args = parse_args_from_dict(MegatronRLHFArguments, cfg)
        self._pipeline = MegatronRLHF(args)
        self._loss_cls_path = loss_cls_path
        self._args = self._pipeline.args

        self._init_trainable()

        if rollout_config:
            self._init_rollout_adapter(rollout_config)

    def _init_trainable(self):
        from .model_worker import TrainableModelWorker

        pipeline = self._pipeline
        args = pipeline.args
        self._megatron = _make_lifecycle_trainer(args, pipeline.template)

        if self._loss_cls_path:
            loss_cls = _import_class(self._loss_cls_path)
            self._loss_fn = loss_cls(args)
            self._megatron.forward_step = self._loss_fn.forward_step

        self.actor = TrainableModelWorker(args, self._megatron)
        if args.tuner_type == 'full' and self._megatron.ref_models:
            from .model_worker import MegatronModelWorker
            self.ref = MegatronModelWorker(args, self._megatron.ref_models)

    @dispatch_collect(dispatch='broadcast', collect='first')
    def init_teacher_model(self, model_dir: str):
        """Load a colocated teacher model (same parallelism as student)."""
        from .model_worker import MegatronModelWorker

        # Prefer the worker's own resolved teacher_model_dir; bridge.load_weights needs a
        # real local path to locate safetensors (a raw model id yields an empty state dict).
        model_dir = getattr(self._args, 'teacher_model_dir', None) or model_dir
        self.teacher = MegatronModelWorker.from_pretrained(self._args, model_dir)
        logger.info('Colocated teacher model loaded from %s', model_dir)
        if getattr(self._args, 'offload_teacher_model', False):
            self.teacher.offload_to_cpu()

    @dispatch_collect(dispatch='dp', collect='first')
    def compute_teacher_logits(self, micro_batches: List[ModelInputs]) -> None:
        """Forward the teacher on each driver-collated micro-batch's ``teacher_model_inputs``
        and cache one batched ``TeacherOutput`` per micro-batch (worker-local).

        Cached (never returned to the driver): for context parallel each rank forwards its
        own sequence shard, so the cache is already the correct local slice. ``train_step``
        attaches it to the matching micro-batch (same dispatch dict ⇒ same order).
        """
        teacher_inputs = [mi['teacher_model_inputs'] for mi in micro_batches]
        if getattr(self._args, '_teacher_use_disable_adapter', False):
            # Self-distillation (LoRA): teacher = student base model with the LoRA adapter
            # disabled — no separate teacher loaded.
            from contextlib import ExitStack
            megatron = self._megatron
            with ExitStack() as stack:
                for m in megatron.peft_models:
                    stack.enter_context(m.disable_adapter())
                self._cached_teacher_logits = self._loss_fn.compute_teacher_logits(megatron.unwrapped_models[0],
                                                                                   teacher_inputs, self._args)
        else:
            assert self.teacher is not None, 'Teacher model not initialized. Call init_teacher_model first.'
            with self.teacher.loaded_context():
                self._cached_teacher_logits = self._loss_fn.compute_teacher_logits(self.teacher.models[0],
                                                                                   teacher_inputs, self._args)
        gc_collect()

    def get_parallel_info(self) -> Dict[str, Any]:
        from megatron.core import mpu
        info = {
            'dp_rank':
            mpu.get_data_parallel_rank(),
            'dp_size':
            mpu.get_data_parallel_world_size(),
            'is_collector':
            (mpu.get_tensor_model_parallel_rank() == 0
             and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
             and mpu.get_context_parallel_rank() == 0),
        }
        return info

    def get_padding_to(self) -> Optional[int]:
        """Delegates to ``swift.megatron.utils.get_padding_to`` (handles SP, CP, fp8)."""
        from swift.megatron.utils import get_padding_to
        return get_padding_to(self._args)

    @dispatch_collect(dispatch='broadcast', collect='first')
    def setup(self, args_override: Dict[str, Any]) -> Dict[str, Any]:
        """Apply pre-computed args from driver, then set up model training.

        The driver is responsible for computing train_iters, eval_iters,
        save_steps, etc.  The worker just applies the overrides.
        """
        megatron = self._megatron
        for k, v in args_override.items():
            if v is not None:
                setattr(megatron.args, k, v)
        megatron.setup_model_training()
        return {
            'train_iters': megatron.args.train_iters,
            'iteration': megatron.state.iteration,
        }

    @dispatch_collect(dispatch='dp', collect='all')
    def train_step(self,
                   micro_batches: List[ModelInputs],
                   extra_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from swift.utils import to_device
        megatron = self._megatron
        args = megatron.args
        assert isinstance(micro_batches, list), \
            f'train_step expects List[ModelInputs], got {type(micro_batches).__name__}'
        self._inject_extra_metrics(extra_metrics)
        # GKD colocated teacher: attach the per-micro-batch TeacherOutput cached by
        # compute_teacher_logits (worker-local, already CP-correct), aligned by order.
        cached_teacher = getattr(self, '_cached_teacher_logits', None)
        if cached_teacher is not None:
            for mi, t_out in zip(micro_batches, cached_teacher):
                if t_out is not None:
                    mi['teacher_output'] = t_out
            self._cached_teacher_logits = None
        # Driver-side collate produces CPU tensors; move each micro-batch to the GPU.
        device = get_current_device()
        moved: List[ModelInputs] = []
        for mi in micro_batches:
            mi.pop('teacher_model_inputs', None)  # consumed by compute_teacher_logits
            grpo_batch = mi.pop('grpo_batch', None)
            teacher_output = mi.pop('teacher_output', None)  # GPU (cache) or CPU (replicas)
            mi = to_device(mi, device)
            if grpo_batch is not None:
                mi['grpo_batch'] = grpo_batch.to_device(device)
            if teacher_output is not None:
                mi['teacher_output'] = teacher_output.to_device(device)
            moved.append(mi)
        micro_batches = moved
        data_iterator = iter(micro_batches)
        assert len(micro_batches) == args.num_microbatches, (
            f'Worker got {len(micro_batches)} micro-batches but args.num_microbatches='
            f'{args.num_microbatches}; check per_device_generation_batch_size / micro_batch_size config.')
        router_replay_mode = getattr(args, 'router_replay_mode', 'disabled')
        need_routing_replay = router_replay_mode != 'disabled'
        RouterReplay = None
        if need_routing_replay:
            try:
                from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
                RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
            except ImportError:
                need_routing_replay = False

        try:
            megatron.run_train_step(data_iterator, None)
        finally:
            if need_routing_replay and RouterReplay is not None:
                RouterReplay.clear_global_indices()
                RouterReplay.clear_global_router_replay_action()
        del data_iterator
        gc_collect()
        return self._extract_step_metrics(megatron)

    def _inject_extra_metrics(self, extra_metrics) -> None:
        """Inject driver-computed metrics (reward, MathAccuracy, data_source, ...) into
        the megatron trainer's ``_train_metrics`` so they flow through the standard
        ``on_log`` path (console PrintCallback + tensorboard + swanlab), unifying ALL
        logging in the worker's megatron callbacks (the driver no longer prints metrics).

        Values are stored as ``[sum, count]`` pairs to match ``_aggregated_metrics`` /
        ``_log_callback`` (which divides sum/count), so a per-step scalar logs as itself.
        """
        if not extra_metrics:
            return
        megatron = self._megatron
        tm = getattr(megatron, '_train_metrics', None)
        if tm is None:
            tm = megatron._train_metrics = {}
        device = get_current_device()
        for k, v in extra_metrics.items():
            if v is None:
                continue
            add = torch.tensor([float(v), 1.0], dtype=torch.float32, device=device)
            tm[k] = tm[k] + add if k in tm else add

    @staticmethod
    def _extract_step_metrics(megatron) -> Dict[str, Any]:
        """Extract training metrics from the last logged step.

        After ``run_train_step``, Megatron's ``on_log`` stores metrics
        in ``_last_logged_metrics``.  We extract numeric values and
        normalize key names (e.g. ``learning_rate`` → ``lr``) so the
        driver receives a clean dict for aggregation and logging.
        """
        result: Dict[str, Any] = {'iteration': megatron.state.iteration}
        logged = getattr(megatron, '_last_logged_metrics', None) or {}
        for k, v in logged.items():
            if isinstance(v, (int, float)):
                result[k] = v
            else:
                try:
                    result[k] = float(v)
                except (TypeError, ValueError):
                    continue
        if 'learning_rate' in result:
            result['lr'] = result.pop('learning_rate')
        return result

    @dispatch_collect(dispatch='dp', collect='dp_flat')
    def compute_logps(self, micro_batches: List[ModelInputs]) -> List[LogpsRow]:
        """Compute per-token logps under the current policy model.

        Receives this dp_rank's collated micro-batches (driver-side collate, CPU)
        and runs the rank-local forward; returns one row per sample.
        """
        model = self._megatron.unwrapped_models[0]
        return self._compute_logps_micro_batches(micro_batches, model, 'per_token_logps')

    @dispatch_collect(dispatch='dp', collect='dp_flat')
    def compute_ref_logps(self, micro_batches: List[ModelInputs]) -> List[LogpsRow]:
        """Compute per-token logps under the frozen reference model."""
        if self.ref is not None:
            return self._compute_logps_micro_batches(micro_batches, self.ref.models[0], 'ref_per_token_logps')
        with self.actor.null_ref_context() as ref_models:
            return self._compute_logps_micro_batches(micro_batches, ref_models[0], 'ref_per_token_logps')

    @dispatch_collect(dispatch='dp', collect='dp_flat')
    def compute_teacher_logps(self, micro_batches: List[ModelInputs]) -> List[LogpsRow]:
        """OPD-RL: per-token teacher logp on the sampled tokens (token-in-token-out).

        Same forward/frame as ``compute_logps`` so the teacher logp aligns with the policy's;
        same-model LoRA self-distillation disables the student's adapter, otherwise the colocated
        teacher (loaded via ``init_teacher_model``) is used.
        """
        if getattr(self._args, '_teacher_use_disable_adapter', False):
            from contextlib import ExitStack
            with ExitStack() as stack:
                for m in self._megatron.peft_models:
                    stack.enter_context(m.disable_adapter())
                return self._compute_logps_micro_batches(micro_batches, self._megatron.unwrapped_models[0],
                                                         'teacher_per_token_logps')
        # Dynamic self-distillation (teacher is None): teacher = student (same weights
        # including LoRA). No offload/load needed.
        model = self.teacher.models[0] if self.teacher else self._megatron.unwrapped_models[0]
        with (self.teacher.loaded_context() if self.teacher else nullcontext()):
            return self._compute_logps_micro_batches(micro_batches, model, 'teacher_per_token_logps')

    def _compute_logps_micro_batches(
        self,
        micro_batches: List[ModelInputs],
        model,
        output_key: str,
    ) -> List[LogpsRow]:
        from swift.utils import to_device
        device = get_current_device()
        rows: List[LogpsRow] = []
        for model_inputs in micro_batches:
            grpo_batch = model_inputs.pop('grpo_batch').to_device(device)
            model_inputs = to_device(model_inputs, device)
            model_inputs['grpo_batch'] = grpo_batch  # _compute_logps pops it again
            out = self._compute_logps(model_inputs, model, output_key)
            if output_key not in out:
                continue
            rows.extend(
                self._split_logps_rows(
                    out[output_key],
                    output_key,
                    routed_experts=out.get('routed_experts'),
                    seq_lengths=grpo_batch.seq_lengths))
        return rows

    @staticmethod
    def _split_logps_rows(
        logps: Optional[torch.Tensor],
        key: str,
        *,
        routed_experts: Optional[torch.Tensor] = None,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> List[LogpsRow]:
        if logps is None:
            return []
        routed_rows: List[Optional[torch.Tensor]] = [None] * int(logps.shape[0])
        if routed_experts is not None:
            routed = routed_experts.detach().cpu() if isinstance(routed_experts, torch.Tensor) \
                else torch.as_tensor(routed_experts)
            if routed.dim() > 0 and routed.shape[0] == logps.shape[0]:
                routed_rows = [routed[i] for i in range(logps.shape[0])]
            elif routed.dim() > 1 and routed.shape[0] == 1 and seq_lengths is not None:
                seq_cpu = seq_lengths.detach().cpu().tolist()
                start = 0
                for i in range(logps.shape[0]):
                    seq_len = int(seq_cpu[i])
                    end = start + seq_len
                    routed_rows[i] = routed[0, start:end]
                    start = end
        rows: List[LogpsRow] = []
        for i in range(logps.shape[0]):
            item: LogpsRow = {key: logps[i].detach().cpu()}
            if routed_rows[i] is not None:
                item['routed_experts'] = routed_rows[i].detach().cpu()
            rows.append(item)
        return rows

    def _compute_logps(self, model_inputs: ModelInputs, model, output_key: str) -> Dict[str, torch.Tensor]:
        """Compute per-token logps for a single collated micro-batch."""
        from swift.rlhf_trainers.utils import pad_logps_back_to_batch
        megatron = self._megatron
        args = self._args
        temperature = getattr(args, 'temperature', 1.0)
        # grpo_batch carries the per-batch masks/seq_lengths; pop it so what
        # remains is the pure ``model(**model_inputs)`` forward kwargs.
        grpo_batch: GRPOBatch = model_inputs.pop('grpo_batch')
        seq_lengths = grpo_batch.seq_lengths
        batch_size = grpo_batch.completion_mask.shape[0]
        max_seq_len = grpo_batch.completion_mask.shape[1]

        enable_routing_replay = bool(getattr(megatron, 'enable_routing_replay', False))
        router_mode = getattr(args, 'router_replay_mode', 'disabled')

        RouterReplay = None
        RouterReplayAction = None
        if enable_routing_replay:
            try:
                from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
            except ImportError:
                enable_routing_replay = False

        if enable_routing_replay and RouterReplay is not None:
            if router_mode == 'R2':
                RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
            elif router_mode == 'R3':
                RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

        routing_topk_idx = None
        try:
            logps_packed, routing_topk_idx = megatron.compute_per_token_logps(
                model, iter([model_inputs]), temperature=temperature)
        finally:
            if enable_routing_replay and RouterReplay is not None:
                RouterReplay.clear_global_indices()
                RouterReplay.clear_global_router_replay_action()

        out: Dict[str, torch.Tensor] = {}
        if logps_packed is not None:
            if args.padding_free:
                logps, _ = pad_logps_back_to_batch(
                    logps_rmpad=logps_packed,
                    logits_to_keep=max_seq_len,
                    batch_size=batch_size,
                    seq_lengths=seq_lengths)
            else:
                logps = logps_packed
            out[output_key] = logps.detach().cpu()
        if routing_topk_idx is not None:
            out['routed_experts'] = routing_topk_idx.detach().cpu()
        return out

    @dispatch_collect(dispatch='broadcast', collect='first')
    def finalize(self) -> Dict[str, Any]:
        from swift.utils import is_last_rank
        megatron = self._megatron
        megatron.finalize_training()
        self._pipeline._handle_trainer_state(megatron, is_last_rank())
        state = megatron.state
        return {
            'last_model_checkpoint': state.last_model_checkpoint,
            'best_model_checkpoint': state.best_model_checkpoint,
            'best_metric': state.best_metric,
        }

    def _init_rollout_adapter(self, rollout_config: Dict[str, Any]) -> None:
        """Create the internal RolloutAdapter.

        The adapter lazily resolves the VllmServer handle via named actor,
        so it can be created before the server is fully started.
        """
        from .rollout.adapter import RolloutAdapter

        tp = rollout_config['rollout_tp_size']
        dp = rollout_config['rollout_dp_size']
        world_per_replica = tp * dp
        rank = int(os.environ.get('RANK', '0'))
        replica_rank = rank // world_per_replica
        rollout_rank = rank % world_per_replica
        bucket_mb = rollout_config.get('bucket_size_mb', 2048)

        self.rollout = RolloutAdapter(
            replica_rank=replica_rank,
            rollout_rank=rollout_rank,
            bucket_size_mb=bucket_mb,
        )
        logger.info('MegatronWorker[rank=%s]: rollout adapter created (replica=%d, rollout_rank=%d)', rank,
                    replica_rank, rollout_rank)

    @dispatch_collect(dispatch='broadcast', collect='first')
    def merge_lora(self):
        """Merge LoRA adapters into base weights (must be called before offload)."""
        megatron = self._megatron
        if megatron.args.tuner_type in ('lora', 'lora_llm'):
            megatron.merge_lora_adapters()

    @dispatch_collect(dispatch='broadcast', collect='first')
    def unmerge_lora(self):
        """Unmerge LoRA adapters to restore training state (call after reload)."""
        megatron = self._megatron
        if megatron.args.tuner_type in ('lora', 'lora_llm'):
            megatron.unmerge_lora_adapters()

    @dispatch_collect(dispatch='broadcast', collect='first')
    def update_weights(self, adapter_only: bool = False):
        """Push training weights to rollout via IPC (streaming).

        All TP ranks must call export_weights (contains TP collectives).
        Only the primary rank sends; others drain the iterator.

        For full-weight sync with LoRA, the caller must ensure merge_lora()
        was called beforehand and unmerge_lora() is called after reload.

        Args:
            adapter_only: When True, export only LoRA adapter weights
                (peft_format=True) and pass peft_config to vLLM for
                TensorLoRARequest loading. When False, export full
                merged weights.
        """
        megatron = self._megatron
        target_device = 'cpu' if megatron.args.offload_bridge else None

        if adapter_only:
            weight_iter = megatron.bridge.export_weights(
                megatron.unwrapped_models, target_device=target_device, peft_format=True)
            peft_config = self.get_peft_config_dict()
            lora_names = None
        else:
            weight_iter = megatron.bridge.export_weights(megatron.unwrapped_models, target_device=target_device)
            peft_config = None
            lora_names = self._resolve_lora_param_names()

        if self.rollout is None:
            # No rollout adapter attached just drain the
            # iterator so all TP ranks finish the collective export.
            for _ in weight_iter:
                pass
            return

        self.rollout.update_weights(
            weight_iter,
            vllm_lora_param_names=lora_names,
            peft_config=peft_config,
            base_sync_done=adapter_only,
        )
        self.rollout.reset_prefix_cache()

    def _resolve_lora_param_names(self) -> Optional[set]:
        """Get vLLM param names for LoRA mapping, if applicable."""
        megatron = self._megatron
        if not (megatron.args.tuner_type == 'lora' and megatron.args.vllm_enable_lora):
            return None
        raw_names = self.rollout.get_model_param_names()
        if not raw_names:
            return None
        from swift.rlhf_trainers.utils import expand_vllm_param_name_aliases
        expanded = expand_vllm_param_name_aliases(set(raw_names))
        stripped = set()
        for n in expanded:
            stripped.add(n)
            if n.startswith('model.'):
                stripped.add(n[len('model.'):])
        return stripped

    @dispatch_collect(dispatch='broadcast', collect='first')
    def finalize_generation(self):
        if self.rollout is not None:
            self.rollout.reset_prefix_cache()

    @dispatch_collect(dispatch='broadcast', collect='first')
    def offload_to_cpu(self):
        self.actor.offload_to_cpu()
        if self.ref is not None:
            self.ref.offload_to_cpu()
        if self.teacher is not None:
            self.teacher.offload_to_cpu()

    @dispatch_collect(dispatch='broadcast', collect='first')
    def reload_to_gpu(self):
        self.actor.reload_to_gpu()
        if self.ref is not None:
            self.ref.reload_to_gpu(load_grad=False)
        # When offload_teacher_model is set, the teacher is managed by compute_teacher_logits
        # (loaded only for the teacher forward), so keep it on CPU here.
        if self.teacher is not None and not getattr(self._args, 'offload_teacher_model', False):
            self.teacher.reload_to_gpu(load_grad=False)

    @staticmethod
    def _align_seq_len(t, target_len, pad_val=0):
        """Pad or truncate a tensor along dim=1 to target_len. Works for 2D [B,S] and 3D [B,S,*]."""
        cur = t.shape[1]
        if cur == target_len:
            return t
        if cur < target_len:
            pad = (0, target_len - cur) if t.dim() == 2 else (0, 0, 0, target_len - cur)
            return torch.nn.functional.pad(t, pad, value=pad_val)
        return t[:, :target_len]

    @staticmethod
    def _collate_teacher_outputs(
        teacher_outputs: List['TeacherOutput'],
        device: torch.device,
        padding_free: bool = False,
        target_seq_len: Optional[int] = None,
        is_opsd: bool = False,
    ) -> 'TeacherOutput':
        """Collate per-sample TeacherOutputs into a batched one (driver-side).

        For non-OPSD: each tensor is aligned to target_seq_len (pad or truncate).
        For OPSD: teacher keeps its own length (target_seq_len ignored).
        """
        from swift.rlhf_trainers.gkd_loss import TeacherOutput
        effective_target = None if is_opsd else target_seq_len
        pad_vals = {'topk_logprobs': float('-inf'), 'labels': -100}
        fields = ('full_logits', 'topk_logprobs', 'topk_indices', 'labels')
        kwargs = {}
        for field in fields:
            tensors = [getattr(t, field) for t in teacher_outputs]
            tensors = [t for t in tensors if t is not None]
            if not tensors:
                continue
            pad_val = pad_vals.get(field, 0)
            if effective_target is not None:
                tensors = [MegatronWorker._align_seq_len(t, effective_target, pad_val) for t in tensors]
            if padding_free:
                non_empty = [t for t in tensors if t.shape[0] > 0]
                kwargs[field] = torch.cat(non_empty, dim=1).to(device)
            else:
                kwargs[field] = torch.cat(tensors, dim=0).to(device)
        return TeacherOutput(**kwargs)

    def send_checkpoint_weights(self, adapter_only: bool = False) -> None:
        """Export and send model weights via NCCL checkpoint engine."""
        import asyncio
        megatron = self._megatron
        engine = self._get_or_create_checkpoint_engine()
        target_device = 'cpu' if megatron.args.offload_bridge else None
        weight_iter = megatron.bridge.export_weights(
            megatron.unwrapped_models, target_device=target_device, peft_format=adapter_only)
        asyncio.run(engine.send_weights(weight_iter))

    def get_peft_config_dict(self) -> dict:
        """Return the PEFT config for LoRA-only sync."""
        from dataclasses import asdict
        peft_config = self._megatron.unwrapped_models[0].peft_config['default']
        return asdict(peft_config)

    def shutdown(self):
        self.rollout = None
        self._megatron = None
        self._loss_fn = None
        self._checkpoint_engine = None
        self.actor = None
        self.ref = None
        self.teacher = None
