# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import os
import torch
from typing import Any, Dict, List, Optional

from swift.rl_core.data import GRPOBatch
from swift.utils import gc_collect, get_current_device, get_logger, to_device
from .checkpoint_engine import CheckpointEngineMixin
from .inference_utils import split_micro_batches
from .worker_group import dispatch_collect

logger = get_logger()


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

    @dispatch_collect(dispatch='dp_split', collect='dp_flat')
    def compute_teacher_logits(self, batch) -> None:
        """Forward the teacher and cache per-sample TeacherOutput locally on this worker.

        Core logic lives in GKDLoss.compute_teacher_logits. The result is ALWAYS cached
        worker-local (never returned to the driver): for context parallel each rank must
        inject its own sequence shard at train_step via _inject_cached_teacher_logits.
        """
        if getattr(self._args, '_teacher_use_disable_adapter', False):
            # Self-distillation (LoRA): teacher = student base model with the LoRA adapter
            # disabled — no separate teacher loaded.
            from contextlib import ExitStack
            megatron = self._megatron
            with ExitStack() as stack:
                for m in megatron.peft_models:
                    stack.enter_context(m.disable_adapter())
                self._cached_teacher_logits = self._loss_fn.compute_teacher_logits(megatron.unwrapped_models[0], batch,
                                                                                   self._pipeline.template, self._args)
        else:
            assert self.teacher is not None, 'Teacher model not initialized. Call init_teacher_model first.'
            with self.teacher.loaded_context():
                self._cached_teacher_logits = self._loss_fn.compute_teacher_logits(self.teacher.models[0], batch,
                                                                                   self._pipeline.template, self._args)
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

    @dispatch_collect(dispatch='dp_split', collect='all')
    def train_step(self, batch, extra_metrics=None) -> Dict[str, Any]:
        megatron = self._megatron
        args = megatron.args
        assert isinstance(batch, list), f'train_step expects List[Dict], got {type(batch).__name__}'
        self._inject_cached_teacher_logits(batch)
        self._inject_extra_metrics(extra_metrics)
        micro_batches = self._build_local_micro_batches(batch)
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

    def _inject_cached_teacher_logits(self, batch: list) -> None:
        """Inject worker-local cached teacher outputs into samples.

        The colocated teacher always caches its per-sample ``TeacherOutput``
        (top-k or full-vocab) on the worker's own GPU and returns an empty list
        to the driver — avoiding a driver round-trip. This is REQUIRED for
        context parallel (CP): each CP rank computes and keeps its own sequence
        shard, so injecting the local shard keeps teacher/student CP slices
        aligned (a driver round-trip would only collect the cp-rank-0 shard and
        broadcast it to all ranks, corrupting the alignment).
        """
        cached = getattr(self, '_cached_teacher_logits', None)
        if not cached:
            return
        for sample, t_out in zip(batch, cached):
            if t_out is not None:
                sample['teacher_output'] = t_out
        self._cached_teacher_logits = None

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

    @dispatch_collect(dispatch='dp_split', collect='dp_flat')
    def compute_logps(self, batch) -> Any:
        """Compute per-token logps under the current policy model.

        Receives a local sample shard from DP dispatch and performs
        collate locally before logp computation.
        """
        model = self._megatron.unwrapped_models[0]
        if isinstance(batch, list):
            return self._compute_logps_from_samples(batch, model, 'per_token_logps', with_completion_mask=True)
        result = self._compute_logps_batched(batch, model, 'per_token_logps')
        return self._split_logps_rows(
            result.get('per_token_logps'),
            batch.get('completion_mask'),
            'per_token_logps',
            routed_experts=result.get('routed_experts'),
            seq_lengths=batch.get('seq_lengths'))

    @dispatch_collect(dispatch='dp_split', collect='dp_flat')
    def compute_ref_logps(self, batch) -> Any:
        """Compute per-token logps under the frozen reference model."""
        if self.ref is not None:
            model = self.ref.models[0]
            if isinstance(batch, list):
                return self._compute_logps_from_samples(batch, model, 'ref_per_token_logps', with_completion_mask=False)
            result = self._compute_logps_batched(batch, model, 'ref_per_token_logps')
            return self._split_logps_rows(
                result.get('ref_per_token_logps'), None, 'ref_per_token_logps', seq_lengths=batch.get('seq_lengths'))
        with self.actor.null_ref_context() as ref_models:
            if isinstance(batch, list):
                return self._compute_logps_from_samples(
                    batch, ref_models[0], 'ref_per_token_logps', with_completion_mask=False)
            result = self._compute_logps_batched(batch, ref_models[0], 'ref_per_token_logps')
        return self._split_logps_rows(
            result.get('ref_per_token_logps'), None, 'ref_per_token_logps', seq_lengths=batch.get('seq_lengths'))

    def _compute_logps_batched(self, batch, model, output_key: str) -> Dict[str, Any]:
        """Split a DP shard into micro-batches, compute logps, concatenate."""
        micro_batches = split_micro_batches(batch, self._args.micro_batch_size)
        parts = []
        routed_parts = []
        for mb in micro_batches:
            mb_gpu = to_device(mb, get_current_device())
            result = self._compute_logps(mb_gpu, model, output_key)
            del mb_gpu
            if output_key in result:
                parts.append(result[output_key])
            if result.get('routed_experts') is not None:
                routed_parts.append(result['routed_experts'])
        if not parts:
            return {}
        out = {output_key: torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]}
        if routed_parts:
            if all(r.dim() > 0 and r.shape[0] == 1 for r in routed_parts):
                out['routed_experts'] = torch.cat(routed_parts, dim=1)
            else:
                out['routed_experts'] = torch.cat(routed_parts, dim=0) if len(routed_parts) > 1 else routed_parts[0]
        del parts
        del routed_parts
        gc_collect()
        return out

    def _compute_logps_from_samples(
        self,
        samples: List[Dict[str, Any]],
        model,
        output_key: str,
        *,
        with_completion_mask: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        rows: List[Dict[str, torch.Tensor]] = []
        for sample_chunk in self._split_sample_chunks(samples):
            mb = self._collate_local_grpo_samples(sample_chunk)
            grpo_batch = mb['grpo_batch']
            out = self._compute_logps(mb, model, output_key)
            if output_key not in out:
                continue
            completion_mask = grpo_batch.completion_mask if with_completion_mask else None
            rows.extend(
                self._split_logps_rows(
                    out[output_key],
                    completion_mask,
                    output_key,
                    routed_experts=out.get('routed_experts'),
                    seq_lengths=grpo_batch.seq_lengths))
        return rows

    @staticmethod
    def _split_logps_rows(
        logps: Optional[torch.Tensor],
        completion_mask: Optional[torch.Tensor],
        key: str,
        *,
        routed_experts: Optional[torch.Tensor] = None,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, torch.Tensor]]:
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
        rows = []
        for i in range(logps.shape[0]):
            item: Dict[str, torch.Tensor] = {key: logps[i].detach().cpu()}
            if completion_mask is not None:
                item['completion_mask'] = completion_mask[i].detach().cpu()
            if routed_rows[i] is not None:
                item['routed_experts'] = routed_rows[i].detach().cpu()
            rows.append(item)
        return rows

    def _compute_logps(self, batch: Dict[str, Any], model, output_key: str) -> Dict[str, Any]:
        """Compute per-token logps for a single micro-batch."""
        from swift.rlhf_trainers.utils import pad_logps_back_to_batch
        megatron = self._megatron
        args = self._args
        temperature = getattr(args, 'temperature', 1.0)
        grpo_batch: GRPOBatch = batch.pop('grpo_batch')
        seq_lengths = grpo_batch.seq_lengths
        batch_size = grpo_batch.completion_mask.shape[0]
        max_seq_len = grpo_batch.completion_mask.shape[1]

        model_inputs = batch
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

        out = {}
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

    def _split_sample_chunks(self, samples: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        micro_batch_size = self._args.micro_batch_size
        return [samples[i:i + micro_batch_size] for i in range(0, len(samples), micro_batch_size)]

    def _build_local_micro_batches(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._collate_local_grpo_samples(chunk) for chunk in self._split_sample_chunks(samples)]

    @staticmethod
    def _normalize_routed_experts_tensor(value: Any) -> torch.Tensor:
        routed = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if routed.dim() >= 4 and routed.shape[0] == 1:
            routed = routed.squeeze(0)
        if routed.dim() < 2:
            raise ValueError(f'Invalid routed_experts shape: {tuple(routed.shape)}')
        return routed.to(dtype=torch.int64)

    @staticmethod
    def _pad_or_trim_routed_experts(routed: torch.Tensor, target_len: int, *, padding_right: bool) -> torch.Tensor:
        current_len = int(routed.shape[0])
        if current_len == target_len:
            return routed
        if current_len > target_len:
            return routed[:target_len] if padding_right else routed[-target_len:]

        pad_len = target_len - current_len
        pad = [0] * (2 * routed.dim())
        if padding_right:
            pad[2 * (routed.dim() - 1) + 1] = pad_len
        else:
            pad[2 * (routed.dim() - 1)] = pad_len
        return torch.nn.functional.pad(routed, tuple(pad), 'constant', 0)

    def _build_routed_experts_batch(
        self,
        samples: List[Dict[str, Any]],
        *,
        seq_lengths: torch.Tensor,
        max_seq_len: int,
        template,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not samples:
            return None
        if all(sample.get('routed_experts') is None for sample in samples):
            return None

        router_mode = getattr(self._args, 'router_replay_mode', 'disabled')
        padding_right = template.padding_side == 'right'
        n_samples = len(samples)

        current_seq_lengths = seq_lengths
        if seq_lengths.size(0) > n_samples:
            current_seq_lengths = seq_lengths[:n_samples].clone()
            current_seq_lengths[n_samples - 1] = seq_lengths[n_samples - 1:].sum()

        routed_tensors: List[torch.Tensor] = []
        for sample, cur_seq_len in zip(samples, current_seq_lengths):
            routed_value = sample.get('routed_experts')
            if routed_value is None:
                if router_mode == 'R3':
                    raise AssertionError('When router_replay_mode = R3, routed_experts must be in rollout data')
                return None

            routed = self._normalize_routed_experts_tensor(routed_value)

            expected_len = sample.get('encoded', {}).get('length')
            experts_seq_len = int(routed.shape[0])
            if router_mode == 'R3' and expected_len is not None:
                if experts_seq_len not in (expected_len, expected_len - 1):
                    raise AssertionError(
                        f'The seq_len of routed_experts({experts_seq_len}) does not match encoded length '
                        f'({expected_len}); expected same length or one less.')
            target_len = int(cur_seq_len.item()) if template.padding_free else max_seq_len
            routed = self._pad_or_trim_routed_experts(routed, target_len, padding_right=padding_right)
            routed_tensors.append(routed)

        if template.padding_free:
            return torch.cat(routed_tensors, dim=0).unsqueeze(0).to(device=device)
        return torch.stack(routed_tensors).to(device=device)

    @staticmethod
    def _collate_teacher_outputs(teacher_outputs, device, padding_free=False, target_seq_len=None):
        """Collate per-sample TeacherOutputs into a batched one.

        Only used for topk mode; full_logits mode uses _cached_teacher_logits
        (kept on-GPU per TP rank) and bypasses this path entirely.

        Layout must match the student labels:
        - padding_free: student labels are [1, total_tokens] (samples concatenated
          along the sequence dim), so concatenate per-sample teacher tensors along
          dim=1. Empty placeholders ([0, ...], emitted by the colocated path for all
          but the first sample of a micro-batch) are dropped first.
        - otherwise: stack per-sample tensors along the batch dim (dim=0).

        ``target_seq_len`` is the student's collated sequence length. The student
        collation pads the sequence to a multiple via ``get_padding_to`` (SP/CP/fp8),
        so the standalone-teacher tensors (built from each sample's raw, unpadded
        length) can be 1+ tokens short. Pad the teacher seq dim to ``target_seq_len``
        so extract_active's label mask aligns; the padded tail has labels == -100 and
        is masked out, leaving the loss unchanged.

        OPSD: when ``labels`` is present on the TeacherOutput, the teacher scores a *different*
        prompt (so its sequence length differs from the student) and extract_active
        aligns by mask (``labels != -100``), not by position. In that case
        the teacher keeps its own length (``target_seq_len`` is ignored).
        """
        from swift.rlhf_trainers.gkd_loss import TeacherOutput
        opsd = any(getattr(t, 'labels', None) is not None for t in teacher_outputs)
        effective_target = None if opsd else target_seq_len
        pad_vals = {'topk_logprobs': float('-inf'), 'labels': -100}
        fields = ('full_logits', 'topk_logprobs', 'topk_indices', 'labels')
        kwargs = {}
        for field in fields:
            tensors = [getattr(t, field) for t in teacher_outputs]
            tensors = [t for t in tensors if t is not None]
            if not tensors:
                continue
            pad_val = pad_vals.get(field, 0)
            if padding_free:
                non_empty = [t for t in tensors if t.shape[0] > 0]
                cat = torch.cat(non_empty, dim=1)
                if effective_target is not None and cat.dim() == 3 and cat.shape[1] < effective_target:
                    cat = torch.nn.functional.pad(cat, (0, 0, 0, effective_target - cat.shape[1]), value=pad_val)
                kwargs[field] = cat.to(device)
            else:
                max_len = effective_target or max(t.shape[1] for t in tensors)
                padded = []
                for t in tensors:
                    # labels is 2D [1, S]; topk_*/full_logits are 3D [1, S, *].
                    # Pad the sequence dim (dim=1) of either to max_len so torch.cat works
                    # even when per-sample teacher seq lengths differ within a micro-batch.
                    if t.dim() == 3 and t.shape[1] < max_len:
                        t = torch.nn.functional.pad(t, (0, 0, 0, max_len - t.shape[1]), value=pad_val)
                    elif t.dim() == 2 and t.shape[1] < max_len:
                        t = torch.nn.functional.pad(t, (0, max_len - t.shape[1]), value=pad_val)
                    padded.append(t)
                kwargs[field] = torch.cat(padded, dim=0).to(device)
        return TeacherOutput(**kwargs)

    def _collate_local_grpo_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        from swift.megatron.utils import get_padding_to
        from swift.rlhf_trainers.utils import build_completion_mask_and_seq_lengths, build_rollout_logps

        template = self._pipeline.template
        device = get_current_device()
        encoded_list = [sample['encoded'] for sample in samples]
        encoded_batch = template.data_collator(encoded_list, padding_to=get_padding_to(self._args))
        encoded_batch = to_device(encoded_batch, device)

        labels = encoded_batch['labels']
        batch_size = len(samples)
        completion_mask, seq_lengths, max_seq_len = build_completion_mask_and_seq_lengths(
            labels,
            batch_size,
            padding_free=template.padding_free,
            encoded_batch=encoded_batch,
            device=device,
        )
        rollout_logps = build_rollout_logps([s.get('rollout_logprobs') for s in samples], completion_mask, device)
        routed_experts = self._build_routed_experts_batch(
            samples, seq_lengths=seq_lengths, max_seq_len=max_seq_len, template=template, device=device)
        grpo_batch = GRPOBatch(
            completion_mask=completion_mask,
            truncated_mask=torch.tensor([bool(s.get('is_truncated', False)) for s in samples],
                                        dtype=torch.bool,
                                        device=device),
            seq_lengths=seq_lengths,
            rollout_per_token_logps=rollout_logps,
        )
        if routed_experts is not None:
            encoded_batch['routed_experts'] = routed_experts

        if samples and samples[0].get('old_per_token_logps') is not None:
            grpo_batch.old_per_token_logps = torch.stack([s['old_per_token_logps'].to(device) for s in samples], dim=0)
        if samples and samples[0].get('ref_per_token_logps') is not None:
            grpo_batch.ref_per_token_logps = torch.stack([s['ref_per_token_logps'].to(device) for s in samples], dim=0)
        if samples and samples[0].get('advantage') is not None:
            grpo_batch.advantages = torch.tensor([float(s.get('advantage', 0.0)) for s in samples],
                                                 dtype=torch.float32,
                                                 device=device)
        encoded_batch['grpo_batch'] = grpo_batch
        if samples and samples[0].get('teacher_output') is not None:
            cp_size = getattr(self._args, 'context_parallel_size', 1)
            if cp_size > 1:
                raise ValueError('Standalone teacher replicas (teacher.gpus > 0) do not support '
                                 'context_parallel_size > 1: per-sample teacher token-logprobs are built from '
                                 'raw sequence lengths and cannot be CP-sharded to align with the student. '
                                 'Use a colocated teacher_model for CP>1.')
            encoded_batch['teacher_output'] = self._collate_teacher_outputs([s['teacher_output'] for s in samples],
                                                                            device,
                                                                            padding_free=template.padding_free,
                                                                            target_seq_len=labels.shape[-1])
        if samples and samples[0].get('data_source') is not None:
            encoded_batch['data_source'] = samples[0]['data_source']
        return encoded_batch

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
