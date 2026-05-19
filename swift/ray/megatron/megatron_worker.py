# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import os
import torch
from typing import Any, Dict, List, Optional, Sequence

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

    def init_model(
        self,
        cfg: Dict[str, Any],
        loss_cls_path: Optional[str] = None,
        rollout_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise the Megatron model and (optionally) the rollout adapter.

        Args:
            cfg: Merged config dict (shared + group overrides).
                 Parsed via ``HfArgumentParser.parse_dict`` and the resulting
                 args instance is passed directly to ``MegatronRLHF``.
            rollout_config: When provided, creates an internal RolloutAdapter.
                Expected keys: ``rollout_tp_size``, ``rollout_dp_size``,
                ``bucket_size_mb`` (optional, default 2048).
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
        pipeline = self._pipeline
        args = pipeline.args

        # reuse megatron trainer to
        self._megatron = _make_lifecycle_trainer(args, pipeline.template)

        if self._loss_cls_path:
            loss_cls = _import_class(self._loss_cls_path)
            self._loss_fn = loss_cls(args)
            self._megatron.forward_step = self._loss_fn.forward_step

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
    def train_step(self, batch) -> Dict[str, Any]:
        megatron = self._megatron
        args = megatron.args
        # Megatron Core reads args.num_microbatches for gradient accumulation
        # scheduling. In Ray, the actual batch per worker is determined by DP
        # dispatch and may differ from the initial config. Temporarily override
        # to match the actual micro-batch count, then restore.
        orig_num_microbatches = args.num_microbatches
        orig_micro_batch_size = args.micro_batch_size
        if isinstance(batch, list):
            micro_batches = self._build_local_micro_batches(batch)
            data_iterator = iter(micro_batches)
        else:
            data_iterator, micro_batches = self._batch_to_iterator(batch)
        args.num_microbatches = len(micro_batches)
        if micro_batches:
            mb0 = micro_batches[0].get('input_ids')
            if isinstance(mb0, torch.Tensor) and mb0.dim() > 0:
                args.micro_batch_size = int(mb0.shape[0])

        try:
            megatron.run_train_step(data_iterator, None)
        finally:
            args.num_microbatches = orig_num_microbatches
            args.micro_batch_size = orig_micro_batch_size
        del data_iterator
        gc_collect()
        return self._extract_step_metrics(megatron)

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
        return self._split_logps_rows(result.get('per_token_logps'), batch.get('completion_mask'), 'per_token_logps')

    @dispatch_collect(dispatch='dp_split', collect='dp_flat')
    def compute_ref_logps(self, batch) -> Any:
        """Compute per-token logps under the frozen reference model."""
        megatron = self._megatron
        with megatron.null_ref_context() as ref_models:
            if isinstance(batch, list):
                return self._compute_logps_from_samples(
                    batch, ref_models[0], 'ref_per_token_logps', with_completion_mask=False)
            result = self._compute_logps_batched(batch, ref_models[0], 'ref_per_token_logps')
        return self._split_logps_rows(result.get('ref_per_token_logps'), None, 'ref_per_token_logps')

    def _compute_logps_batched(self, batch, model, output_key: str) -> Dict[str, Any]:
        """Split a DP shard into micro-batches, compute logps, concatenate."""
        micro_batches = split_micro_batches(batch, getattr(self._args, 'micro_batch_size', 1))
        parts = []
        for mb in micro_batches:
            mb_gpu = to_device(mb, get_current_device())
            result = self._compute_logps(mb_gpu, model, output_key)
            del mb_gpu
            if output_key in result:
                parts.append(result[output_key])
        if not parts:
            return {}
        out = {output_key: torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]}
        del parts
        gc_collect()
        return out

    def _compute_logps_from_samples(
        self,
        samples: Sequence[Dict[str, Any]],
        model,
        output_key: str,
        *,
        with_completion_mask: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        rows: List[Dict[str, torch.Tensor]] = []
        for sample_chunk in self._split_sample_chunks(samples):
            mb = self._collate_local_grpo_samples(sample_chunk)
            out = self._compute_logps(mb, model, output_key)
            if output_key not in out:
                continue
            completion_mask = mb.get('completion_mask') if with_completion_mask else None
            rows.extend(self._split_logps_rows(out[output_key], completion_mask, output_key))
        return rows

    @staticmethod
    def _split_logps_rows(logps: Optional[torch.Tensor], completion_mask: Optional[torch.Tensor],
                          key: str) -> List[Dict[str, torch.Tensor]]:
        if logps is None:
            return []
        rows = []
        for i in range(logps.shape[0]):
            item: Dict[str, torch.Tensor] = {key: logps[i].detach().cpu()}
            if completion_mask is not None:
                item['completion_mask'] = completion_mask[i].detach().cpu()
            rows.append(item)
        return rows

    def _compute_logps(self, batch: Dict[str, Any], model, output_key: str) -> Dict[str, Any]:
        """Compute per-token logps for a single micro-batch."""
        from swift.megatron.trainers.grpo_trainer import GRPO_NON_MODEL_KEYS
        from swift.rlhf_trainers.utils import pad_logps_back_to_batch
        megatron = self._megatron
        args = self._args
        temperature = getattr(args, 'temperature', 1.0)
        seq_lengths = batch['seq_lengths']
        batch_size = batch['num_samples']
        max_seq_len = batch['completion_mask'].shape[1]

        model_inputs = {k: v for k, v in batch.items() if k not in GRPO_NON_MODEL_KEYS}
        logps_packed, _ = megatron.compute_per_token_logps(model, iter([model_inputs]), temperature=temperature)

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

        if self.rollout is not None and self.rollout.is_primary:
            self.rollout.update_weights(
                weight_iter,
                vllm_lora_param_names=lora_names,
                peft_config=peft_config,
                base_sync_done=adapter_only,
            )
            self.rollout.reset_prefix_cache()
        else:
            for _ in weight_iter:
                pass

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
        from swift.megatron.trainers.utils import offload_megatron_model_to_cpu, offload_megatron_optimizer
        offload_megatron_model_to_cpu(self._megatron.wrapped_models)
        if getattr(self._megatron, 'optimizer', None) and self._args.offload_optimizer:
            offload_megatron_optimizer(self._megatron.optimizer)
        gc_collect()

    @dispatch_collect(dispatch='broadcast', collect='first')
    def reload_to_gpu(self):
        from swift.megatron.trainers.utils import load_megatron_model_to_gpu, load_megatron_optimizer
        load_megatron_model_to_gpu(self._megatron.wrapped_models)
        if getattr(self._megatron, 'optimizer', None) and self._args.offload_optimizer:
            load_megatron_optimizer(self._megatron.optimizer)

    def _batch_to_iterator(self, batch):
        micro_batches = split_micro_batches(batch, getattr(self._args, 'micro_batch_size', 1))
        micro_batches = [to_device(mb, get_current_device()) for mb in micro_batches]
        return iter(micro_batches), micro_batches

    def _split_sample_chunks(self, samples: Sequence[Dict[str, Any]]) -> List[Sequence[Dict[str, Any]]]:
        micro_batch_size = max(int(getattr(self._args, 'micro_batch_size', 1) or 1), 1)
        return [samples[i:i + micro_batch_size] for i in range(0, len(samples), micro_batch_size)]

    def _build_local_micro_batches(self, samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._collate_local_grpo_samples(chunk) for chunk in self._split_sample_chunks(samples)]

    def _collate_local_grpo_samples(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        from swift.megatron.utils import get_padding_to
        from swift.rlhf_trainers.utils import build_completion_mask_and_seq_lengths, build_rollout_logps

        template = self._pipeline.template
        device = get_current_device()
        encoded_list = [sample['encoded'] for sample in samples]
        encoded_batch = template.data_collator(encoded_list, padding_to=get_padding_to(self._args))
        encoded_batch = to_device(encoded_batch, device)

        labels = encoded_batch['labels']
        batch_size = len(samples)
        completion_mask, seq_lengths, _ = build_completion_mask_and_seq_lengths(
            labels,
            batch_size,
            padding_free=template.padding_free,
            encoded_batch=encoded_batch,
            device=device,
        )
        rollout_logps = build_rollout_logps(samples, completion_mask, device)
        encoded_batch.update({
            'completion_mask':
            completion_mask,
            'truncated_mask':
            torch.tensor([bool(s.get('is_truncated', False)) for s in samples], dtype=torch.bool, device=device),
            'num_samples':
            batch_size,
            'seq_lengths':
            seq_lengths,
            'rollout_per_token_logps':
            rollout_logps,
        })

        if samples and samples[0].get('old_per_token_logps') is not None:
            encoded_batch['old_per_token_logps'] = torch.stack([s['old_per_token_logps'].to(device) for s in samples],
                                                               dim=0)
        if samples and samples[0].get('ref_per_token_logps') is not None:
            encoded_batch['ref_per_token_logps'] = torch.stack([s['ref_per_token_logps'].to(device) for s in samples],
                                                               dim=0)
        if samples and samples[0].get('advantage') is not None:
            encoded_batch['advantages'] = torch.tensor([float(s.get('advantage', 0.0)) for s in samples],
                                                       dtype=torch.float32,
                                                       device=device)
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
