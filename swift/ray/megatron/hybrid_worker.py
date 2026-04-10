# Copyright (c) ModelScope Contributors. All rights reserved.
"""HybridWorker — co-located Ray actor holding multiple model roles.

In co-located (hybrid) mode, a single Ray actor process hosts multiple
roles on the same GPU(s) to avoid cross-process serialization overhead.
The ``role`` string determines which sub-models are initialized:

    - ``"model"``:    Megatron training model
    - ``"rollout"``:  vLLM inference engine (via RolloutReplica HYBRID)
    - ``"ref"``:      Reference model (for DPO/KTO)
    - ``"teacher"``:  Teacher model (for GKD)

Common role combinations:
    - ``"model_rollout"``:       GRPO (train + rollout on same GPUs)
    - ``"model_rollout_ref"``:   DPO with ref on same GPUs
    - ``"model_ref"``:           DPO co-located ref, separated rollout

Architecture (rollout path):
    HybridWorker → RolloutReplica(HYBRID) → VllmServer → AsyncLLM
                                                        → WeightSyncWorkerExtension

In separated mode, each role runs as an independent worker
(MegatronWorker or VllmWorker) in its own Ray actor process.
"""
import os
import torch
from typing import Any, Dict, List, Optional

from swift.utils import get_logger
from .inference_mixin import create_inference_model, inference_forward, slice_batch
from .worker_group import dispatch_collect

logger = get_logger()


class HybridWorker:

    VALID_ROLES = {'model', 'rollout', 'ref', 'teacher'}

    def init_model(
        self,
        argv: List[str],
        role: str = 'model',
        pipeline_cls_path: Optional[str] = None,
        trainer_cls_path: Optional[str] = None,
    ):
        """Initialize sub-models based on role string.

        Args:
            argv: CLI-style arguments for model configuration.
            role: Underscore-joined role names, e.g. "model_rollout_ref".
            pipeline_cls_path: Custom pipeline class path.
            trainer_cls_path: Custom trainer class path.
        """
        self._role = role
        self._roles = set(role.split('_'))
        unknown = self._roles - self.VALID_ROLES
        if unknown:
            raise ValueError(f'Unknown roles: {unknown}. Valid: {self.VALID_ROLES}')

        self.model = None
        self.rollout = None
        self.ref = None
        self.teacher = None

        self._pipeline = self._create_pipeline(argv, pipeline_cls_path)
        self._trainer_cls_path = trainer_cls_path
        self._argv = argv

        self._dp_rank = 0
        self._dp_size = 1
        self._is_collector = True

        if 'model' in self._roles:
            self._init_model(trainer_cls_path)

        if 'ref' in self._roles:
            self._init_ref()

        if 'teacher' in self._roles:
            self._init_teacher()

        if 'model' in self._roles:
            self._register_parallel_info()

        if 'rollout' in self._roles:
            self._init_rollout()

    def _create_pipeline(self, argv, pipeline_cls_path=None):
        if pipeline_cls_path:
            import importlib
            mod, cls = pipeline_cls_path.rsplit('.', 1)
            pipeline_cls = getattr(importlib.import_module(mod), cls)
        else:
            from swift.megatron.pipelines.train.rlhf import MegatronRLHF
            pipeline_cls = MegatronRLHF
        return pipeline_cls(argv)

    # ------------------------------------------------------------------
    # Sub-model initialization
    # ------------------------------------------------------------------

    def _init_model(self, trainer_cls_path=None):
        """Initialize the Megatron training model."""
        p = self._pipeline
        args = p.args
        if args.train_iters is None:
            logger.warning('train_iters not set in argv — using placeholder 1. '
                           'LR scheduler may be misconfigured.')
            args.train_iters = 1

        if trainer_cls_path:
            import importlib
            mod, cls = trainer_cls_path.rsplit('.', 1)
            trainer_cls = getattr(importlib.import_module(mod), cls)
            self.trainer = trainer_cls(args, p.template)
        else:
            self.trainer = p.prepare_trainer()

        self.model = self.trainer

    def _init_ref(self):
        """Initialize the reference model (inference-only Megatron)."""
        self.ref = create_inference_model(self._pipeline)

    def _init_teacher(self):
        """Initialize the teacher model (inference-only Megatron)."""
        model_id_override = getattr(self._pipeline.args, 'teacher_model_id', None)
        self.teacher = create_inference_model(self._pipeline, model_id_override=model_id_override)

    def _init_rollout(self):
        """Initialize the co-located vLLM engine via RolloutReplica(HYBRID)."""
        from .rollout.replica import RolloutMode, RolloutReplica

        args = self._pipeline.args
        tp_size = getattr(args, 'vllm_tensor_parallel_size', None)
        if tp_size is None:
            visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            tp_size = len([d for d in visible.split(',') if d.strip()]) if visible else 1

        self.rollout = RolloutReplica(mode=RolloutMode.HYBRID)
        self.rollout.init_engine(
            model_id=args.model_info.model_dir,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=getattr(args, 'vllm_gpu_memory_utilization', 0.9),
            max_num_seqs=getattr(args, 'vllm_max_num_seqs', 256),
            enforce_eager=getattr(args, 'vllm_enforce_eager', False),
            max_model_len=getattr(args, 'vllm_max_model_len', None),
            enable_prefix_caching=getattr(args, 'vllm_enable_prefix_caching', False),
            enable_sleep_mode=True,
            dtype=str(args.torch_dtype).split('.')[-1] if args.torch_dtype else 'auto',
        )

        sleep_level = getattr(args, 'sleep_level', 0)
        if sleep_level > 0:
            self.rollout.sleep(sleep_level)

        self._vllm_sampling_params = {
            'max_tokens': getattr(args, 'max_completion_length', 512),
            'temperature': getattr(args, 'temperature', 1.0),
            'top_p': getattr(args, 'top_p', 1.0),
            'top_k': getattr(args, 'top_k', -1),
            'n': 1,
        }
        logger.info('HybridWorker[role=%s] rollout initialized (tp=%d)', self._role, tp_size)

    # ------------------------------------------------------------------
    # Training methods (delegated to self.model / self.trainer)
    # ------------------------------------------------------------------

    @dispatch_collect(dispatch='broadcast', collect='first')
    def setup(self, iter_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Configure DDP, inject iter params from driver."""
        trainer = self.trainer
        if iter_params:
            args = trainer.args
            for k in ('train_iters', 'eval_iters', 'save_steps', 'eval_steps'):
                if k in iter_params and iter_params[k] is not None:
                    setattr(args, k, iter_params[k])
        trainer.setup_model_training()
        return {
            'train_iters': trainer.args.train_iters,
            'iteration': trainer.state.iteration,
        }

    @dispatch_collect(dispatch='dp_split', collect='all')
    def train_step(self, batch) -> Dict[str, Any]:
        data_iterator = self._batch_to_iterator(batch)
        self.trainer.run_train_step(data_iterator, None)
        return {'iteration': self.trainer.state.iteration}

    @dispatch_collect(dispatch='broadcast', collect='all')
    def finalize(self) -> Dict[str, Any]:
        from swift.utils import is_last_rank
        trainer = self.trainer
        trainer.finalize_training()
        self._pipeline._handle_trainer_state(trainer, is_last_rank())
        s = trainer.state
        return {
            'last_model_checkpoint': s.last_model_checkpoint,
            'best_model_checkpoint': s.best_model_checkpoint,
            'best_metric': s.best_metric,
        }

    def _batch_to_iterator(self, batch):
        from swift.utils import to_device

        if isinstance(batch, list):
            return iter([to_device(mb, 'cuda') for mb in batch])

        batch = to_device(batch, 'cuda')
        n_micro = self.trainer.args.num_microbatches
        if n_micro <= 1:
            return iter([batch])

        mbs = self.trainer.args.micro_batch_size
        return iter([slice_batch(batch, i * mbs, (i + 1) * mbs) for i in range(n_micro)])

    # ------------------------------------------------------------------
    # Generation (co-located vLLM via RolloutReplica)
    # ------------------------------------------------------------------

    @dispatch_collect(dispatch='broadcast', collect='first')
    def generate(self, batch: List[Dict], request_config: Optional[Dict] = None):
        """Generate completions using co-located vLLM engine.

        Handles offload/wake/sleep cycle for GPU memory management.
        Uses set_expandable_segments + aggressive_empty_cache for proper
        GPU memory lifecycle (matching verl's rollout_mode / trainer_mode).
        """
        from swift.rlhf_trainers.utils import aggressive_empty_cache, set_expandable_segments
        from .megatron_worker_utils import offload_megatron_for_vllm, reload_megatron_after_vllm

        replica = self.rollout
        args = self._pipeline.args
        enable_offload = (getattr(args, 'offload_model', False) or getattr(args, 'offload_optimizer', False))

        aggressive_empty_cache(force_sync=True)
        set_expandable_segments(False)

        if enable_offload and self.model is not None:
            offload_megatron_for_vllm(self)

        try:
            replica.wake_up(tags=['weights'])

            self._do_sync_weights_to_rollout()

            replica.wake_up(tags=['kv_cache'])

            sampling_params = dict(self._vllm_sampling_params)
            if request_config:
                sampling_params.update(request_config)

            encoded_batch = self._encode_batch(batch)
            outputs = replica.generate(encoded_batch, sampling_params)

            sleep_level = getattr(args, 'sleep_level', 0)
            if sleep_level > 0:
                replica.reset_prefix_cache()
                replica.sleep(level=sleep_level)

        finally:
            if enable_offload and self.model is not None:
                reload_megatron_after_vllm(self)
            aggressive_empty_cache(force_sync=True)
            set_expandable_segments(True)

        return outputs if self._is_collector else None

    def _encode_batch(self, batch: List[Dict]) -> List[Dict]:
        """Ensure each item in batch has 'input_ids'."""
        result = []
        for item in batch:
            if 'input_ids' in item:
                result.append(item)
                continue
            from swift.llm import EncodePreprocessor
            encoded = EncodePreprocessor(self._pipeline.template)([item])[0]
            result.append(encoded)
        return result

    # ------------------------------------------------------------------
    # Weight synchronization (co-located, via rollout package)
    # ------------------------------------------------------------------

    @dispatch_collect(dispatch='broadcast', collect='first')
    def sync_weights_to_rollout(self):
        """Sync Megatron training weights to co-located vLLM engine."""
        self._do_sync_weights_to_rollout()
        return {'status': 'ok'}

    def _do_sync_weights_to_rollout(self):
        if self.rollout is None or self.model is None:
            return

        from .rollout.weight_sync import sync_megatron_weights_to_vllm

        trainer = self.trainer
        target_device = 'cpu' if getattr(trainer.args, 'offload_bridge', False) else None

        sync_megatron_weights_to_vllm(
            bridge=trainer.bridge,
            models=trainer.unwrapped_models,
            replica=self.rollout,
            target_device=target_device,
            use_ipc=True,
            wake_up_first=False,
            reset_cache=True,
        )

    @dispatch_collect(dispatch='broadcast', collect='first')
    def export_weights(self) -> list:
        """Export model weights as (name, cpu_tensor) pairs for separated sync."""
        from .rollout.weight_sync import export_weights_for_transfer
        return export_weights_for_transfer(self.trainer.bridge, self.trainer.unwrapped_models)

    # ------------------------------------------------------------------
    # Reference model forward (co-located)
    # ------------------------------------------------------------------

    @dispatch_collect(dispatch='dp_split', collect='dp')
    def ref_forward(self, raw_batch):
        """Forward pass on the co-located reference model."""
        if self.ref is None:
            return None

        if isinstance(raw_batch, list):
            parts = [inference_forward(self._pipeline.args, self.ref, mb) for mb in raw_batch]
            if any(p is None for p in parts):
                return None
            return {'ref_logps': torch.cat([p['logps'] for p in parts], dim=0)}
        result = inference_forward(self._pipeline.args, self.ref, raw_batch)
        if result is None:
            return None
        return {'ref_logps': result['logps']}

    # ------------------------------------------------------------------
    # Teacher model forward (co-located)
    # ------------------------------------------------------------------

    @dispatch_collect(dispatch='dp_split', collect='dp')
    def teacher_forward(self, raw_batch):
        """Forward pass on the co-located teacher model."""
        if self.teacher is None:
            return None

        if isinstance(raw_batch, list):
            parts = [inference_forward(self._pipeline.args, self.teacher, mb) for mb in raw_batch]
            if any(p is None for p in parts):
                return None
            return {'teacher_logps': torch.cat([p['logps'] for p in parts], dim=0)}
        result = inference_forward(self._pipeline.args, self.teacher, raw_batch)
        if result is None:
            return None
        return {'teacher_logps': result['logps']}

    # ------------------------------------------------------------------
    # Parallel info & utilities
    # ------------------------------------------------------------------

    def _register_parallel_info(self):
        from megatron.core import mpu
        self._dp_rank = mpu.get_data_parallel_rank()
        self._dp_size = mpu.get_data_parallel_world_size()
        self._is_collector = (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            and mpu.get_context_parallel_rank() == 0)

    def get_parallel_info(self) -> Dict[str, Any]:
        return {
            'dp_rank': self._dp_rank,
            'dp_size': self._dp_size,
            'is_collector': self._is_collector,
        }

    def ping(self) -> str:
        parts = []
        if self.model is not None:
            parts.append('model')
        if self.rollout is not None:
            parts.append('rollout')
        if self.ref is not None:
            parts.append('ref')
        if self.teacher is not None:
            parts.append('teacher')
        mode = '+'.join(parts) if parts else 'empty'
        return '%s_rank%s' % (mode, os.environ.get('RANK', '?'))

    def call_trainer(self, method_name: str, *args, **kwargs) -> Any:
        return getattr(self.trainer, method_name)(*args, **kwargs)
