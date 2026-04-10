# Copyright (c) ModelScope Contributors. All rights reserved.
"""MegatronWorker — single-GPU Ray actor wrapping a Megatron model.

For training or inference only (no vLLM). Used in separated mode where
each role runs as an independent worker process.

Training:   init_model(trainable=True) → setup() → train_step(batch)* → finalize()
Inference:  init_model(trainable=False) → forward(batch)*

For co-located multi-role setups, use HybridWorker instead.
"""
import os
import torch
from typing import Any, Dict, List, Optional

from swift.utils import get_logger
from .inference_mixin import create_inference_model, inference_forward, slice_batch
from .worker_group import dispatch_collect

logger = get_logger()


class MegatronWorker:

    def init_model(
        self,
        argv: List[str],
        trainable: bool = True,
        pipeline_cls_path: Optional[str] = None,
        trainer_cls_path: Optional[str] = None,
    ):
        self.trainable = trainable

        if pipeline_cls_path:
            import importlib
            mod, cls = pipeline_cls_path.rsplit('.', 1)
            pipeline_cls = getattr(importlib.import_module(mod), cls)
        else:
            from swift.megatron.pipelines.train.rlhf import MegatronRLHF
            pipeline_cls = MegatronRLHF

        self._pipeline = pipeline_cls(argv)
        self._trainer_cls_path = trainer_cls_path

        if trainable:
            self._init_trainable()
        else:
            self._init_inference()

        self._register_parallel_info()

    def _init_trainable(self):
        p = self._pipeline
        args = p.args
        if args.train_iters is None:
            logger.warning('train_iters not set in argv — using placeholder 1. '
                           'LR scheduler may be misconfigured.')
            args.train_iters = 1

        if self._trainer_cls_path:
            import importlib
            mod, cls = self._trainer_cls_path.rsplit('.', 1)
            trainer_cls = getattr(importlib.import_module(mod), cls)
            self.trainer = trainer_cls(args, p.template)
        else:
            self.trainer = p.prepare_trainer()

    def _init_inference(self):
        """Create model only — no optimizer, no dataset, no DDP.

        Uses ``create_inference_model`` which calls ``MegatronRLHF(argv)``
        internally. The pipeline's ``__init__`` handles Megatron parallel
        initialization via ``_init_megatron_args``.
        """
        self._infer_models = create_inference_model(self._pipeline)
        self._args = self._pipeline.args
        self._template = self._pipeline.template

    @dispatch_collect(dispatch='broadcast', collect='first')
    def setup(self, iter_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Configure DDP, inject iter params from driver. Returns loop metadata."""
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
        """One training step with a batch dispatched from the driver."""
        data_iterator = self._batch_to_iterator(batch)
        self.trainer.run_train_step(data_iterator, None)
        return {'iteration': self.trainer.state.iteration}

    def _batch_to_iterator(self, batch):
        """Convert dispatched batch into micro-batch iterator."""
        from swift.utils import to_device

        if isinstance(batch, list):
            return iter([to_device(mb, 'cuda') for mb in batch])

        batch = to_device(batch, 'cuda')
        n_micro = self.trainer.args.num_microbatches
        if n_micro <= 1:
            return iter([batch])

        mbs = self.trainer.args.micro_batch_size
        return iter([slice_batch(batch, i * mbs, (i + 1) * mbs) for i in range(n_micro)])

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

    @dispatch_collect(dispatch='dp_split', collect='dp')
    def forward(self, raw_batch):
        """PP-aware forward for inference workers."""
        if isinstance(raw_batch, list):
            parts = [inference_forward(self._args, self._infer_models, mb) for mb in raw_batch]
            if any(p is None for p in parts):
                return None
            return {'ref_logps': torch.cat([p['logps'] for p in parts], dim=0)}
        result = inference_forward(self._args, self._infer_models, raw_batch)
        if result is None:
            return None
        return {'ref_logps': result['logps']}

    # ------------------------------------------------------------------
    # Weight export (for separated mode sync via driver)
    # ------------------------------------------------------------------

    @dispatch_collect(dispatch='broadcast', collect='first')
    def export_weights(self) -> List:
        """Export model weights as a list of (name, cpu_tensor) pairs."""
        bridge = self.trainer.bridge
        models = self.trainer.unwrapped_models
        return [(n, t.cpu().clone()) for n, t in bridge.export_weights(models, target_device='cpu')]

    # ------------------------------------------------------------------
    # Parallel info
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

    def call_trainer(self, method_name: str, *args, **kwargs) -> Any:
        """Proxy to call an arbitrary method on the internal trainer."""
        return getattr(self.trainer, method_name)(*args, **kwargs)

    def ping(self) -> str:
        mode = 'train' if self.trainable else 'infer'
        return '%s_rank%s' % (mode, os.environ.get('RANK', '?'))
