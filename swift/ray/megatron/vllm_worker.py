# Copyright (c) ModelScope Contributors. All rights reserved.
"""VllmWorker — Ray actor wrapping vLLM inference for rollout.

Architecture follows verl's client/server pattern:
  - A VllmServer (Ray Actor) hosts the actual AsyncLLM engine
  - VllmWorker acts as a lightweight client connecting to the server
  - The server lifecycle is managed externally by the pipeline

This avoids each VllmWorker starting its own vLLM engine, which
would waste resources when multiple rollout workers exist.

Lifecycle: init_model() → generate() → update_weights() → shutdown()
"""
import os
import ray
import threading
import torch
import torch.distributed as dist
from typing import Any, Dict, List, Optional

from swift.utils import get_logger
from .worker_group import dispatch_collect

logger = get_logger()

_PARSE_ARGS_LOCK = threading.Lock()


class VllmWorker:
    """vLLM inference worker for Ray — client mode.

    Connects to an external VllmServer Ray Actor rather than hosting
    its own engine. Not @ray.remote — decorated by WorkerGroup.
    """

    def init_model(
        self,
        argv: List[str],
        server_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize vLLM client.

        Args:
            argv: CLI-style arguments for model configuration.
            server_name: Named Ray Actor for the VllmServer. If None,
                creates and manages its own server.
        """
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()

        self._args, self._template = self._parse_args(argv)

        self._server_name = server_name
        self._server_handle = None
        self._owns_server = server_name is None

        if self._owns_server:
            self._start_server()
        else:
            self._server_handle = ray.get_actor(server_name)

        self._sampling_params = {
            'max_tokens': getattr(self._args, 'max_completion_length', 512),
            'temperature': getattr(self._args, 'temperature', 1.0),
            'top_p': getattr(self._args, 'top_p', 1.0),
            'top_k': getattr(self._args, 'top_k', -1),
            'n': 1,
        }
        logger.info('VllmWorker[rank=%d] ready (server=%s)', self._rank, server_name or 'self-managed')

    def _start_server(self):
        """Start a self-managed VllmServer for standalone use."""
        from .rollout.replica import RolloutMode, RolloutReplica

        self._replica = RolloutReplica(mode=RolloutMode.SEPARATED)

        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        num_visible = len([d for d in visible.split(',') if d.strip()]) if visible else 1
        tp_size = getattr(self._args, 'vllm_tensor_parallel_size', None) or num_visible

        args = self._args
        self._replica.init_engine(
            model_id=args.model_info.model_dir,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=getattr(args, 'vllm_gpu_memory_utilization', 0.9),
            max_num_seqs=getattr(args, 'vllm_max_num_seqs', 256),
            enforce_eager=getattr(args, 'vllm_enforce_eager', False),
            max_model_len=getattr(args, 'vllm_max_model_len', None),
            enable_prefix_caching=getattr(args, 'vllm_enable_prefix_caching', False),
            dtype=str(args.torch_dtype).split('.')[-1] if args.torch_dtype else 'auto',
        )

    def _parse_args(self, argv):
        """Lightweight arg parsing without Megatron initialization.

        Uses a global lock to protect monkey-patching of class-level
        methods from concurrent Ray actor initialization.
        """
        from swift.megatron.arguments.sft_args import MegatronSftArguments

        with _PARSE_ARGS_LOCK:
            saved_init = MegatronSftArguments._init_megatron_args
            orig_post = MegatronSftArguments.__post_init__

            def _patched_post_init(self_):
                if not self_.dataset and not self_.cached_dataset:
                    self_.dataset = ['__vllm_placeholder__']
                orig_post(self_)

            try:
                MegatronSftArguments._init_megatron_args = lambda self: None
                MegatronSftArguments.__post_init__ = _patched_post_init

                from swift.megatron.pipelines.train.rlhf import MegatronRLHF
                from swift.pipelines.base import SwiftPipeline
                from swift.utils import to_abspath

                class _VllmPipeline(MegatronRLHF):

                    def __init__(self_, _argv):
                        self_.train_msg = {}
                        SwiftPipeline.__init__(self_, _argv)
                        args = self_.args
                        if args.output_dir is None:
                            args.output_dir = f'megatron_output/{args.model_suffix}'
                        args.output_dir = to_abspath(args.output_dir)
                        os.makedirs(args.output_dir, exist_ok=True)
                        with torch.device('meta'):
                            self_.model, self_.processor = args.get_model_processor(
                                load_model=False, download_model=True)
                        self_._prepare_template()

                pipeline = _VllmPipeline(argv)
                return pipeline.args, pipeline.template
            finally:
                MegatronSftArguments._init_megatron_args = saved_init
                MegatronSftArguments.__post_init__ = orig_post

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @dispatch_collect(dispatch='broadcast', collect='first')
    def generate(self, batch: List[Dict], request_config: Optional[Dict] = None):
        """Generate completions. Only rank 0 returns results."""
        sampling_params = dict(self._sampling_params)
        if request_config:
            sampling_params.update(request_config)

        encoded_batch = self._encode_batch(batch)

        if self._owns_server:
            outputs = self._replica.generate(encoded_batch, sampling_params)
        else:
            outputs = ray.get(self._server_handle.generate.remote(encoded_batch, sampling_params))

        return outputs if self._rank == 0 else None

    def _encode_batch(self, batch: List[Dict]) -> List[Dict]:
        result = []
        for item in batch:
            if 'input_ids' in item:
                result.append(item)
                continue
            from swift.llm import EncodePreprocessor
            encoded = EncodePreprocessor(self._template)([item])[0]
            result.append(encoded)
        return result

    # ------------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------------

    @dispatch_collect(dispatch='broadcast', collect='first')
    def update_weights(self, weight_iter):
        """Update model weights."""
        if self._owns_server:
            self._replica.update_weights_from_iter(weight_iter, use_ipc=True)
        else:
            ray.get(self._server_handle.update_weights_ipc.remote(weight_iter))
        logger.info('VllmWorker[rank=%d] weights updated', self._rank)
        return {'status': 'ok'}

    # ------------------------------------------------------------------
    # Sleep / Wake up
    # ------------------------------------------------------------------

    @dispatch_collect(dispatch='broadcast', collect='first')
    def sleep(self, level: int = 2):
        if self._owns_server:
            self._replica.sleep(level)
        else:
            ray.get(self._server_handle.sleep.remote(level))
        return {'status': 'sleeping'}

    @dispatch_collect(dispatch='broadcast', collect='first')
    def wake_up(self, tags: Optional[List[str]] = None):
        if self._owns_server:
            self._replica.wake_up(tags)
        else:
            ray.get(self._server_handle.wake_up.remote(tags))
        return {'status': 'awake'}

    # ------------------------------------------------------------------
    # Parallel info
    # ------------------------------------------------------------------

    def get_parallel_info(self) -> Dict[str, Any]:
        return {
            'dp_rank': self._rank % self._world_size,
            'dp_size': self._world_size,
            'is_collector': self._rank == 0,
        }

    def ping(self) -> str:
        mode = 'client' if not self._owns_server else 'standalone'
        return 'vllm_%s_rank%d' % (mode, self._rank)

    def shutdown(self):
        if self._owns_server and hasattr(self, '_replica'):
            try:
                self._replica.shutdown()
            except Exception as e:
                logger.warning('VllmWorker shutdown error: %s', e)
