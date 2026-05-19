# Copyright (c) ModelScope Contributors. All rights reserved.
"""RayVllmEngine — vLLM engine wrapper for Ray rollout actors.

Reuses ``VllmEngine``'s template / processor initialisation but creates
the ``AsyncLLM`` engine inside a **persistent event-loop thread** so that
vLLM's ZMQ-based ``AsyncMPClient`` (which caches event-loop-bound tasks
and sockets) stays consistent across all async operations.

``VllmServer`` holds a ``RayVllmEngine`` instance and delegates all
engine operations to it.
"""
import asyncio
import os
import threading
import torch
from typing import Any, Dict, List, Optional

from swift.rlhf_trainers.utils import set_expandable_segments
from swift.utils import gc_collect
from swift.utils.logger import get_logger

logger = get_logger()


class RayVllmEngine:

    def __init__(
        self,
        model_id: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        max_num_seqs: int = 256,
        enable_sleep_mode: bool = False,
        enable_lora: bool = False,
        max_lora_rank: int = 8,
        enable_prefix_caching: bool = False,
        enforce_eager: bool = False,
        trust_remote_code: bool = True,
        dtype: str = 'auto',
        load_format: str = 'auto',
        template_kwargs: Optional[Dict[str, Any]] = None,
        **engine_kwargs,
    ):
        os.environ.setdefault('VLLM_USE_V1', '1')
        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
        os.environ.setdefault('VLLM_ENGINE_ITERATION_TIMEOUT_S', '86400')

        self.model_id = model_id
        self.tp_size = tensor_parallel_size
        self.enable_sleep_mode = enable_sleep_mode
        self.enable_lora = enable_lora

        distributed_executor_backend = 'mp' if tensor_parallel_size > 1 else None

        extra_engine_kwargs = dict(engine_kwargs)
        extra_engine_kwargs['worker_extension_cls'] = ('swift.pipelines.infer.rollout.WeightSyncWorkerExtension')

        from swift.model import get_processor
        from swift.template import get_template
        processor = get_processor(model_id_or_path=model_id, download_model=True)
        template = get_template(processor, **(template_kwargs or {}))
        self.template = template
        self.tokenizer = template.tokenizer

        # --- Start persistent event loop thread ---
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name='RayVllmEngine-EventLoop',
        )
        self._loop_dead = threading.Event()
        self._loop_exception: Optional[BaseException] = None
        self._loop_thread.start()

        # --- Create VllmEngine (with engine) inside the event loop ---
        self._vllm_engine = self._run_in_loop(
            self._create_engine_async(
                model_id,
                template=template,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                enable_sleep_mode=enable_sleep_mode,
                enable_lora=enable_lora,
                max_lora_rank=max_lora_rank,
                enable_prefix_caching=enable_prefix_caching,
                enforce_eager=enforce_eager,
                load_format=load_format,
                distributed_executor_backend=distributed_executor_backend,
                logprobs_mode='processed_logprobs',
                engine_kwargs=extra_engine_kwargs,
            ))
        self.engine = self._vllm_engine.engine

        logger.info(
            'RayVllmEngine: ready (model=%s, tp=%d, sleep=%s, load_format=%s)',
            model_id,
            tensor_parallel_size,
            enable_sleep_mode,
            load_format,
        )

    @staticmethod
    async def _create_engine_async(model_id: str, *, template, **kwargs):
        from swift.infer_engine import VllmEngine
        engine = VllmEngine(
            model_id,
            use_async_engine=True,
            template=template,
            **kwargs,
        )
        return engine

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        except Exception as exc:
            logger.exception('RayVllmEngine event loop died unexpectedly')
            self._loop_exception = exc
            self._loop_dead.set()

    def _run_in_loop(self, coro):
        if self._loop_dead.is_set():
            cause = self._loop_exception
            raise RuntimeError('RayVllmEngine event loop is no longer running. '
                               f'Original error: {type(cause).__name__}: {cause}'
                               if cause else 'RayVllmEngine event loop is no longer running') from cause
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    # ------------------------------------------------------------------
    # Sleep / Wake-up
    # ------------------------------------------------------------------

    def sleep(self, level: int = 2):
        if not self.enable_sleep_mode:
            return
        self._run_in_loop(self.engine.sleep(level=level))
        gc_collect()
        set_expandable_segments(True)
        logger.debug('RayVllmEngine: sleeping at level %d', level)

    def wake_up(self, tags: Optional[List[str]] = None):
        if not self.enable_sleep_mode:
            return
        if tags is None or 'kv_cache' in tags:
            gc_collect()
            set_expandable_segments(False)
        self._run_in_loop(self.engine.wake_up(tags=tags))
        logger.debug('RayVllmEngine: woke up with tags %s', tags)

    def generate_batch(
        self,
        infer_requests: List[Any],
        request_config: Optional[Any] = None,
    ) -> List[Any]:
        from swift.infer_engine.protocol import RequestConfig
        if request_config is None:
            request_config = RequestConfig()

        async def _gen():
            tasks = [self._vllm_engine.infer_async(req, request_config) for req in infer_requests]
            return await asyncio.gather(*tasks)

        return list(self._run_in_loop(_gen()))

    def get_model_param_names(self) -> List[str]:
        """Return parameter names from vLLM model via collective_rpc."""

        async def _get():
            result = await self.engine.collective_rpc('get_state_keys')
            if result and isinstance(result[0], list):
                return result[0]
            return []

        return self._run_in_loop(_get())

    def reset_prefix_cache(self):
        self._run_in_loop(self.engine.reset_prefix_cache())

    def update_weights_ipc(
        self,
        zmq_handle: str,
        use_shm: bool = False,
        timeout_s: int = 600,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
    ):
        """Trigger the vLLM worker extension's ``update_weights_from_ipc``."""

        async def _rpc():
            rpc_kwargs = {
                'use_shm': use_shm,
                'zmq_handle': zmq_handle,
                'timeout_s': timeout_s,
            }
            if peft_config is not None and base_sync_done:
                rpc_kwargs['peft_config'] = peft_config
                rpc_kwargs['base_sync_done'] = base_sync_done
            return await asyncio.wait_for(
                self.engine.collective_rpc('update_weights_from_ipc', kwargs=rpc_kwargs),
                timeout=timeout_s,
            )

        self._run_in_loop(_rpc())

    def shutdown(self):
        if self.engine is not None:
            try:
                self.engine.shutdown()
            except Exception as e:  # noqa: BLE001
                logger.warning('RayVllmEngine shutdown error: %s', e)
            self.engine = None

        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
                if self._loop_thread is not None:
                    self._loop_thread.join(timeout=5)
            except Exception:
                pass

        gc_collect()
