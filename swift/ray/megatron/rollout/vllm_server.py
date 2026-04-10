# Copyright (c) ModelScope Contributors. All rights reserved.
"""VllmServer — Ray Actor hosting a vLLM AsyncLLM engine.

Corresponds to verl's ``vLLMHttpServer``.  The key difference is that
we skip the HTTP serving layer and expose RPC methods directly via
Ray Actor calls, which is simpler for within-cluster communication.

The server manages:
- AsyncLLM engine lifecycle (create / shutdown)
- Sampling via ``generate()`` with concurrent batch submission
- Sleep / wake_up for GPU memory management in co-located mode
- Weight synchronization via ``update_weights()`` which delegates to
  SwiftWorkerExtension in vLLM subprocesses through ``collective_rpc``

For weight sync it reuses:
- ``swift.rlhf_trainers.utils.FlattenedTensorBucket`` for bucket packing
- ``swift.pipelines.infer.rollout.WeightSyncWorkerExtension`` concepts
"""
import asyncio
import gc
import inspect
import os
import threading
import time
import torch
import uuid
from typing import Any, Dict, List, Optional, Tuple

from swift.utils import get_logger

logger = get_logger()


class BucketedWeightSender:
    """Send model weights via bucketed IPC transfer over ZMQ.

    Packs weight tensors into a fixed-size buffer and sends them in
    buckets to the receiver. Supports CUDA IPC and shared memory fallback.
    """

    def __init__(self, zmq_handle: str, bucket_size_mb: int = 512, use_shm: bool = False):
        self.zmq_handle = zmq_handle
        self.bucket_size = int(bucket_size_mb) << 20
        self.use_shm = use_shm
        self.socket = None
        self.buffer = None
        self.shm = None

    def init_socket(self):
        import zmq
        ctx = zmq.Context.instance()
        self.socket = ctx.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 300_000)
        self.socket.setsockopt(zmq.SNDTIMEO, 300_000)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(self.zmq_handle)

    def init_buffer(self):
        from torch.multiprocessing.reductions import reduce_tensor

        if not self.use_shm:
            self.buffer = torch.empty(self.bucket_size, dtype=torch.uint8, device='cuda')
            handle = reduce_tensor(self.buffer)
            self.socket.send_pyobj(handle)
        else:
            from multiprocessing import shared_memory
            shm_name = f'swift_weights_{uuid.uuid4().hex}'
            self.shm = shared_memory.SharedMemory(name=shm_name, create=True, size=self.bucket_size)
            self.buffer = torch.frombuffer(self.shm.buf, dtype=torch.uint8)
            self.socket.send_pyobj({'name': shm_name, 'size': self.bucket_size})
        self.socket.recv()

    async def async_send_weights(self, weights):
        """Send weight tensors via bucketed IPC. Accepts sync or async iterators."""
        loop = asyncio.get_running_loop()

        offset = 0
        bucket_meta = {}
        n_weights = 0

        def _zmq_send_recv(payload):
            self.socket.send_pyobj(payload)
            return self.socket.recv()

        async def _flush(is_last):
            nonlocal offset, bucket_meta
            if not bucket_meta and not is_last:
                return
            if self.buffer.device.type != 'cpu':
                torch.cuda.synchronize()
            await loop.run_in_executor(None, _zmq_send_recv, {'bucket_meta': bucket_meta, 'is_last': is_last})
            offset = 0
            bucket_meta = {}

        if isinstance(weights, dict):
            items = weights.items()
        elif hasattr(weights, '__iter__'):
            items = weights
        else:
            items = weights

        for name, weight in items:
            if self.use_shm and weight.device.type != 'cpu':
                weight = weight.cpu()
            if not weight.is_contiguous():
                weight = weight.contiguous()

            nbytes = weight.nbytes
            if offset + nbytes > self.bucket_size:
                await _flush(False)
            assert offset + nbytes <= self.bucket_size, (
                f'Weight {name}({weight.shape}, {weight.dtype}) too large for bucket. '
                f'Increase bucket_size_mb.')

            bucket_meta[name] = {
                'name': name,
                'shape': weight.shape,
                'dtype': weight.dtype,
                'offset': offset,
            }
            self.buffer[offset:offset + nbytes].copy_(weight.view(-1).view(torch.uint8), non_blocking=True)
            offset += nbytes
            n_weights += 1

        await _flush(True)
        logger.info('BucketedWeightSender: sent %d weights', n_weights)

    def cleanup(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.zmq_handle.startswith('ipc://'):
            ipc_path = self.zmq_handle[len('ipc://'):]
            try:
                if os.path.exists(ipc_path):
                    os.remove(ipc_path)
            except OSError:
                pass
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            self.shm = None
        gc.collect()


class VllmServer:
    """Ray Actor that wraps a vLLM v1 AsyncLLM engine.

    Not decorated with @ray.remote here — the caller (RolloutReplica or
    VllmWorker) wraps it with ``ray.remote()``.

    Lifecycle::

        __init__() → init_engine() → [generate() / update_weights() /
                                        sleep() / wake_up()]* → shutdown()
    """

    def __init__(self):
        self.engine = None
        self._async_loop = None
        self._async_thread = None
        self._model_id = None
        self._tp_size = 1
        self._enable_sleep_mode = False

    def init_engine(
        self,
        model_id: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        max_num_seqs: int = 256,
        enable_sleep_mode: bool = False,
        enable_prefix_caching: bool = False,
        enforce_eager: bool = False,
        trust_remote_code: bool = True,
        dtype: str = 'auto',
        load_format: str = 'auto',
        worker_extension_cls: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Initialize the AsyncLLM engine in a dedicated background event loop."""
        os.environ['VLLM_USE_V1'] = '1'
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '86400'

        self._model_id = model_id
        self._tp_size = tensor_parallel_size
        self._enable_sleep_mode = enable_sleep_mode

        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(target=self._run_event_loop, daemon=True, name='VllmServer-EventLoop')
        self._async_thread.start()

        engine_config = {
            'model': model_id,
            'tensor_parallel_size': tensor_parallel_size,
            'gpu_memory_utilization': gpu_memory_utilization,
            'max_num_seqs': max_num_seqs,
            'trust_remote_code': trust_remote_code,
            'enforce_eager': enforce_eager,
            'dtype': dtype,
            'load_format': load_format,
            'disable_log_stats': True,
        }

        if tensor_parallel_size > 1:
            engine_config['distributed_executor_backend'] = 'mp'
        if max_model_len is not None:
            engine_config['max_model_len'] = max_model_len
        if enable_prefix_caching:
            engine_config['enable_prefix_caching'] = True
        if enable_sleep_mode:
            engine_config['enable_sleep_mode'] = True

        if worker_extension_cls is None:
            worker_extension_cls = ('swift.pipelines.infer.rollout.WeightSyncWorkerExtension')
        engine_config['worker_extension_cls'] = worker_extension_cls

        engine_config.update(kwargs)

        async def _create():
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.usage.usage_lib import UsageContext
            from vllm.v1.engine.async_llm import AsyncLLM

            valid_args = inspect.signature(AsyncEngineArgs).parameters.keys()
            filtered = {k: v for k, v in engine_config.items() if k in valid_args}
            invalid = set(engine_config.keys()) - set(valid_args)
            if invalid:
                logger.warning('VllmServer: filtered invalid args: %s', invalid)

            engine_args = AsyncEngineArgs(**filtered)
            vllm_config = engine_args.create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER)
            return AsyncLLM.from_vllm_config(
                vllm_config=vllm_config,
                usage_context=UsageContext.OPENAI_API_SERVER,
            )

        self.engine = self._run_in_loop(_create())
        logger.info('VllmServer: engine initialized (model=%s, tp=%d, '
                    'load_format=%s)', model_id, tensor_parallel_size, load_format)
        return {'model_id': model_id, 'tp_size': tensor_parallel_size}

    # ------------------------------------------------------------------
    # Event loop management
    # ------------------------------------------------------------------

    def _run_event_loop(self):
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_forever()

    def _run_in_loop(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._async_loop)
        return future.result()

    # ------------------------------------------------------------------
    # Sampling — concurrent batch submission
    # ------------------------------------------------------------------

    def generate(
        self,
        batch: List[Dict],
        sampling_params: Optional[Dict] = None,
    ) -> List[Dict]:
        """Generate completions for a batch of prompts.

        Submits all requests concurrently to leverage vLLM's
        continuous batching, rather than processing sequentially.
        """
        if sampling_params is None:
            sampling_params = {}

        async def _gen():
            from vllm import SamplingParams as VllmSP
            from vllm.inputs import TokensPrompt

            vllm_params = VllmSP(**sampling_params)

            async def _gen_single(item):
                input_ids = item.get('input_ids', [])
                if hasattr(input_ids, 'tolist'):
                    input_ids = input_ids.tolist()

                prompt = TokensPrompt(prompt_token_ids=input_ids)
                mm_data = item.get('multi_modal_data')
                if mm_data:
                    prompt['multi_modal_data'] = mm_data

                request_id = uuid.uuid4().hex
                result = None
                async for output in self.engine.generate(
                        prompt=prompt,
                        sampling_params=vllm_params,
                        request_id=request_id,
                ):
                    result = output

                if result is None:
                    return {'token_ids': [], 'logprobs': None, 'finish_reason': 'error'}

                out = result.outputs[0]
                token_ids = list(out.token_ids)
                log_probs = None
                if out.logprobs is not None:
                    log_probs = []
                    for i, lp_dict in enumerate(out.logprobs):
                        if i < len(token_ids) and token_ids[i] in lp_dict:
                            log_probs.append(lp_dict[token_ids[i]].logprob)
                        else:
                            log_probs.append(float('-inf'))

                return {
                    'token_ids': token_ids,
                    'logprobs': log_probs,
                    'finish_reason': out.finish_reason,
                }

            tasks = [_gen_single(item) for item in batch]
            return await asyncio.gather(*tasks)

        return list(self._run_in_loop(_gen()))

    # ------------------------------------------------------------------
    # Sleep / Wake up
    # ------------------------------------------------------------------

    def sleep(self, level: int = 2) -> None:
        if not self._enable_sleep_mode:
            return
        self._run_in_loop(self.engine.sleep(level=level))
        logger.debug('VllmServer: sleeping at level %d', level)

    def wake_up(self, tags: Optional[List[str]] = None) -> None:
        if not self._enable_sleep_mode:
            return
        if tags is None:
            tags = ['weights', 'kv_cache']
        self._run_in_loop(self.engine.wake_up(tags=tags))
        logger.debug('VllmServer: woke up with tags %s', tags)

    def reset_prefix_cache(self) -> None:
        self._run_in_loop(self.engine.reset_prefix_cache())

    # ------------------------------------------------------------------
    # Weight synchronization
    # ------------------------------------------------------------------

    def update_weights_direct(
        self,
        weight_list: List[Tuple[str, torch.Tensor]],
    ) -> Dict[str, str]:
        """Load weights via FlattenedTensorBucket + collective_rpc broadcast.

        Uses the same protocol as ``WeightSyncWorkerExtension.update_flattened_params``:
        flatten tensors into a bucket, send metadata via collective_rpc,
        then broadcast the flattened tensor to all TP workers.
        """
        from swift.rlhf_trainers.utils import FlattenedTensorBucket, patch_vllm_moe_model_weight_loader

        async def _update():
            bucket_size_mb = int(os.environ.get('SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE', '512'))
            bucket_size_bytes = bucket_size_mb * 1024 * 1024

            current_bucket = []
            current_size = 0

            for name, param in weight_list:
                param_size = param.numel() * param.element_size()
                current_bucket.append((name, param))
                current_size += param_size

                if current_size > bucket_size_bytes:
                    bucket = FlattenedTensorBucket(named_tensors=current_bucket)
                    metadatas = [vars(m) for m in bucket.get_metadata()]
                    await self.engine.collective_rpc(
                        'update_flattened_params',
                        kwargs={'metadatas': metadatas},
                    )
                    current_bucket = []
                    current_size = 0

            if current_bucket:
                bucket = FlattenedTensorBucket(named_tensors=current_bucket)
                metadatas = [vars(m) for m in bucket.get_metadata()]
                await self.engine.collective_rpc(
                    'update_flattened_params',
                    kwargs={'metadatas': metadatas},
                )

        self._run_in_loop(_update())
        logger.info('VllmServer: weights updated via direct path')
        return {'status': 'ok'}

    def update_weights_ipc(
        self,
        weight_iter,
        bucket_size_mb: int = 2048,
    ) -> Dict[str, str]:
        """Update weights via ZMQ + CUDA IPC using BucketedWeightSender.

        Delegates IPC protocol to BucketedWeightSender for cleaner separation.
        The worker extension must support ``update_weights_from_ipc``.
        """
        use_shm = not torch.cuda.is_available()

        async def _update():
            start_time = time.time()

            device_uuid = self._get_device_uuid()
            sync_id = uuid.uuid4().hex
            zmq_handle = (f'ipc:///tmp/swift-vllm-ipc-{device_uuid}'
                          f'-{os.getpid()}-{sync_id}.sock')

            sender = BucketedWeightSender(
                zmq_handle=zmq_handle,
                bucket_size_mb=bucket_size_mb,
                use_shm=use_shm,
            )

            try:
                sender.init_socket()

                worker_task = asyncio.ensure_future(
                    self.engine.collective_rpc(
                        'update_weights_from_ipc',
                        kwargs={
                            'use_shm': use_shm,
                            'zmq_handle': zmq_handle,
                        },
                    ))

                sender.init_buffer()
                await sender.async_send_weights(weight_iter)
                await worker_task
            finally:
                sender.cleanup()

            elapsed = time.time() - start_time
            logger.info('VllmServer: updated weights via %s in %.2fs', 'SHM' if use_shm else 'IPC', elapsed)

        self._run_in_loop(_update())
        return {'status': 'ok'}

    def _get_device_uuid(self) -> str:
        try:
            from vllm.platforms import current_platform
            return current_platform.get_device_uuid(0)
        except Exception:
            return f'gpu-{os.getpid()}'

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if self.engine is not None:
            try:
                self._run_in_loop(self.engine.shutdown())
            except Exception as e:
                logger.warning('VllmServer shutdown error: %s', e)
            self.engine = None
        if self._async_loop is not None:
            try:
                self._async_loop.call_soon_threadsafe(self._async_loop.stop)
                if self._async_thread is not None:
                    self._async_thread.join(timeout=5)
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ping(self) -> str:
        return f'vllm_server(model={self._model_id}, tp={self._tp_size})'
