# Copyright (c) ModelScope Contributors. All rights reserved.
import datetime as dt
import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch.distributed as dist

from swift.rlhf_trainers import VllmArguments
from swift.utils import get_logger, init_process_group, is_dist, to_abspath
from .base_args import BaseArguments
from .merge_args import MergeArguments

logger = get_logger()


@dataclass
class LmdeployArguments:
    """Holds the configuration arguments for lmdeploy.

    Args:
        lmdeploy_tp (int): The tensor parallelism size. Defaults to 1.
        lmdeploy_session_len (Optional[int]): The maximum session length. Defaults to None.
        lmdeploy_cache_max_entry_count (float): The percentage of GPU memory to be used by the K/V cache. Defaults
            to 0.8.
        lmdeploy_quant_policy (int): The quantization policy for the K/V cache. Set to 4 or 8 for 4-bit or 8-bit
            quantization respectively. Defaults to 0, which means no quantization.
        lmdeploy_vision_batch_size (int): The `max_batch_size` parameter to be passed to `VisionConfig`. Defaults to 1.
    """

    # lmdeploy
    lmdeploy_tp: int = 1
    lmdeploy_session_len: Optional[int] = None
    lmdeploy_cache_max_entry_count: float = 0.8
    lmdeploy_quant_policy: int = 0  # e.g. 4, 8
    lmdeploy_vision_batch_size: int = 1  # max_batch_size in VisionConfig

    def get_lmdeploy_engine_kwargs(self):
        kwargs = {
            'tp': self.lmdeploy_tp,
            'session_len': self.lmdeploy_session_len,
            'cache_max_entry_count': self.lmdeploy_cache_max_entry_count,
            'quant_policy': self.lmdeploy_quant_policy,
            'vision_batch_size': self.lmdeploy_vision_batch_size
        }
        if dist.is_initialized():
            kwargs.update({'devices': [dist.get_rank()]})
        return kwargs


@dataclass
class SglangArguments:
    """Arguments for configuring the SGLang backend.

    Args:
        sglang_tp_size (int): The number of tensor parallel workers. Defaults to 1.
        sglang_pp_size (int): The number of pipeline parallel workers. Defaults to 1.
        sglang_dp_size (int): The number of data parallel workers. Defaults to 1.
        sglang_ep_size (int): The number of expert parallel workers. Defaults to 1.
        sglang_enable_ep_moe (bool): Whether to enable expert parallelism for MoE.
            Note: This argument has been removed in recent versions of SGLang. Defaults to False.
        sglang_mem_fraction_static (Optional[float]): The fraction of GPU memory for the static allocation of model
            weights and the KV cache memory pool. Try lowering this value if you encounter GPU out-of-memory errors.
            Defaults to None.
        sglang_context_length (Optional[int]): The maximum context length for the model. If None, the value from the
            model's `config.json` will be used. Defaults to None.
        sglang_disable_cuda_graph (bool): Disable CUDA graph for inference. Defaults to False.
        sglang_quantization (Optional[str]): The quantization method to use. Defaults to None.
        sglang_kv_cache_dtype (str): The data type for K/V cache storage. 'auto' will use the model's data type.
            'fp8_e5m2' and 'fp8_e4m3' are available for CUDA 11.8 and later. Defaults to 'auto'.
        sglang_enable_dp_attention (bool): Enables data parallelism for the attention mechanism and tensor parallelism
            for the feed-forward network (FFN). The data parallel size (dp_size) must equal the tensor parallel size
            (tp_size). Currently supported for DeepSeek-V2/3 and Qwen2/3 MoE models. Defaults to False.
        sglang_disable_custom_all_reduce (bool): Disable the custom all-reduce kernel and fall back to NCCL. Enabled by
            default (True) for stability. Defaults to True.
        sglang_speculative_algorithm (Optional[str]): The speculative decoding algorithm. Options include "EAGLE",
            "EAGLE3", "NEXTN", "STANDALONE", "NGRAM". Defaults to None.
        sglang_speculative_num_steps (Optional[int]): The number of steps to sample from the draft model during
            speculative decoding. Defaults to None.
        sglang_speculative_eagle_topk (Optional[int]): The number of tokens to sample from the draft model at each step
            for the EAGLE2 algorithm. Defaults to None.
        sglang_speculative_num_draft_tokens (Optional[int]): The number of tokens to sample from the draft model during
            speculative decoding. Defaults to None.
    """
    sglang_tp_size: int = 1
    sglang_pp_size: int = 1
    sglang_dp_size: int = 1
    sglang_ep_size: int = 1
    sglang_enable_ep_moe: bool = False
    sglang_mem_fraction_static: Optional[float] = None
    sglang_context_length: Optional[int] = None
    sglang_disable_cuda_graph: bool = False
    sglang_quantization: Optional[str] = None
    sglang_kv_cache_dtype: str = 'auto'
    sglang_enable_dp_attention: bool = False
    sglang_disable_custom_all_reduce: bool = True
    # speculative decoding
    # e.g. EAGLE, EAGLE3, NEXTN
    sglang_speculative_algorithm: Optional[str] = None
    sglang_speculative_num_steps: Optional[int] = None
    sglang_speculative_eagle_topk: Optional[int] = None
    sglang_speculative_num_draft_tokens: Optional[int] = None

    def get_sglang_engine_kwargs(self):
        kwargs = {
            'tp_size': self.sglang_tp_size,
            'pp_size': self.sglang_pp_size,
            'dp_size': self.sglang_dp_size,
            'ep_size': self.sglang_ep_size,
            'enable_ep_moe': self.sglang_enable_ep_moe,
            'mem_fraction_static': self.sglang_mem_fraction_static,
            'context_length': self.sglang_context_length,
            'disable_cuda_graph': self.sglang_disable_cuda_graph,
            'quantization': self.sglang_quantization,
            'kv_cache_dtype': self.sglang_kv_cache_dtype,
            'enable_dp_attention': self.sglang_enable_dp_attention,
            'disable_custom_all_reduce': self.sglang_disable_custom_all_reduce,
            'speculative_algorithm': self.sglang_speculative_algorithm,
            'speculative_num_steps': self.sglang_speculative_num_steps,
            'speculative_eagle_topk': self.sglang_speculative_eagle_topk,
            'speculative_num_draft_tokens': self.sglang_speculative_num_draft_tokens,
        }
        if self.task_type == 'embedding':
            kwargs['task_type'] = 'embedding'
        return kwargs


@dataclass
class InferArguments(MergeArguments, LmdeployArguments, SglangArguments, VllmArguments, BaseArguments):
    """Arguments for model inference.

    A dataclass that extends BaseArguments, MergeArguments, VllmArguments, and LmdeployArguments to define all
    arguments required for model inference.

    Args:
        infer_backend (Literal['transformers', 'vllm', 'sglang', 'lmdeploy']): The inference acceleration
            backend to use. Defaults to 'transformers'.
        result_path (Optional[str]): The path to store inference results in JSONL format. If the file already exists,
            new results will be appended. If None, results are saved in the checkpoint directory (if available) or
            './result'. The final path will be printed to the console. Defaults to None.
        write_batch_size (int): The batch size for writing results to `result_path`. A value of -1 means no limit.
            Defaults to 1000.
        metric (Optional[str]): The metric to use for evaluating inference results. Supported values are 'acc' and
            'rouge'. If None, no evaluation is performed. Defaults to None.
        max_batch_size (int): The maximum batch size for inference, effective only when `infer_backend` is
            'transformers'. A value of -1 means no limit. Defaults to 1.
        val_dataset_sample (Optional[int]): The number of samples to use from the inference dataset. If None, the
            entire dataset is used. Defaults to None.
        reranker_use_activation (bool): Whether to apply a sigmoid activation to the scores during reranker inference.
            Defaults to True.
    """
    # `pt` is used for swift3.x shell script compatibility.
    infer_backend: Literal['vllm', 'transformers', 'sglang', 'lmdeploy', 'pt'] = 'transformers'

    result_path: Optional[str] = None
    write_batch_size: int = 1000
    metric: Literal['acc', 'rouge'] = None
    # for transformers engine
    max_batch_size: int = 1

    # only for inference
    val_dataset_sample: Optional[int] = None

    # for reranker
    reranker_use_activation: bool = True

    def _get_result_path(self, folder_name: str) -> str:
        result_dir = self.ckpt_dir or f'result/{self.model_suffix}'
        os.makedirs(result_dir, exist_ok=True)
        result_dir = to_abspath(os.path.join(result_dir, folder_name))
        os.makedirs(result_dir, exist_ok=True)
        time = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        return os.path.join(result_dir, f'{time}.jsonl')

    def _init_result_path(self, folder_name: str) -> None:
        if self.result_path is not None:
            self.result_path = to_abspath(self.result_path)
            return
        # By default, a result_path file is automatically created
        # when a validation or evaluation dataset is present.
        if self._val_dataset_exists or getattr(self, 'eval_dataset', None):
            self.result_path = self._get_result_path(folder_name)
            logger.info(f'args.result_path: {self.result_path}')

    def _init_stream(self):
        self.eval_human = not self._val_dataset_exists
        logger.info(f'Setting args.eval_human: {self.eval_human}')
        if self.stream is None:
            self.stream = self.eval_human
        if self.stream and self.num_beams != 1:
            self.stream = False
            logger.info('Setting args.stream: False')

    def _init_ddp(self):
        if not is_dist():
            return
        eval_human = getattr(self, 'eval_human', False)
        assert not eval_human and not self.stream, (
            'In DDP scenarios, interactive interfaces and streaming output are not supported.'
            f'args.eval_human: {eval_human}, args.stream: {self.stream}')
        self._init_device()
        init_process_group(backend=self.ddp_backend, timeout=self.ddp_timeout)

    def __post_init__(self) -> None:
        if self.infer_backend == 'pt':
            self.infer_backend = 'transformers'  # compat swift3.x
            logger.warning('args.infer_backend: `pt` is deprecated, please use args.infer_backend: `transformers`.')
        BaseArguments.__post_init__(self)
        VllmArguments.__post_init__(self)
        self._init_vllm_async_engine()
        # Default to False for swift infer (non-encode tasks)
        if self.vllm_use_async_engine is None:
            self.vllm_use_async_engine = False
        self._init_result_path('infer_result')
        self._init_ddp()

    def _init_vllm_async_engine(self):
        """Initialize vllm_use_async_engine based on task_type.

        Encode tasks (embedding, seq_cls, reranker, generative_reranker) require
        async engine because vLLM's synchronous LLMEngine does not have the `encode` method.

        Note: This method only handles encode tasks. For non-encode tasks, the default value
        should be set by subclasses (DeployArguments sets True, RolloutArguments uses
        _set_default_engine_type, InferArguments defaults to False).
        """
        # Task types that require vLLM's encode() method, which is only available in AsyncLLMEngine
        encode_task_types = ('embedding', 'seq_cls', 'reranker', 'generative_reranker')
        is_vllm_encode_task = self.infer_backend == 'vllm' and self.task_type in encode_task_types

        if is_vllm_encode_task:
            if self.vllm_use_async_engine is None:
                self.vllm_use_async_engine = True
            elif not self.vllm_use_async_engine:
                raise ValueError(
                    f'task_type={self.task_type} requires vllm_use_async_engine=True. '
                    f'The synchronous vLLM LLMEngine does not support the `encode` method for encode tasks. '
                    f'Please set --vllm_use_async_engine true or remove the explicit false setting.')
