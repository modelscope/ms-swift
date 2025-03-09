# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, Literal, Optional, Union

import torch
import torch.utils.checkpoint
from transformers.training_args import TrainingArguments as HfTrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments as HfSeq2SeqTrainingArguments

from swift.utils import use_torchacc
from .optimizers.galore import GaLoreConfig


@dataclass
class SwiftArgumentsMixin:
    logging_first_step: bool = True
    acc_strategy: Literal['token', 'seq'] = 'token'
    sequence_parallel_size: int = 1
    check_model: bool = True
    train_sampler_random: bool = True
    is_encoder_decoder: bool = False
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None

    # torchacc
    metric_warmup_step: Optional[float] = 0
    train_dataset_sample: Optional[int] = -1
    fsdp_num: int = 1
    acc_steps: int = 1

    # Value copied from TrainArguments
    train_type: Optional[str] = None
    optimizer: Optional[str] = None
    local_repo_path: Optional[str] = None
    galore_config: Optional[GaLoreConfig] = None

    def _fix_gradient_checkpointing(self):
        # fix use_reentrant
        if hasattr(torch.utils.checkpoint, '_old_checkpoint'):  # avoid double patching
            return
        # Consistent with the default behavior of transformers.
        use_reentrant_ = (
            self.gradient_checkpointing_kwargs.get('use_reentrant', True)
            if self.gradient_checkpointing_kwargs else True)
        _old_checkpoint = torch.utils.checkpoint.checkpoint

        @wraps(_old_checkpoint)
        def _new_checkpoint(*args, use_reentrant=None, **kwargs):
            return _old_checkpoint(*args, use_reentrant=use_reentrant_, **kwargs)

        torch.utils.checkpoint._old_checkpoint = _old_checkpoint
        torch.utils.checkpoint.checkpoint = _new_checkpoint
        try:
            # Fix the old version of transformers.
            import transformers.modeling_utils
            transformers.modeling_utils.checkpoint = _new_checkpoint
        except (ImportError, AttributeError):
            pass

    def __post_init__(self):
        if hasattr(self, 'output_dir'):
            self.output_dir = os.path.abspath(os.path.expanduser(self.output_dir))
        self._fix_gradient_checkpointing()
        super().__post_init__()

    @property
    def place_model_on_device(self):
        return False if use_torchacc() else super().place_model_on_device


@dataclass
class GRPOArgumentsMixin:
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.
    # vllm_device, vllm_gpu_memory_utilization, and vllm_max_model_len are defined in HfGRPOConfig.
    num_infer_workers: int = 1
    vllm_max_num_seqs: int = 256
    vllm_enforce_eager: bool = False
    vllm_limit_mm_per_prompt: Optional[Union[dict, str]] = None  # '{"image": 5, "video": 2}'
    vllm_enable_prefix_caching: bool = True
    # reward function args, see details in swift/plugin/orm.py
    # cosine reward, https://arxiv.org/abs/2502.03373
    cosine_min_len_value_wrong: float = 0.0  # r^w_0 in paper, Reward for wrong answers with zero completion length.
    cosine_max_len_value_wrong: float = -0.5  # r^w_L in paper, Reward for wrong answers with max completion length.
    cosine_min_len_value_correct: float = 1.0  # r^c_0 in paper, Reward for correct answers with zero completion length.
    cosine_max_len_value_correct: float = 0.5  # r^c_L in paper, Reward for correct answers with max completion length.
    cosine_max_len: Optional[int] = None  # Lmax in paper, default equal to max_completion_length
    # repetition penalty, https://arxiv.org/abs/2502.03373
    repetition_n_grams: int = 3
    repetition_max_penalty: float = -1.0

    # LMDeploy in GRPO
    use_lmdeploy: bool = False
    lmdeploy_device: Optional[str] = 'auto'
    lmdeploy_session_len: Optional[int] = None
    lmdeploy_cache_max_entry_count: float = 0.8

    async_generate: bool = False
    tensor_parallel_size: int = 1
    sleep_level: int = 0
    move_model_batches: Optional[int] = None
    offload_optimizer: bool = False
    offload_model: bool = False
    gc_collect_after_offload: bool = False


@dataclass
class TrainingArguments(SwiftArgumentsMixin, HfTrainingArguments):
    pass


@dataclass
class Seq2SeqTrainingArguments(SwiftArgumentsMixin, HfSeq2SeqTrainingArguments):
    pass
