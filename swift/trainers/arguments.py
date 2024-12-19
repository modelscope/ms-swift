# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, Literal, Optional

import torch
import torch.utils.checkpoint
from transformers.training_args import TrainingArguments as HfTrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments as HfSeq2SeqTrainingArguments

from swift.utils import use_torchacc


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

    # Value copied from TrainArguments, Used for external tuners.
    train_type: Optional[str] = None

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
class TrainingArguments(SwiftArgumentsMixin, HfTrainingArguments):
    pass


@dataclass
class Seq2SeqTrainingArguments(SwiftArgumentsMixin, HfSeq2SeqTrainingArguments):
    pass


try:
    from trl import (DPOConfig as HfDPOConfig, CPOConfig as HfCPOConfig, ORPOConfig as HfORPOConfig, KTOConfig as
                     HfKTOConfig, RewardConfig as HfRewardConfig, PPOv2Config as HfPPOConfig)

    @dataclass
    class DPOConfig(SwiftArgumentsMixin, HfDPOConfig):
        pass

    @dataclass
    class CPOConfig(SwiftArgumentsMixin, HfCPOConfig):
        pass

    @dataclass
    class ORPOConfig(SwiftArgumentsMixin, HfORPOConfig):
        pass

    @dataclass
    class KTOConfig(SwiftArgumentsMixin, HfKTOConfig):
        pass

    @dataclass
    class RewardConfig(SwiftArgumentsMixin, HfRewardConfig):
        pass

    @dataclass
    class PPOConfig(SwiftArgumentsMixin, HfPPOConfig):
        pass

except (ImportError, RuntimeError):
    DPOConfig = None
    CPOConfig = None
    ORPOConfig = None
    KTOConfig = None
    RewardConfig = None
    PPOConfig = None
