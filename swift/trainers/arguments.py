# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from transformers.training_args import TrainingArguments as HfTrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments as HfSeq2SeqTrainingArguments
from transformers.utils import is_accelerate_available

from swift.utils import is_dist, use_torchacc
from .loss import LOSS_MAPPING


@dataclass
class SwiftArgumentsMixin:
    # ckpt only save model
    save_only_model: bool = False
    acc_strategy: str = field(default='token', metadata={'choices': ['token', 'sentence']})
    loss_name: Optional[str] = field(default=None, metadata={'help': f'loss_func choices: {list(LOSS_MAPPING.keys())}'})
    additional_saved_files: Optional[List[str]] = None
    # torchacc
    train_sampler_random: bool = True
    metric_warmup_step: Optional[float] = 0
    train_dataset_sample: Optional[int] = -1

    def __post_init__(self):
        if is_dist() and self.ddp_backend == 'nccl' and torch.cuda.is_available() and is_accelerate_available():
            try:
                from accelerate.utils import check_cuda_p2p_ib_support
                if not check_cuda_p2p_ib_support():
                    os.environ['NCCL_P2P_DISABLE'] = '1'
                    os.environ['NCCL_IB_DISABLE'] = '1'
            except ImportError:
                pass
        if self.additional_saved_files is None:
            self.additional_saved_files = []
        super().__post_init__()


@dataclass
class TrainingArguments(SwiftArgumentsMixin, HfTrainingArguments):
    pass


@dataclass
class Seq2SeqTrainingArguments(SwiftArgumentsMixin, HfSeq2SeqTrainingArguments):

    @property
    def place_model_on_device(self):
        return False if use_torchacc() else super().place_model_on_device


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
