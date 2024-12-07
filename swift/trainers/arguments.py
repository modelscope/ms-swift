# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from transformers.training_args import TrainingArguments as HfTrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments as HfSeq2SeqTrainingArguments
from transformers.utils import is_accelerate_available

from swift.utils import is_dist, use_torchacc


@dataclass
class SwiftArgumentsMixin:
    logging_first_step: bool = True
    acc_strategy: Literal['token', 'seq'] = 'token'
    sequence_parallel_size: int = 1
    check_model: bool = True
    train_sampler_random: bool = True
    is_encoder_decoder: bool = False

    # torchacc
    metric_warmup_step: Optional[float] = 0
    train_dataset_sample: Optional[int] = -1
    fsdp_num: int = 1
    acc_steps: int = 1

    # Value copied from TrainArguments, Used for external tuners.
    train_type: Optional[str] = None

    def __post_init__(self):
        if hasattr(self, 'output_dir'):
            self.output_dir = os.path.abspath(os.path.expanduser(self.output_dir))
        if is_dist() and self.ddp_backend == 'nccl' and torch.cuda.is_available() and is_accelerate_available():
            try:
                from accelerate.utils import check_cuda_p2p_ib_support
                if not check_cuda_p2p_ib_support():
                    os.environ['NCCL_P2P_DISABLE'] = '1'
                    os.environ['NCCL_IB_DISABLE'] = '1'
            except ImportError:
                pass
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
