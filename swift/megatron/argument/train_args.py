# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
from dataclasses import dataclass

from swift.llm import BaseArguments
from ..model import get_megatron_model_meta
from .megatron_args import MegatronArguments


@dataclass
class MegatronTrainArguments(MegatronArguments, BaseArguments):

    def _init_model_args(self):
        self.megatron_model_meta = get_megatron_model_meta(self.model)
        kwargs = self.megatron_model_meta.load_config(self.model_info.config)
        for k, v in kwargs.items():
            if getattr(self, k) is None:
                setattr(self, k, v)

    def __post_init__(self):
        BaseArguments.__post_init__(self)
        MegatronArguments.__post_init__(self)
        if self.hf_ckpt_path is None:
            self.hf_ckpt_path = self.model_dir
        self._init_model_args()
        self.extra_args = self.parse_to_megatron()
