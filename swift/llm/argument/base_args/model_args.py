# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import json
import torch

from swift.llm import MODEL_MAPPING, HfConfigFactory, get_model_info_meta, get_model_name
from swift.utils import get_dist_setting, get_logger

logger = get_logger()


@dataclass
class ModelArguments:
    """
    ModelArguments class is a dataclass that holds various arguments related to model configuration and usage.

    Args:
        model (Optional[str]): Model identifier or path. Default is None.
        model_type (Optional[str]): Type of the model group. Default is None.
        model_revision (Optional[str]): Revision of the model. Default is None.
        torch_dtype (Literal): Model data type. Default is None.
        attn_impl (Literal): Attention implementation to use. Default is None.
        rope_scaling (Literal): Type of rope scaling to use. Default is None.
        device_map (Optional[str]): Configuration for device mapping. Default is None.
        local_repo_path (Optional[str]): Path to the local repository for model code. Default is None.
    """
    model: Optional[str] = None  # model id or model path
    model_type: Optional[str] = field(
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    model_revision: Optional[str] = None

    torch_dtype: Literal['bfloat16', 'float16', 'float32', None] = None
    # flash_attn: It will automatically convert names based on the model.
    # None: It will be automatically selected between sdpa and eager.
    attn_impl: Literal['flash_attn', 'sdpa', 'eager', None] = None

    rope_scaling: Literal['linear', 'dynamic'] = None
    device_map: Optional[str] = None
    # When some model code needs to be downloaded from GitHub,
    # this parameter specifies the path to the locally downloaded repository.
    local_repo_path: Optional[str] = None

    @staticmethod
    def parse_to_dict(value: Union[str, Dict, None], strict: bool = True) -> Union[str, Dict]:
        """Convert a JSON string or JSON file into a dict"""
        # If the value could potentially be a string, it is generally advisable to set strict to False.
        if value is None:
            value = {}
        elif isinstance(value, str):
            if os.path.exists(value):  # local path
                with open(value, 'r') as f:
                    value = json.load(f)
            else:  # json str
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    if strict:
                        raise
        return value

    def _init_device_map(self):
        """Prepare device map args"""
        if self.device_map:
            self.device_map: Union[str, Dict[str, Any], None] = self.parse_to_dict(self.device_map, strict=False)
        # compat mp&ddp
        _, local_rank, _, local_world_size = get_dist_setting()
        if local_world_size > 1 and isinstance(self.device_map, dict) and local_rank > 0:
            for k, v in self.device_map.items():
                if isinstance(v, int):
                    self.device_map[k] += local_rank

    def _init_torch_dtype(self) -> None:
        """"If torch_dtype is None, find a proper dtype by the train_type/GPU"""
        from swift.llm import TrainArguments
        if self.torch_dtype is None and isinstance(self, TrainArguments):
            # Compatible with --fp16/--bf16
            for key in ['fp16', 'bf16']:
                value = getattr(self, key)
                if value:
                    self.torch_dtype = {'fp16': 'float16', 'bf16': 'bfloat16'}[key]

        self.torch_dtype: Optional[torch.dtype] = HfConfigFactory.to_torch_dtype(self.torch_dtype)
        self.torch_dtype: torch.dtype = self._init_model_info()
        # Mixed Precision Training
        if isinstance(self, TrainArguments):
            if self.torch_dtype in {torch.float16, torch.float32}:
                self.fp16, self.bf16 = True, False
            elif self.torch_dtype == torch.bfloat16:
                self.fp16, self.bf16 = False, True
            else:
                raise ValueError(f'args.torch_dtype: {self.torch_dtype}')

    def _init_rope_scaling(self):
        assert self.max_length is not None, 'Use max_model_len together with rope_scaling'
        rope_scaling = self.model_info.rope_scaling or {}
        max_model_len = self.model_info.max_model_len
        rope_scaling_factor = 1.0
        if max_model_len:
            rope_scaling_factor = max(float(math.ceil(self.max_length / max_model_len)), 1.0)
        if rope_scaling:
            rope_scaling_factor = max(rope_scaling.get('factor', -1), rope_scaling_factor)
            rope_scaling['type'] = self.rope_scaling
            rope_scaling['factor'] = rope_scaling_factor
        else:
            rope_scaling = {'type': self.rope_scaling, 'factor': rope_scaling_factor}
        self.rope_scaling = rope_scaling
        logger.info(f'rope_scaling is set to type: {self.rope_scaling}')

    def _init_model_info(self) -> torch.dtype:
        self.model_info, self.model_meta = get_model_info_meta(**self.get_model_kwargs())
        self.model_dir = self.model_info.model_dir
        self.model_type = self.model_info.model_type
        if isinstance(self.rope_scaling, str):
            self._init_rope_scaling()
        return self.model_info.torch_dtype

    def __post_init__(self):
        self.model_suffix = get_model_name(self.model)
        self._init_device_map()
        self._init_torch_dtype()

    def get_model_kwargs(self):
        if self.model is None:
            raise ValueError('Please set --model <model_id_or_path>`')
        return {
            'model_id_or_path': self.model,
            'torch_dtype': self.torch_dtype,
            'model_type': self.model_type,
            'revision': self.model_revision,
            'use_hf': self.use_hf,
            'hub_token': self.hub_token,
            'local_repo_path': self.local_repo_path,
            'device_map': self.device_map,
            'quantization_config': self.get_quantization_config(),
            'attn_impl': self.attn_impl,
            'rope_scaling': self.rope_scaling,
        }
