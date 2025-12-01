# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import json
import torch
from transformers.utils import is_torch_mps_available

from swift.llm import MODEL_MAPPING, HfConfigFactory, get_model_info_meta, get_model_name
from swift.utils import get_dist_setting, get_logger, json_parse_to_dict

logger = get_logger()


@dataclass
class ModelArguments:
    """A dataclass that holds various arguments related to model configuration and usage.

    Args:
        model (Optional[str]): The model ID from the Hub or a local path to the model. Defaults to None.
        model_type (Optional[str]): The model type. In ms-swift, a 'model_type' groups models with the same
            architecture, loading process, and template. Defaults to None, which enables auto-selection based on
            the suffix of `--model` and the 'architectures' attribute in `config.json`. The `model_type` for a
            corresponding model can be found in the list of supported models. Note: The concept of `model_type`
            in ms-swift differs from the `model_type` in `config.json`. Custom models usually require registering
            their own `model_type` and `template`.
        model_revision (Optional[str]): The revision of the model. Defaults to None.
        task_type (str): The task type. Can be 'causal_lm', 'seq_cls', 'embedding', 'reranker', or
            'generative_reranker'. If set to 'seq_cls', you usually need to specify `--num_labels` and
            `--problem_type`. Defaults to 'causal_lm'.
        torch_dtype (Optional[str]): The data type of the model weights. Supports 'float16', 'bfloat16', 'float32'.
            Defaults to None, in which case it's read from the 'config.json' file.
        attn_impl (Optional[str]): The attention implementation to use. Options include 'sdpa', 'eager', 'flash_attn',
            'flash_attention_2', 'flash_attention_3', etc. Defaults to None, which means it will be read from
            'config.json'. Note: Support for these implementations depends on the model's transformers implementation.
            If set to 'flash_attn' (for backward compatibility), 'flash_attention_2' will be used.
        new_special_tokens (List[str]): Additional special tokens to be added to the tokenizer. Can also be a path to
            a `.txt` file, where each line is a special token. Defaults to an empty list `[]`.
        num_labels (Optional[int]): The number of labels for classification tasks (when `--task_type` is 'seq_cls').
            Required for such tasks. Defaults to None.
        problem_type (Optional[str]): The problem type for classification tasks (`--task_type` 'seq_cls'). Options are
            'regression', 'single_label_classification', 'multi_label_classification'. Defaults to None, but is
            automatically set to 'regression' if the model is a reward_model or `num_labels` is 1, and
            'single_label_classification' otherwise.
        rope_scaling (Optional[str]): The RoPE scaling type. You can pass a string like 'linear', 'dynamic', or
            'yarn', and ms-swift will automatically set the corresponding `rope_scaling` and override the
            'config.json' value. Alternatively, you can pass a JSON string (e.g., '{"factor":2.0, "type":"yarn"}'),
            which will directly override the `rope_scaling` in 'config.json'. Defaults to None.
        device_map (Optional[str]): The device map configuration for the model, e.g., 'auto', 'cpu', a JSON string,
            or a path to a JSON file. This argument is passed directly to the `from_pretrained` method of transformers.
            Defaults to None, and will be set automatically based on the device and distributed training settings.
        max_memory (Optional[str]): The maximum memory allocation for each device when `device_map` is 'auto' or
            'sequential'. Example: '{0: "20GB", 1: "20GB"}'. This argument is passed directly to the `from_pretrained`
            method of transformers. Defaults to None.
        max_model_len (Optional[int]): The maximum model length. This is used to calculate the RoPE scaling factor
            when `rope_scaling` is specified as a string. If not None, it overrides the `max_position_embeddings`
            value in 'config.json'. Defaults to None.
        local_repo_path (Optional[str]): Path to a local repository for models that require a GitHub repo during
            loading (e.g., deepseek-vl2). This avoids network issues during `git clone`. Defaults to None.
        init_strategy (Optional[str]): The strategy to initialize all uninitialized parameters when loading a model
            (especially for custom architectures). Options include 'zero', 'uniform', 'normal', 'xavier_uniform',
            'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'orthogonal'. Defaults to None.
    """
    model: Optional[str] = None  # model id or model path
    model_type: Optional[str] = field(
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    model_revision: Optional[str] = None
    task_type: Literal['causal_lm', 'seq_cls', 'embedding', 'reranker', 'generative_reranker'] = None

    torch_dtype: Literal['bfloat16', 'float16', 'float32', None] = None
    # flash_attn: It will automatically convert names based on the model.
    # None: It will be automatically selected between sdpa and eager.
    # 'flash_attn', 'sdpa', 'eager', 'flex_attention', 'flash_attention_2', 'flash_attention_3'
    attn_impl: Optional[str] = None
    new_special_tokens: List[str] = field(default_factory=list)

    num_labels: Optional[int] = None
    problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] = None
    rope_scaling: Optional[str] = None
    device_map: Optional[Union[dict, str]] = None
    max_memory: Optional[Union[dict, str]] = None
    max_model_len: Optional[int] = None
    # When some model code needs to be downloaded from GitHub,
    # this parameter specifies the path to the locally downloaded repository.
    local_repo_path: Optional[str] = None
    init_strategy: Literal['zero', 'uniform', 'normal', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform',
                           'kaiming_normal', 'orthogonal'] = None

    def _init_device_map(self):
        """Prepare device map args"""
        if self.device_map:
            self.device_map: Union[str, Dict[str, Any], None] = json_parse_to_dict(self.device_map, strict=False)
        # compat mp&ddp
        _, local_rank, _, local_world_size = get_dist_setting()
        if local_world_size > 1 and isinstance(self.device_map, dict) and local_rank > 0:
            for k, v in self.device_map.items():
                if isinstance(v, int):
                    self.device_map[k] += local_rank

    def _init_max_memory(self):
        if isinstance(self.max_memory, str):
            try:
                self.max_memory = ast.literal_eval(self.max_memory)
            except Exception:
                pass
        self.max_memory = json_parse_to_dict(self.max_memory)
        # compat mp&ddp
        _, local_rank, _, local_world_size = get_dist_setting()
        if local_world_size > 1 and isinstance(self.max_memory, dict) and local_rank > 0:
            for k in list(self.max_memory.keys()):
                if isinstance(k, int):
                    self.max_memory[k + local_rank] = self.max_memory.pop(k)

    def _init_torch_dtype(self) -> None:
        """"If torch_dtype is None, find a proper dtype by the train_type/GPU"""
        from swift.llm import TrainArguments

        self.torch_dtype: Optional[torch.dtype] = HfConfigFactory.to_torch_dtype(self.torch_dtype)
        self.torch_dtype: torch.dtype = self._init_model_info()
        # Mixed Precision Training
        if isinstance(self, TrainArguments):
            self._init_mixed_precision()

    def _init_mixed_precision(self):
        if is_torch_mps_available():
            fp16, bf16 = False, False
        elif self.torch_dtype in {torch.float16, torch.float32}:
            fp16, bf16 = True, False
        elif self.torch_dtype == torch.bfloat16:
            fp16, bf16 = False, True
        else:
            raise ValueError(f'args.torch_dtype: {self.torch_dtype}')
        if self.fp16 is None:
            self.fp16 = fp16
        if self.bf16 is None:
            self.bf16 = bf16

    def _init_rope_scaling(self):
        if self.rope_scaling:
            rope_scaling: dict = json_parse_to_dict(self.rope_scaling, strict=False)
            if isinstance(rope_scaling, str):
                assert rope_scaling in ['linear', 'dynamic', 'yarn']
                rope_scaling = {'type': rope_scaling}
        else:
            rope_scaling = self.model_info.rope_scaling
            # reset the factor
            rope_scaling.pop('factor', None)

        if 'factor' not in rope_scaling and self.max_model_len is None:
            # fix megatron qwen2_5_vl
            self.rope_scaling = rope_scaling
            logger.info(f'Setting args.rope_scaling: {rope_scaling}')
            return

        # get origin_max_model_len
        origin_max_model_len = None
        if rope_scaling and rope_scaling.get('original_max_position_embeddings') is not None:
            origin_max_model_len = rope_scaling['original_max_position_embeddings']
        elif self.model_info.rope_scaling:
            if self.model_info.rope_scaling.get('original_max_position_embeddings') is not None:
                origin_max_model_len = self.model_info.rope_scaling['original_max_position_embeddings']
            elif self.model_info.rope_scaling.get('factor') is not None:
                origin_max_model_len = self.model_info.max_model_len // self.model_info.rope_scaling['factor']
        if origin_max_model_len is None:
            origin_max_model_len = self.model_info.max_model_len
        assert origin_max_model_len is not None, '`origin_max_model_len` from model config is not set'
        rope_scaling['original_max_position_embeddings'] = origin_max_model_len

        if 'factor' not in rope_scaling:
            rope_scaling['factor'] = max(float(math.ceil(self.max_model_len / origin_max_model_len)), 1.0)
        rope_model_len = int(origin_max_model_len * rope_scaling['factor'])
        if self.max_model_len is None:
            self.max_model_len = rope_model_len
        elif self.max_model_len > rope_model_len:
            logger.warning(f'rope config ({rope_model_len} = {rope_scaling["factor"]} * '
                           f'{origin_max_model_len}) should be bigger than max_model_len '
                           f'from command line ({self.max_model_len})')
        self.rope_scaling = rope_scaling
        logger.info(f'Setting args.rope_scaling: {rope_scaling}')
        logger.info(f'Setting args.max_model_len: {self.max_model_len}')

    def _init_model_info(self) -> torch.dtype:
        self.model_info, self.model_meta = get_model_info_meta(**self.get_model_kwargs())
        self.task_type = self.model_info.task_type
        self.num_labels = self.model_info.num_labels

        self.model_dir = self.model_info.model_dir
        self.model_type = self.model_info.model_type
        if self.rope_scaling or self.model_info.rope_scaling and self.max_model_len is not None:
            self._init_rope_scaling()
        return self.model_info.torch_dtype

    def _init_new_special_tokens(self):
        if isinstance(self.new_special_tokens, str):
            self.new_special_tokens = [self.new_special_tokens]
        new_special_tokens = []
        for token in self.new_special_tokens:
            if token.endswith('.txt'):
                assert os.path.isfile(token), f'special_tokens_path: {token}'
                with open(token, 'r') as f:
                    text = f.read()
                new_special_tokens += text.split()
            else:
                new_special_tokens.append(token)
        self.new_special_tokens = new_special_tokens

    def __post_init__(self):
        if self.model is None:
            raise ValueError(f'Please set --model <model_id_or_path>`, model: {self.model}')
        self._init_new_special_tokens()
        self.model_suffix = get_model_name(self.model)
        self._init_device_map()
        self._init_max_memory()
        self._init_torch_dtype()

    def get_model_kwargs(self):
        return {
            'model_id_or_path': self.model,
            'torch_dtype': self.torch_dtype,
            'model_type': self.model_type,
            'revision': self.model_revision,
            'use_hf': self.use_hf,
            'hub_token': self.hub_token,
            'local_repo_path': self.local_repo_path,
            'device_map': self.device_map,
            'max_memory': self.max_memory,
            'quantization_config': self.get_quantization_config(),
            'attn_impl': self.attn_impl,
            'new_special_tokens': self.new_special_tokens,
            'rope_scaling': self.rope_scaling,
            'max_model_len': self.max_model_len,
            'task_type': self.task_type,
            'num_labels': self.num_labels,
            'problem_type': self.problem_type,
            'init_strategy': self.init_strategy,
        }
