# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Literal, Optional

import json
import torch
from transformers.utils import is_torch_npu_available

from swift.hub import get_hub
from swift.plugin import extra_tuners
from swift.utils import check_json_format, get_dist_setting, get_logger, is_dist, is_master, use_hf_hub
from .data_args import DataArguments
from .generation_args import GenerationArguments
from .model_args import ModelArguments
from .quant_args import QuantizeArguments
from .template_args import TemplateArguments
from .utils import to_abspath

logger = get_logger()


def get_supported_tuners():
    return {'lora', 'full', 'longlora', 'adalora', 'llamapro', 'adapter', 'vera', 'boft', 'fourierft', 'reft', 'bone'
            } | set(extra_tuners.keys())


@dataclass
class BaseArguments(GenerationArguments, QuantizeArguments, DataArguments, TemplateArguments, ModelArguments):
    """
    BaseArguments class is a dataclass that inherits from multiple argument classes:
    GenerationArguments, QuantizeArguments, DataArguments, TemplateArguments, ModelArguments.

    Args:
        tuner_backend(str): Support peft or unsloth.
        train_type(str): The training type, support all supported tuners and `full`.
        seed (int): Random seed for reproducibility. Default is 42.
        model_kwargs (Optional[str]): Additional keyword arguments for the model. Default is None.
        load_dataset_config (bool): Flag to determine if dataset configuration should be loaded. Default is False.
        use_hf (bool): Flag to determine if Hugging Face should be used. Default is False.
        hub_token (Optional[str]): SDK token for authentication. Default is None.
        custom_register_path (List[str]): Path to custom .py file for dataset registration. Default is None.
        ignore_args_error (bool): Flag to ignore argument errors for notebook compatibility. Default is False.
        use_swift_lora (bool): Use swift lora, a compatible argument
    """
    tuner_backend: Literal['peft', 'unsloth'] = 'peft'
    train_type: str = field(default='lora', metadata={'help': f'train_type choices: {list(get_supported_tuners())}'})

    seed: int = 42
    model_kwargs: Optional[str] = None
    load_dataset_config: bool = False

    use_hf: bool = False
    # None: use env var `MODELSCOPE_API_TOKEN`
    hub_token: Optional[str] = field(
        default=None, metadata={'help': 'SDK token can be found in https://modelscope.cn/my/myaccesstoken'})
    custom_register_path: List[str] = field(default_factory=list)  # .py

    # extra
    ignore_args_error: bool = False  # True: notebook compatibility
    use_swift_lora: bool = False  # True for using tuner_backend == swift, don't specify this unless you know what you are doing # noqa

    def _init_custom_register(self) -> None:
        """Register custom .py file to datasets"""
        self.custom_register_path = to_abspath(self.custom_register_path, True)
        for path in self.custom_register_path:
            folder, fname = os.path.split(path)
            sys.path.append(folder)
            __import__(fname.rstrip('.py'))
        logger.info(f'Successfully registered `{self.custom_register_path}`')

    def __post_init__(self):
        if self.use_hf or use_hf_hub():
            self.use_hf = True
            os.environ['USE_HF'] = '1'
        self._init_custom_register()
        self._init_model_kwargs()
        self.rank, self.local_rank, world_size, self.local_world_size = get_dist_setting()
        # The Seq2SeqTrainingArguments has a property called world_size, which cannot be assigned a value.
        try:
            self.world_size = world_size
        except AttributeError:
            pass
        logger.info(f'rank: {self.rank}, local_rank: {self.local_rank}, '
                    f'world_size: {world_size}, local_world_size: {self.local_world_size}')
        ModelArguments.__post_init__(self)
        QuantizeArguments.__post_init__(self)
        TemplateArguments.__post_init__(self)
        DataArguments.__post_init__(self)
        self.hub = get_hub(self.use_hf)
        if self.hub.try_login(self.hub_token):
            logger.info('hub login successful!')

    def _init_model_kwargs(self):
        """Prepare model kwargs and set them to the env"""
        self.model_kwargs: Dict[str, Any] = self.parse_to_dict(self.model_kwargs)
        for k, v in self.model_kwargs.items():
            k = k.upper()
            os.environ[k] = str(v)

    @property
    def is_adapter(self) -> bool:
        return self.train_type not in {'full'}

    @property
    def supported_tuners(self):
        return get_supported_tuners()

    @property
    def adapters_can_be_merged(self):
        return {'lora', 'longlora', 'llamapro', 'adalora'}

    def load_args_from_ckpt(self, checkpoint_dir: str) -> None:
        """Load specific attributes from args.json"""
        args_path = os.path.join(checkpoint_dir, 'args.json')
        if not os.path.exists(args_path):
            return
        logger.info(f'Successfully loaded {args_path}...')
        with open(args_path, 'r', encoding='utf-8') as f:
            old_args = json.load(f)
        # read settings
        all_keys = list(f.name for f in fields(self.__class__))
        data_keys = list(f.name for f in fields(DataArguments))
        load_keys = [
            'bnb_4bit_quant_type', 'bnb_4bit_use_double_quant', 'split_dataset_ratio', 'model_name', 'model_author',
            'train_type', 'tuner_backend'
        ]
        skip_keys = [
            'output_dir',
            'deepspeed',
            'temperature',
            'max_new_tokens',
        ]
        for key in all_keys:
            if key in skip_keys:
                continue
            if not self.load_dataset_config and key in data_keys:
                continue
            old_value = old_args.get(key)
            if old_value is None:
                continue
            value = getattr(self, key, None)
            if value is None or isinstance(value, (list, tuple)) and len(value) == 0 or key in load_keys:
                setattr(self, key, old_value)

    def save_args(self) -> None:
        if is_master():
            os.makedirs(self.output_dir, exist_ok=True)
            fpath = os.path.join(self.output_dir, 'args.json')
            logger.info(f'The {self.__class__.__name__} will be saved in: {fpath}')
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(check_json_format(self.__dict__), f, ensure_ascii=False, indent=2)

    def _init_weight_type(self, ckpt_dir):
        if ckpt_dir and (os.path.exists(os.path.join(ckpt_dir, 'adapter_config.json'))
                         or os.path.exists(os.path.join(ckpt_dir, 'default', 'adapter_config.json'))
                         or os.path.exists(os.path.join(ckpt_dir, 'reft'))):
            self.weight_type = 'adapter'
        else:
            self.weight_type = 'full'
            self.model = ckpt_dir or self.model

    def _init_device(self):
        if is_dist():
            if is_torch_npu_available():
                torch.npu.set_device(self.local_rank)
            else:
                torch.cuda.set_device(self.local_rank)
