import os
from dataclasses import dataclass, field, fields
from typing import List, Optional, Union

import json

from swift.hub import get_hub
from swift.utils import check_json_format, get_logger, is_master
from ..tuner_args import TunerArguments, get_supported_tuners
from .data_args import DataArguments
from .model_args import ModelArguments
from .template_args import TemplateArguments
from .utils import to_abspath

logger = get_logger()


@dataclass
class BaseArguments(DataArguments, TemplateArguments, ModelArguments):
    """
    BaseArguments class is a dataclass that inherits from multiple argument classes:
    ModelArguments, TemplateArguments, and DataArguments.

    Args:
        seed (int): Random seed for reproducibility. Default is 42.
        load_dataset_config (bool): Flag to determine if dataset configuration should be loaded. Default is False.
        save_safetensors (bool): Flag to determine if save to safetensors. Default is True.
        hub_token (Optional[str]): SDK token for authentication. Default is None.
        ignore_args_error (bool): Flag to ignore argument errors for notebook compatibility. Default is False.
    """
    seed: int = 42
    strict: bool = False
    load_dataset_config: bool = False
    use_hf: bool = False
    # None: use env var `MODELSCOPE_API_TOKEN`
    hub_token: Optional[str] = field(
        default=None, metadata={'help': 'SDK token can be found in https://modelscope.cn/my/myaccesstoken'})

    # extra
    ignore_args_error: bool = False  # True: notebook compatibility

    def __post_init__(self):
        if self.use_hf:
            os.environ['USE_HF'] = '1'
        ModelArguments.__post_init__(self)
        TemplateArguments.__post_init__(self)
        DataArguments.__post_init__(self)
        self.hub = get_hub(self.use_hf)
        if self.hub.try_login(self.hub_token):
            logger.info('hub login successful!')

    @property
    def supported_tuners(self):
        return get_supported_tuners()

    @property
    def adapters_can_be_merged(self):
        return TunerArguments.adapters_can_be_merged

    @staticmethod
    def load_args_from_ckpt(checkpoint_dir: str) -> None:
        """Load specific attributes from args.json"""
        args_path = os.path.join(checkpoint_dir, 'args.json')
        if not os.path.exists(args_path):
            logger.warning(f'{args_path} not found')
            return
        with open(args_path, 'r', encoding='utf-8') as f:
            old_args = json.load(f)
        # read settings
        all_keys = list(f.name for f in fields(self.__class__))
        data_keys = list(f.name for f in fields(DataArguments))
        for key in all_keys:
            if not self.load_dataset_config and key in data_keys:
                continue
            old_value = old_args.get(key)
            if old_value is None:
                continue
            value = getattr(self, key, None)
            if value is None or isinstance(value, (list, tuple)) and len(value) == 0:
                setattr(self, key, old_value)

    def save_args(self) -> None:
        from swift.llm import InferArguments
        if isinstance(self, InferArguments):
            return
        if is_master():
            fpath = os.path.join(self.output_dir, 'args.json')
            logger.info(f'The {self.__class__.__name__} will be saved in: {fpath}')
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(check_json_format(self.__dict__), f, ensure_ascii=False, indent=2)
