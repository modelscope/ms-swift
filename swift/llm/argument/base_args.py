import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

import json

from swift.utils import get_logger
from .data_args import DataArguments, TemplateArguments
from .model_args import GenerationArguments, ModelArguments, QuantizeArguments

logger = get_logger()


@dataclass
class BaseArguments(ModelArguments, TemplateArguments, QuantizeArguments, GenerationArguments, DataArguments):
    seed: int = 42

    ignore_args_error: bool = False  # True: notebook compatibility
    save_safetensors: bool = True
    # None: use env var `MODELSCOPE_API_TOKEN`
    hub_token: Optional[str] = field(
        default=None, metadata={'help': 'SDK token can be found in https://modelscope.cn/my/myaccesstoken'})

    def __init__(self: Union['SftArguments', 'InferArguments']):
        ModelArguments.__post_init__(self)
        TemplateArguments.__post_init__(self)
        DataArguments.__post_init__(self)
        QuantizeArguments.__post_init__(self)
        GenerationArguments.__post_init__(self)
        self.handle_path()
        from swift.hub import default_hub
        if default_hub.try_login(self.hub_token):
            logger.info('hub login successful!')

    def parse_to_dict(self, key: str) -> None:
        """Convert a JSON string or JSON file into a dict"""
        value = getattr(self, key)
        if value is None:
            value = {}
        elif isinstance(value, str):
            if os.path.exists(value):  # local path
                with open(value, 'r') as f:
                    value = json.load(f)
            else:  # json str
                value = json.loads(value)
        setattr(self, key, value)

    @staticmethod
    def check_path_validity(path: Union[str, List[str]], check_path_exist: bool = False) -> Union[str, List[str]]:
        """Check the path for validity and convert it to an absolute path.

        Args:
            path: The path to be checked/converted
            check_path_exist: Whether to check if the path exists

        Returns:
            Absolute path
        """
        if isinstance(path, str):
            # Remove user path prefix and convert to absolute path.
            path = os.path.abspath(os.path.expanduser(path))
            if check_path_exist and not os.path.exists(path):
                raise FileNotFoundError(f"path: '{path}'")
            return path
        assert isinstance(path, list), f'path: {path}'
        res = []
        for v in path:
            res.append(BaseArguments.check_path_validity(v, check_path_exist))
        return res

    def handle_path(self) -> None:
        """Check all paths in the args correct and exist"""
        check_exist_path = {'ckpt_dir', 'resume_from_checkpoint', 'custom_register_path', 'deepspeed_config_path'}
        other_path = {'output_dir', 'logging_dir'}
        # If it is a path, check it.
        maybe_check_exist_path = ['model_id_or_path', 'custom_dataset_info', 'deepspeed']
        for k in maybe_check_exist_path:
            v = getattr(self, k, None)
            if os.path.exists(v) or isinstance(v, str) and v[:1] in {'~', '/'}:  # startswith
                check_exist_path.add(k)
        # check path
        for k in check_exist_path | other_path:
            value = getattr(self, k, None)
            if value is None:
                continue
            value = self.check_path_validity(value, k in check_exist_path)
            setattr(self, k, value)

    def load_from_ckpt_dir(self: Union['SftArguments', 'InferArguments']) -> None:
        """Load specific attributes from sft_args.json"""
        from swift.llm import SftArguments, ExportArguments, InferArguments
        if isinstance(self, SftArguments):
            ckpt_dir = self.resume_from_checkpoint
        else:
            ckpt_dir = self.ckpt_dir
        # Determine the imported JSON file.
        sft_args_path = os.path.join(ckpt_dir, 'sft_args.json')
        export_args_path = os.path.join(ckpt_dir, 'export_args.json')  # for megatron
        from_sft_args = os.path.exists(sft_args_path)
        if from_sft_args:
            args_path = sft_args_path
        elif os.path.exists(export_args_path):
            args_path = export_args_path
        else:
            logger.warning(f'{sft_args_path} not found')
            return
        with open(args_path, 'r', encoding='utf-8') as f:
            old_args = json.load(f)
        # Determine the keys that need to be imported.
        imported_keys = [
            'model_type', 'model_revision', 'template_type', 'dtype', 'quant_method', 'quantization_bit',
            'bnb_4bit_comp_dtype', 'bnb_4bit_quant_type', 'bnb_4bit_use_double_quant', 'model_id_or_path',
            'custom_register_path', 'custom_dataset_info'
        ]
        if (isinstance(self, SftArguments) and self.train_backend == 'megatron'
                or isinstance(self, ExportArguments) and self.to_hf is True):
            # megatron
            imported_keys += ['tp', 'pp']
        if isinstance(self, InferArguments):
            imported_keys += ['sft_type', 'rope_scaling', 'system']
            if getattr(self, 'load_dataset_config', False) and from_sft_args:
                imported_keys += [
                    'dataset', 'val_dataset', 'dataset_seed', 'val_dataset_ratio', 'check_dataset_strategy',
                    'self_cognition_sample', 'model_name', 'model_author', 'train_dataset_sample', 'val_dataset_sample'
                ]
        # read settings
        for key in imported_keys:
            if not hasattr(self, key):
                continue
            old_value = old_args.get(key)
            if old_value is None:
                continue
            value = getattr(self, key)
            if key in {'dataset', 'val_dataset'} and len(value) > 0:
                continue
            if key in {
                    'system', 'quant_method', 'model_id_or_path', 'custom_register_path', 'custom_dataset_info',
                    'dataset_seed'
            } and value is not None:
                continue
            if key in {'template_type', 'dtype'} and value != 'auto':
                continue
            setattr(self, key, old_value)
