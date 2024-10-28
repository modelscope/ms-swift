import os
from dataclasses import dataclass, field, fields
from typing import List, Optional, Union

import json

from swift.utils import check_json_format, get_logger, is_master
from .data_args import DataArguments
from .generation_args import GenerationArguments
from .model_args import ModelArguments
from .quant_args import QuantizeArguments
from .template_args import TemplateArguments

logger = get_logger()


@dataclass
class BaseArguments(ModelArguments, TemplateArguments, QuantizeArguments, GenerationArguments, DataArguments):
    seed: int = 42
    load_args: bool = True
    load_dataset_config: bool = False
    save_safetensors: bool = True
    # None: use env var `MODELSCOPE_API_TOKEN`
    hub_token: Optional[str] = field(
        default=None, metadata={'help': 'SDK token can be found in https://modelscope.cn/my/myaccesstoken'})

    # extra
    gpu_memory_fraction: Optional[float] = None
    ignore_args_error: bool = False  # True: notebook compatibility

    def __post_init__(self):
        if self.load_args:
            self._load_args()

        self._save_args()
        ModelArguments.__post_init__(self)
        TemplateArguments.__post_init__(self)
        DataArguments.__post_init__(self)
        QuantizeArguments.__post_init__(self)
        GenerationArguments.__post_init__(self)
        self.handle_path()
        from swift.hub import default_hub
        if default_hub.try_login(self.hub_token):
            logger.info('hub login successful!')

    @staticmethod
    def to_abspath(path: Union[str, List[str]], check_path_exist: bool = False) -> Union[str, List[str]]:
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
            res.append(BaseArguments.to_abspath(v, check_path_exist))
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
            value = self.to_abspath(value, k in check_exist_path)
            setattr(self, k, value)

    def _load_args(self) -> None:
        """Load specific attributes from sft_args.json"""
        from swift.llm import SftArguments, ExportArguments, InferArguments
        if isinstance(self, SftArguments):
            ckpt_dir = self.resume_from_checkpoint
        else:
            ckpt_dir = self.ckpt_dir
        if ckpt_dir is None:
            return
        # Determine the imported JSON file.
        args_path = os.path.join(ckpt_dir, 'args.json')
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
            value = getattr(self, key)
            old_value = old_args.get(key)  # value in checkpoint
            if old_value and not value:
                # TODO: check;  system=''
                setattr(self, key, old_value)

    def _save_args(self) -> None:
        from swift.llm import InferArguments
        if isinstance(self, InferArguments):
            return
        self.args_type = self.__class__.__name__
        if is_master():
            fpath = os.path.join(self.output_dir, 'args.json')
            logger.info(f'The {args.__class__.__name__} will be saved in: {fpath}')
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(check_json_format(args.__dict__), f, ensure_ascii=False, indent=2)
