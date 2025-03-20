# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Literal, Optional, Union

import json

from swift.hub import get_hub
from swift.llm import Processor, Template, get_model_tokenizer, get_template, load_by_unsloth, safe_snapshot_download
from swift.llm.utils import get_ckpt_dir
from swift.plugin import extra_tuners
from swift.utils import (check_json_format, get_dist_setting, get_logger, import_external_file, is_dist, is_master,
                         set_device, use_hf_hub)
from .data_args import DataArguments
from .generation_args import GenerationArguments
from .model_args import ModelArguments
from .quant_args import QuantizeArguments
from .template_args import TemplateArguments

logger = get_logger()


def get_supported_tuners():
    return {'lora', 'full', 'longlora', 'adalora', 'llamapro', 'adapter', 'vera', 'boft', 'fourierft', 'reft', 'bone'
            } | set(extra_tuners.keys())


@dataclass
class CompatArguments:
    #
    ckpt_dir: Optional[str] = None
    load_dataset_config: Optional[bool] = None
    lora_modules: List[str] = field(default_factory=list)

    def _handle_ckpt_dir(self: 'BaseArguments'):
        assert os.path.isdir(self.ckpt_dir), f'self.ckpt_dir: {self.ckpt_dir}'
        if (os.path.exists(os.path.join(self.ckpt_dir, 'adapter_config.json'))
                or os.path.exists(os.path.join(self.ckpt_dir, 'default', 'adapter_config.json'))
                or os.path.exists(os.path.join(self.ckpt_dir, 'reft'))):
            if self.ckpt_dir in self.adapters:
                return
            self.adapters.insert(0, self.ckpt_dir)
        else:
            self.model = self.ckpt_dir
        self.ckpt_dir = None
        logger.warning('The `--ckpt_dir` parameter will be removed in `ms-swift>=3.4`. '
                       'Please use `--model`, `--adapters`.')

    def __post_init__(self: 'BaseArguments'):
        if self.ckpt_dir is not None:
            self._handle_ckpt_dir()

        if len(self.lora_modules) > 0:
            self.adapters += self.lora_modules
            logger.warning('The `--lora_modules` parameter will be removed in `ms-swift>=3.4`. '
                           'Please use `--adapters`.')


@dataclass
class BaseArguments(CompatArguments, GenerationArguments, QuantizeArguments, DataArguments, TemplateArguments,
                    ModelArguments):
    """
    BaseArguments class is a dataclass that inherits from multiple argument classes:
    GenerationArguments, QuantizeArguments, DataArguments, TemplateArguments, ModelArguments.

    Args:
        tuner_backend(str): Support peft or unsloth.
        train_type(str): The training type, support all supported tuners and `full`.
        seed (int): Random seed for reproducibility. Default is 42.
        model_kwargs (Optional[str]): Additional keyword arguments for the model. Default is None.
        load_data_args (bool): Flag to determine if dataset configuration should be loaded. Default is False.
        use_hf (bool): Flag to determine if Hugging Face should be used. Default is False.
        hub_token (Optional[str]): SDK token for authentication. Default is None.
        custom_register_path (List[str]): Path to custom .py file for dataset registration. Default is None.
        ignore_args_error (bool): Flag to ignore argument errors for notebook compatibility. Default is False.
        use_swift_lora (bool): Use swift lora, a compatible argument
    """
    tuner_backend: Literal['peft', 'unsloth'] = 'peft'
    train_type: str = field(default='lora', metadata={'help': f'train_type choices: {list(get_supported_tuners())}'})
    adapters: List[str] = field(default_factory=list)
    external_plugins: List[str] = field(default_factory=list)

    seed: int = 42
    model_kwargs: Optional[Union[dict, str]] = None
    load_args: bool = True
    load_data_args: bool = False

    use_hf: bool = False
    # None: use env var `MODELSCOPE_API_TOKEN`
    hub_token: Optional[str] = field(
        default=None, metadata={'help': 'SDK token can be found in https://modelscope.cn/my/myaccesstoken'})
    custom_register_path: List[str] = field(default_factory=list)  # .py

    # extra
    ignore_args_error: bool = False  # True: notebook compatibility
    use_swift_lora: bool = False  # True for using tuner_backend == swift, don't specify this unless you know what you are doing # noqa

    def _prepare_training_args(self, training_args: Dict[str, Any]) -> None:
        pass

    def _init_custom_register(self) -> None:
        """Register custom .py file to datasets"""
        if isinstance(self.custom_register_path, str):
            self.custom_register_path = [self.custom_register_path]
        if not self.custom_register_path:
            return
        for path in self.custom_register_path:
            import_external_file(path)
        logger.info(f'Successfully registered {self.custom_register_path}.')

    def _import_external_plugins(self):
        if isinstance(self.external_plugins, str):
            self.external_plugins = [self.external_plugins]
        if not self.external_plugins:
            return
        for external_plugin in self.external_plugins:
            import_external_file(external_plugin)
        logger.info(f'Successfully imported external_plugins: {self.external_plugins}.')

    @staticmethod
    def _check_is_adapter(adapter_dir: str) -> bool:
        if (os.path.exists(os.path.join(adapter_dir, 'adapter_config.json'))
                or os.path.exists(os.path.join(adapter_dir, 'default', 'adapter_config.json'))
                or os.path.exists(os.path.join(adapter_dir, 'reft'))):
            return True
        return False

    def _init_adapters(self):
        if isinstance(self.adapters, str):
            self.adapters = [self.adapters]
        self.adapters = [
            safe_snapshot_download(adapter, use_hf=self.use_hf, hub_token=self.hub_token) for adapter in self.adapters
        ]

    def __post_init__(self):
        if self.use_hf or use_hf_hub():
            self.use_hf = True
            os.environ['USE_HF'] = '1'
        CompatArguments.__post_init__(self)
        self._init_adapters()
        self._init_ckpt_dir()
        self._init_custom_register()
        self._import_external_plugins()
        self._init_model_kwargs()
        # The Seq2SeqTrainingArguments has a property called world_size, which cannot be assigned a value.
        self.rank, self.local_rank, self.global_world_size, self.local_world_size = get_dist_setting()
        logger.info(f'rank: {self.rank}, local_rank: {self.local_rank}, '
                    f'world_size: {self.global_world_size}, local_world_size: {self.local_world_size}')
        if self.train_type not in extra_tuners:
            for adapter in self.adapters:
                assert self._check_is_adapter(adapter), (
                    f'`{adapter}` is not an adapter, please try using `--model` to pass it.')
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

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str):
        self = super().__new__(cls)
        self.load_data_args = True
        self.ckpt_dir = checkpoint_dir
        self.load_args_from_ckpt()
        all_keys = list(f.name for f in fields(BaseArguments))
        for key in all_keys:
            if not hasattr(self, key):
                setattr(self, key, None)
        return self

    def _init_ckpt_dir(self, adapters=None):
        # compat megatron
        model = self.model or getattr(self, 'mcore_model', None) or getattr(self, 'load', None)
        self.ckpt_dir = get_ckpt_dir(model, adapters or self.adapters)
        if self.ckpt_dir and self.load_args:
            self.load_args_from_ckpt()

    def load_args_from_ckpt(self) -> None:
        from ..train_args import TrainArguments
        args_path = os.path.join(self.ckpt_dir, 'args.json')
        assert os.path.exists(args_path), f'args_path: {args_path}'
        with open(args_path, 'r', encoding='utf-8') as f:
            old_args = json.load(f)
        all_keys = list(f.name for f in fields(BaseArguments))
        data_keys = list(f.name for f in fields(DataArguments))
        load_keys = [
            # quant_args
            'bnb_4bit_quant_type',
            'bnb_4bit_use_double_quant',
            # base_args
            'train_type',
            'tuner_backend',
            'use_swift_lora',
            # data_args
            'model_name',
            'model_author',
            'split_dataset_ratio',
            # template_args
            'tools_prompt',
            'use_chat_template',
        ]
        skip_keys = list(f.name for f in fields(GenerationArguments) + fields(CompatArguments)) + ['adapters']
        if not isinstance(self, TrainArguments):
            skip_keys += ['max_length']
        all_keys = set(all_keys) - set(skip_keys)
        for key, old_value in old_args.items():
            if key not in all_keys or old_value is None:
                continue
            if not self.load_data_args and key in data_keys:
                continue
            value = getattr(self, key, None)
            if value is None or isinstance(value, (list, tuple)) and len(value) == 0 or key in load_keys:
                setattr(self, key, old_value)
        logger.info(f'Successfully loaded {args_path}.')

    def save_args(self, output_dir=None) -> None:
        if is_master():
            output_dir = output_dir or self.output_dir
            os.makedirs(output_dir, exist_ok=True)
            fpath = os.path.join(output_dir, 'args.json')
            logger.info(f'The {self.__class__.__name__} will be saved in: {fpath}')
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(check_json_format(self.__dict__), f, ensure_ascii=False, indent=2)

    def _init_device(self):
        if is_dist():
            set_device()

    def get_template(self, processor: 'Processor') -> 'Template':
        template_kwargs = self.get_template_kwargs()
        template = get_template(self.template, processor, **template_kwargs)
        logger.info(f'default_system: {template.template_meta.default_system}')
        return template

    def get_model_processor(self,
                            *,
                            model=None,
                            model_type=None,
                            model_revision=None,
                            task_type=None,
                            num_labels=None,
                            **kwargs):
        if self.tuner_backend == 'unsloth':
            return load_by_unsloth(self)
        kwargs.update(self.get_model_kwargs())
        # compat rlhf
        kwargs['model_id_or_path'] = model or self.model
        kwargs['model_type'] = model_type or self.model_type
        kwargs['model_revision'] = model_revision or self.model_revision
        kwargs['task_type'] = task_type or self.task_type
        kwargs['num_labels'] = num_labels or self.num_labels

        return get_model_tokenizer(**kwargs)
