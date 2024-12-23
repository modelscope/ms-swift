# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2023-present the HuggingFace Inc. team.

import hashlib
import os
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, field
from types import FunctionType
from typing import Dict, Optional, Union

import json
import numpy as np
import torch
from modelscope import snapshot_download
from modelscope.hub.utils.utils import get_cache_dir
from packaging import version
from peft.utils import CONFIG_NAME
from peft.utils import ModulesToSaveWrapper as _ModulesToSaveWrapper
from peft.utils import _get_submodules

from swift.llm import MODEL_ARCH_MAPPING, ModelKeys
from swift.utils.constants import BIN_EXTENSIONS
from swift.utils.logger import get_logger

logger = get_logger()


@dataclass
class SwiftConfig:

    swift_type: str = field(default=None)

    model_key_mapping: Optional[Union[dict, ModelKeys]] = field(default=None)

    @property
    def __dict__(self):
        return asdict(self)

    def to_dict(self):
        return self.__dict__

    def save_pretrained(self, save_directory, **kwargs):
        r"""
        This method saves the configuration of your adapter model in a directory.

        Args:
            save_directory (`str`):
                The directory where the configuration will be saved.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f'Provided path ({save_directory}) should be a directory, not a file')

        os.makedirs(save_directory, exist_ok=True)

        output_dict = self.__dict__
        output_dict.update(kwargs)
        output_path = os.path.join(save_directory, CONFIG_NAME)

        # save it
        with open(output_path, 'w') as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        This method loads the configuration of your adapter model from a directory.

        Args:
            pretrained_model_name_or_path (`str`):
                The directory or the hub-id where the configuration is saved.
            **kwargs:
                Additional keyword arguments passed along to the child class initialization.
        """
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, CONFIG_NAME)):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        else:
            try:
                model_dir = snapshot_download(pretrained_model_name_or_path, ignore_patterns=BIN_EXTENSIONS)
                config_file = os.path.join(model_dir, CONFIG_NAME)
            except Exception:
                raise ValueError(f"Can't find config.json at '{pretrained_model_name_or_path}'")

        loaded_attributes = cls.from_json_file(config_file)

        from .mapping import SWIFT_MAPPING
        assert loaded_attributes.get('swift_type', '') in SWIFT_MAPPING
        config = SWIFT_MAPPING[loaded_attributes['swift_type']][0](**kwargs)

        for key, value in loaded_attributes.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    @classmethod
    def from_json_file(cls, path_json_file, **kwargs):
        r"""
        Loads a configuration file from a json file.

        Args:
            path_json_file (`str`):
                The path to the json file.
        """
        with open(path_json_file, 'r') as file:
            json_object = json.load(file)

        return json_object


@dataclass
class SwiftOutput:
    """The output class returned by all tuners.

    Args:
        model (`torch.nn.Module`): The model wrapped
        config (`SwiftConfig`): The swift config instance.
        state_dict_callback (`FunctionType`): A callback returned by the tuner
            which is used to get the tuner's state dict among the model's state dict.
            This callback should receive a state dict, and returns a created state dict.
            Examples:
                >>> def state_dict_callback(state_dict, adapter_name):
                >>>     return {
                >>>         key: value
                >>>         for key, value in state_dict.items() if adapter_name in key
                >>>     }
        save_callback (`FunctionType`): A callback used to save trained model.
        mark_trainable_callback (`FunctionType`): A callback returned by the tuner
            which is used to mark the tuner's adapter's parameters to trainable.
            This callback should receive a model instance, and returns nothing.
            Examples:
                >>> def mark_trainable_callback(model):
                >>>     mark_lora_as_trainable(model, config.bias)
        optimizer_group_callback (`FunctionType`): A callback returned the param group cared by the tuner.
        load_state_dict_callback (`FunctionType`): A callback called before load_state_dict of the tuner.
        load_callback (`FunctionType`): A callback used to load trained model.
    """
    model: torch.nn.Module = None
    config: SwiftConfig = None
    state_dict_callback: FunctionType = None
    save_callback: FunctionType = None
    mark_trainable_callback: FunctionType = None
    optimizer_group_callback: FunctionType = None
    load_state_dict_callback: FunctionType = None
    load_callback: FunctionType = None


class ActivationMixin:

    USE_UNIQUE_THREAD = 'USE_UNIQUE_THREAD'

    REMINEDED = False

    def __init__(self, module_key):
        self.module_key = module_key
        self._thread_inf: Dict[int, Dict[str, bool]] = {}
        self._unique_thread = bool(int(os.environ.get(ActivationMixin.USE_UNIQUE_THREAD, '1')))
        if not self._unique_thread and not ActivationMixin.REMINEDED:
            ActivationMixin.REMINEDED = True
            logger.warn('Using multiple thread mode, gradient checkpointing is not supported.')

    def mark_all_sub_modules_as_plugin(self: torch.nn.Module):
        self.plugin = True
        for name, module in self.named_modules():
            if 'base_layer' not in name:
                module.plugin = True

    @property
    def indent(self):
        return 0 if self.unique_thread else threading.get_ident()

    @property
    def unique_thread(self):
        return self._unique_thread

    def set_activation(self, adapter_name, activate=True):
        tid = self.indent
        if tid not in self._thread_inf:
            self._thread_inf[tid] = {}
        self._thread_inf[tid][adapter_name] = activate

    def is_activated(self, adapter_name):
        tid = self.indent
        return self._thread_inf.get(tid, {}).get(adapter_name, False)

    def get_activated_adapters(self):
        return [key for key, value in self._thread_inf.get(self.indent, {}).items() if value]


class OffloadHelper:

    def __init__(self):
        sub_dir = os.path.join('offload_cache', str(uuid.uuid4().hex))
        self.cache_dir = os.path.join(get_cache_dir(), sub_dir)
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.index = {}

    def __del__(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    @staticmethod
    def offload_weight(weight, weight_name, offload_folder, index=None):
        dtype = None
        if str(weight.dtype) == 'torch.bfloat16':
            weight = weight.view(torch.int16)
            dtype = 'bfloat16'
        array = weight.cpu().numpy()
        tensor_file = os.path.join(offload_folder, f'{weight_name}.dat')
        if index is not None:
            if dtype is None:
                dtype = str(array.dtype)
            index[weight_name] = {'dtype': dtype, 'shape': list(array.shape)}
        if array.ndim == 0:
            array = array[None]
        file_array = np.memmap(tensor_file, dtype=array.dtype, mode='w+', shape=array.shape)
        file_array[:] = array[:]
        file_array.flush()
        return index

    @staticmethod
    def load_offloaded_weight(weight_file, weight_info):
        shape = tuple(weight_info['shape'])
        if shape == ():
            shape = (1, )

        dtype = weight_info['dtype']
        if dtype == 'bfloat16':
            dtype = 'int16'

        weight = np.memmap(weight_file, dtype=dtype, shape=shape, mode='r')

        if len(weight_info['shape']) == 0:
            weight = weight[0]
        weight = torch.tensor(weight)
        if weight_info['dtype'] == 'bfloat16':
            weight = weight.view(torch.bfloat16)

        return weight

    def offload_disk(self, module: torch.nn.Module, adapter_name, module_key):
        key = adapter_name + ':' + module_key
        md5 = hashlib.md5(key.encode('utf-8')).hexdigest()
        sub_folder = os.path.join(self.cache_dir, md5)
        os.makedirs(sub_folder, exist_ok=True)
        state_dict = module.state_dict()
        self.index[md5] = {}
        for key, tensor in state_dict.items():
            OffloadHelper.offload_weight(tensor, key, sub_folder, self.index[md5])

    def load_disk(self, module: torch.nn.Module, adapter_name, module_key):
        key = adapter_name + ':' + module_key
        md5 = hashlib.md5(key.encode('utf-8')).hexdigest()
        sub_folder = os.path.join(self.cache_dir, md5)
        state_dict = {}
        for key, value in self.index[md5].items():
            file = os.path.join(sub_folder, f'{key}.dat')
            state_dict[key] = OffloadHelper.load_offloaded_weight(file, self.index[md5][key])
        if version.parse(torch.__version__) >= version.parse('2.1.0'):
            module.load_state_dict(state_dict, assign=True)
        else:
            for name, _module in module.named_modules():
                if len(list(_module.modules())) > 1:
                    continue

                buffers = {}
                prefix = name if not name else name + '.'
                for sub_name, buffer in _module.named_buffers():
                    buffer_cls = type(buffer)
                    buffers[sub_name] = buffer_cls(state_dict[prefix + sub_name])
                _module._buffers.update(buffers)
                params = {}
                for sub_name, param in _module.named_parameters():
                    param_cls = type(param)
                    params[sub_name] = param_cls(state_dict[prefix + sub_name], requires_grad=param.requires_grad)
                _module._parameters.update(params)
        shutil.rmtree(sub_folder, ignore_errors=True)


class SwiftAdapter:

    offload_helper = OffloadHelper()

    @staticmethod
    def prepare_model(model: torch.nn.Module, config: SwiftConfig, adapter_name: str) -> SwiftOutput:
        raise NotImplementedError

    @staticmethod
    def activate_adapter(module: torch.nn.Module, adapter_name: str, activate: bool, offload: str = None):
        raise NotImplementedError

    @staticmethod
    def save_memory(module: torch.nn.Module, adapter_name: str, module_key: str, activate: bool, offload: str = None):
        if not isinstance(module, torch.nn.Module):
            return
        if activate:
            SwiftAdapter.load(module, adapter_name, module_key)
        else:
            SwiftAdapter.offload(module, adapter_name, module_key, offload=offload)

    @staticmethod
    def offload(module: torch.nn.Module, adapter_name, module_key, offload: str):
        if not offload:
            return
        device = next(iter(module.parameters())).device
        if hasattr(module, 'origin_device') and module.origin_device != str(device):
            return
        module.origin_device = str(device)
        if offload == 'cpu':
            if str(device) != 'cpu':
                module.to('cpu')
        elif offload == 'meta':
            if str(device) != 'meta':
                SwiftAdapter.offload_helper.offload_disk(module, adapter_name=adapter_name, module_key=module_key)
                module.to('meta')
        else:
            raise NotImplementedError
        torch.cuda.empty_cache()

    @staticmethod
    def load(module: torch.nn.Module, adapter_name, module_key):
        device = next(iter(module.parameters())).device
        if not hasattr(module, 'origin_device') or module.origin_device == str(device):
            return
        if str(device) == 'cpu':
            module.to(module.origin_device)
            delattr(module, 'origin_device')
        elif str(device) == 'meta':
            SwiftAdapter.offload_helper.load_disk(module, adapter_name=adapter_name, module_key=module_key)
            module.to(module.origin_device)
            delattr(module, 'origin_device')

    @classmethod
    def get_model_key_mapping(cls, model_type, config) -> ModelKeys:

        if model_type in MODEL_ARCH_MAPPING.keys():
            model_key_mapping = MODEL_ARCH_MAPPING[model_type]
        else:
            model_key_mapping = config.model_key_mapping

        if model_key_mapping is None:
            raise ValueError(f'{model_type} is not defined in MODEL_KEYS_MAPPING, '
                             f'please consider pass the information through the config.model_key_mapping')

        if isinstance(model_key_mapping, dict):
            model_key_mapping: ModelKeys = ModelKeys(**model_key_mapping)
        return model_key_mapping

    @staticmethod
    def state_dict_load_hook(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]):
        pass

    @staticmethod
    def has_additional_modules():
        return True


class ModulesToSaveWrapper(ActivationMixin, _ModulesToSaveWrapper):

    def __init__(self, *args, module_key, **kwargs):
        super(ModulesToSaveWrapper, self).__init__(module_key)
        super(ActivationMixin, self).__init__(*args, **kwargs)
        SwiftAdapter.save_memory(self.original_module, 'original_module', self.module_key, False, offload='cpu')

    @property
    def active_adapter(self):
        active_adapters = self.get_activated_adapters()
        if not active_adapters:
            return None
        elif len(active_adapters) > 1:
            raise ValueError('ModulesToSaveWrapper does not support multiple active adapters')
        return active_adapters[0]

    def set_adapter(self, adapter_name: str, offload: str = None):
        if adapter_name not in self.modules_to_save:
            raise ValueError(f'Adapter {adapter_name} not found in {self.modules_to_save.keys()}')
        self.modules_to_save[adapter_name].requires_grad_(True)
        self.set_activation(adapter_name, True)
        SwiftAdapter.save_memory(self.modules_to_save[adapter_name], adapter_name, self.module_key, True)
        SwiftAdapter.save_memory(self.original_module, 'original_module', self.module_key, False, offload=offload)

    def deactivate_adapter(self, adapter_name: str, offload: str = None):
        if adapter_name in self.modules_to_save and self.unique_thread:
            self.modules_to_save[adapter_name].requires_grad_(False)
        self.set_activation(adapter_name, False)
        SwiftAdapter.save_memory(
            self.modules_to_save[adapter_name], adapter_name, self.module_key, False, offload=offload)
        if not self.get_activated_adapters():
            SwiftAdapter.save_memory(self.original_module, 'original_module', self.module_key, True)

    def enable_adapters(self, enabled: bool):
        super().enable_adapters(enabled)
        if not enabled:
            SwiftAdapter.save_memory(self.original_module, 'original_module', self.module_key, False, offload='meta')
        else:
            SwiftAdapter.save_memory(self.original_module, 'original_module', self.module_key, True)


def set_adapter(model, adapter_name, activate, offload):
    for module in model.modules():
        if isinstance(module, ModulesToSaveWrapper):
            if activate:
                module.set_adapter(adapter_name, offload)
            else:
                module.deactivate_adapter(adapter_name, offload)


def set_trainable(model, adapter_name):
    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        target_module_found = any(key.endswith(target_key) for target_key in model.modules_to_save)
        if target_module_found:
            parent, target, target_name = _get_submodules(model, key)
            if isinstance(target, ModulesToSaveWrapper):
                target.update(adapter_name)
                target.set_adapter(target.active_adapter)
            else:
                new_module = ModulesToSaveWrapper(target, module_key=key, adapter_name=adapter_name)
                new_module.set_adapter(adapter_name)
                setattr(parent, target_name, new_module)


def swift_to_peft_format(ckpt_dir: str, output_dir: str) -> str:
    if 'default' in os.listdir(ckpt_dir):  # swift_backend
        from swift import Swift
        Swift.save_to_peft_format(ckpt_dir, output_dir)
        ckpt_dir = output_dir
        logger.info(f'Converting the swift format checkpoint to peft format, and saving it to: `{output_dir}`')
    else:
        logger.info('The format of the checkpoint is already in peft format.')
    return ckpt_dir
