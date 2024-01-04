# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2023-present the HuggingFace Inc. team.

import hashlib
import os
import shutil
import threading
from dataclasses import asdict, dataclass, field
from types import FunctionType
from typing import Dict

import json
import torch
from peft.utils import CONFIG_NAME
from peft.utils import ModulesToSaveWrapper as _ModulesToSaveWrapper
from peft.utils import _get_submodules

from swift.hub.snapshot_download import snapshot_download
from swift.hub.utils.utils import get_cache_dir
from swift.utils.constants import BIN_EXTENSIONS
from swift.utils.logger import get_logger

logger = get_logger()


@dataclass
class SwiftConfig:

    swift_type: str = field(default=None)

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
            raise AssertionError(
                f'Provided path ({save_directory}) should be a directory, not a file'
            )

        os.makedirs(save_directory, exist_ok=True)

        output_dict = self.__dict__
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
        if os.path.isfile(
                os.path.join(pretrained_model_name_or_path, CONFIG_NAME)):
            config_file = os.path.join(pretrained_model_name_or_path,
                                       CONFIG_NAME)
        else:
            try:
                model_dir = snapshot_download(
                    pretrained_model_name_or_path,
                    ignore_file_pattern=BIN_EXTENSIONS)
                config_file = os.path.join(model_dir, CONFIG_NAME)
            except Exception:
                raise ValueError(
                    f"Can't find config.json at '{pretrained_model_name_or_path}'"
                )

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
        mark_trainable_callback (`FunctionType`): A callback returned by the tuner
            which is used to mark the tuner's adapter's parameters to trainable.
            This callback should receive a model instance, and returns nothing.
            Examples:
                >>> def mark_trainable_callback(model):
                >>>     mark_lora_as_trainable(model, config.bias)
    """

    config: SwiftConfig = None
    state_dict_callback: FunctionType = None
    mark_trainable_callback: FunctionType = None


class ActivationMixin:

    USE_UNIQUE_THREAD = 'USE_UNIQUE_THREAD'

    def __init__(self, module_key):
        self.module_key = module_key
        self.offloads = {}
        self._thread_inf: Dict[int, Dict[str, bool]] = {}
        self._unique_thread = bool(
            int(os.environ.get(ActivationMixin.USE_UNIQUE_THREAD, '1')))
        if not self._unique_thread:
            logger.info(
                'Using multiple thread mode, gradient checkpointing is not supported.'
            )

    def add_offload(self, adapter_name: str, offload=None):
        self.offloads[adapter_name] = offload

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
        return [
            key
            for key, value in self._thread_inf.get(self.indent, {}).items()
            if value
        ]


class OffloadHelper:

    sub_dir = 'offload_cache'
    cache_dir = os.path.join(get_cache_dir(), sub_dir)
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def read_safe_tensors(safe_tensor_file):
        if os.path.exists(safe_tensor_file):
            from safetensors.torch import load_file as safe_load_file
            return safe_load_file(
                safe_tensor_file,
                device='cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def write_safe_tensors(state_dict, safe_tensor_file):
        from safetensors.torch import save_file as safe_save_file
        safe_save_file(state_dict, safe_tensor_file, metadata={'format': 'pt'})

    @staticmethod
    def offload_weight(weight, weight_name, offload_folder, index=None):
        dtype = None
        # Check the string instead of the dtype to be compatible with versions of PyTorch that don't have bfloat16.
        if str(weight.dtype) == "torch.bfloat16":
            # Need to reinterpret the underlined data as int16 since NumPy does not handle bfloat16s.
            weight = weight.view(torch.int16)
            dtype = "bfloat16"
        array = weight.cpu().numpy()
        tensor_file = os.path.join(offload_folder, f"{weight_name}.dat")
        if index is not None:
            if dtype is None:
                dtype = str(array.dtype)
            index[weight_name] = {"dtype": dtype, "shape": list(array.shape)}
        if array.ndim == 0:
            array = array[None]
        file_array = np.memmap(tensor_file, dtype=array.dtype, mode="w+", shape=array.shape)
        file_array[:] = array[:]
        file_array.flush()
        return index

    @staticmethod
    def load_offloaded_weight(weight_file, weight_info):
        shape = tuple(weight_info["shape"])
        if shape == ():
            # NumPy memory-mapped arrays can't have 0 dims so it was saved as 1d tensor
            shape = (1,)

        dtype = weight_info["dtype"]
        if dtype == "bfloat16":
            # NumPy does not support bfloat16 so this was saved as a int16
            dtype = "int16"

        weight = np.memmap(weight_file, dtype=dtype, shape=shape, mode="r")

        if len(weight_info["shape"]) == 0:
            weight = weight[0]
        weight = torch.tensor(weight)
        if weight_info["dtype"] == "bfloat16":
            weight = weight.view(torch.bfloat16)

        return weight

    @staticmethod
    def offload_disk(module: torch.nn.Module, adapter_name, module_key):
        key = adapter_name + ':' + module_key
        md5 = hashlib.md5(key.encode('utf-8')).hexdigest()
        file = os.path.join(OffloadHelper.cache_dir, md5 + '.safetensors')
        OffloadHelper.write_safe_tensors(module.state_dict(), file)

    @staticmethod
    def load_disk(module: torch.nn.Module, adapter_name, module_key):
        key = adapter_name + ':' + module_key
        md5 = hashlib.md5(key.encode('utf-8')).hexdigest()
        file = os.path.join(OffloadHelper.cache_dir, md5 + '.safetensors')
        state_dict = OffloadHelper.read_safe_tensors(file)
        print(module.load_state_dict(state_dict, assign=True))
        shutil.rmtree(file, ignore_errors=True)
        try:
            print('here1!!!')
            module.to(module.origin_device)
            print('here2!!!')
        except:
            print()


class SwiftAdapter:

    @staticmethod
    def prepare_model(model: torch.nn.Module, config: SwiftConfig,
                      adapter_name: str) -> SwiftOutput:
        raise NotImplementedError

    @staticmethod
    def activate_adapter(module: torch.nn.Module,
                         adapter_name: str,
                         activate: bool,
                         offload: str = None):
        raise NotImplementedError

    @staticmethod
    def save_memory(module: torch.nn.Module,
                    adapter_name: str,
                    module_key: str,
                    activate: bool,
                    offload: str = None):
        if offload is not None:
            if activate:
                SwiftAdapter.load(
                    module, adapter_name, module_key, offload=offload)
            else:
                SwiftAdapter.offload(
                    module, adapter_name, module_key, offload=offload)

    @staticmethod
    def offload(module: torch.nn.Module, adapter_name, module_key,
                offload: str):
        device = next(iter(module.parameters())).device
        if hasattr(module, 'origin_device') and module.origin_device != str(device):
            return
        module.origin_device = str(device)
        if offload == 'cpu':
            if str(device) != 'cpu':
                module.to('cpu')
        if offload == 'meta':
            if str(device) != 'meta':
                OffloadHelper.offload_disk(
                    module, adapter_name=adapter_name, module_key=module_key)
                module.to('meta')
        else:
            raise NotImplementedError

    @staticmethod
    def load(module: torch.nn.Module, adapter_name, module_key, offload: str):
        device = next(iter(module.parameters())).device
        if not hasattr(module, 'origin_device') or module.origin_device == str(device):
            return
        if offload == 'cpu':
            module.to(module.origin_device)
            delattr(module, 'origin_device')
        elif offload == 'meta':
            OffloadHelper.load_disk(
                module, adapter_name=adapter_name, module_key=module_key)
            try:
                module.to(module.origin_device)
            except:
                print()
            delattr(module, 'origin_device')
        else:
            raise NotImplementedError

    @staticmethod
    def freeze_model():
        return True


class ModulesToSaveWrapper(ActivationMixin, _ModulesToSaveWrapper):

    def __init__(self, *args, module_key, **kwargs):
        self.module_key = module_key
        self.offloads = {}
        super(ModulesToSaveWrapper, self).__init__()
        super(ActivationMixin, self).__init__(*args, **kwargs)

    @property
    def active_adapter(self):
        active_adapters = self.get_activated_adapters()
        if not active_adapters:
            return None
        elif len(active_adapters) > 1:
            raise ValueError(
                'ModulesToSaveWrapper does not support multiple active adapters'
            )
        return active_adapters[0]

    def add_offload(self, adapter_name: str, offload=None):
        self.offloads[adapter_name] = offload

    def set_adapter(self, adapter_name: str):
        if adapter_name not in self.modules_to_save:
            raise ValueError(
                f'Adapter {adapter_name} not found in {self.modules_to_save.keys()}'
            )
        self.modules_to_save[adapter_name].requires_grad_(True)
        self.set_activation(adapter_name, True)
        SwiftAdapter.save_memory(
            self.modules_to_save[adapter_name],
            adapter_name,
            self.module_key,
            True,
            offload=self.offloads.get(adapter_name))

    def deactivate_adapter(self, adapter_name: str):
        if adapter_name in self.modules_to_save and self.unique_thread:
            self.modules_to_save[adapter_name].requires_grad_(False)
        self.set_activation(adapter_name, False)
        SwiftAdapter.save_memory(
            self.modules_to_save[adapter_name],
            adapter_name,
            self.module_key,
            False,
            offload=self.offloads.get(adapter_name))


def set_adapter(model, adapter_name, activate):
    for module in model.modules():
        if isinstance(module, ModulesToSaveWrapper):
            if activate:
                module.set_adapter(adapter_name)
            else:
                module.deactivate_adapter(adapter_name)


def set_trainable(model, adapter_name):
    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        target_module_found = any(
            key.endswith(target_key) for target_key in model.modules_to_save)
        if target_module_found:
            parent, target, target_name = _get_submodules(model, key)
            if isinstance(target, ModulesToSaveWrapper):
                target.update(adapter_name)
                target.set_adapter(target.active_adapter)
            else:
                new_module = ModulesToSaveWrapper(target, adapter_name)
                new_module.set_adapter(adapter_name)
                setattr(parent, target_name, new_module)
