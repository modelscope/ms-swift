# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2023-present the HuggingFace Inc. team.

import os
import threading
from dataclasses import asdict, dataclass, field
from types import FunctionType
from typing import Dict, List, Optional

import json
import peft.utils
import torch
from peft.utils import CONFIG_NAME
from peft.utils import ModulesToSaveWrapper as _ModulesToSaveWrapper
from peft.utils import _get_submodules

from swift.hub.snapshot_download import snapshot_download
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

    def __init__(self):
        self._thread_inf: Dict[int, Dict[str, bool]] = {}
        self._unique_thread = bool(
            int(os.environ.get(ActivationMixin.USE_UNIQUE_THREAD, '1')))
        if not self._unique_thread:
            logger.info(
                'Using multiple thread mode, gradient checkpointing is not supported.'
            )

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


class SwiftAdapter:

    @staticmethod
    def prepare_model(model: torch.nn.Module, config: SwiftConfig,
                      adapter_name: str) -> SwiftOutput:
        raise NotImplementedError

    @staticmethod
    def activate_adapter(module: torch.nn.Module, adapter_name: str,
                         activate: bool):
        raise NotImplementedError

    @staticmethod
    def freeze_model():
        return True


class ModulesToSaveWrapper(ActivationMixin, _ModulesToSaveWrapper):

    def __init__(self, *args, **kwargs):
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

    def set_adapter(self, adapter_name: str):
        if adapter_name not in self.modules_to_save:
            raise ValueError(
                f'Adapter {adapter_name} not found in {self.modules_to_save.keys()}'
            )
        self.modules_to_save[adapter_name].requires_grad_(True)
        self.set_activation(adapter_name, True)

    def deactivate_adapter(self, adapter_name: str):
        if adapter_name in self.modules_to_save and self.unique_thread:
            self.modules_to_save[adapter_name].requires_grad_(False)
        self.set_activation(adapter_name, False)


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
