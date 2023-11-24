# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2023-present the HuggingFace Inc. team.

import os
import threading
from dataclasses import asdict, dataclass, field
from types import FunctionType
from typing import Dict

import json
import torch
from peft.utils import CONFIG_NAME

from swift.hub.snapshot_download import snapshot_download
from swift.utils.constants import BIN_EXTENSIONS


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
        self._thread_inf: Dict[int, bool] = {}
        self._unique_thread = bool(
            int(os.environ.get(ActivationMixin.USE_UNIQUE_THREAD, '1')))

    def set_activation(self, activate=True):
        tid = 0 if self._unique_thread else threading.get_ident()
        self._thread_inf[tid] = activate

    def is_activated(self):
        tid = 0 if self._unique_thread else threading.get_ident()
        return self._thread_inf.get(tid, True)


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
