# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2023-present the HuggingFace Inc. team.

import importlib.util
import os
from itertools import chain
from types import ModuleType
from typing import Any

from .logger import get_logger

logger = get_logger()  # pylint: disable=invalid-name


def is_vllm_available():
    return importlib.util.find_spec('vllm') is not None


def is_lmdeploy_available():
    return importlib.util.find_spec('lmdeploy') is not None


def is_liger_available():
    return importlib.util.find_spec('liger_kernel') is not None


def is_xtuner_available():
    return importlib.util.find_spec('xtuner') is not None


def is_megatron_available():
    return importlib.util.find_spec('megatron') is not None


def is_unsloth_available() -> bool:
    return importlib.util.find_spec('unsloth') is not None


def is_pyreft_available() -> bool:
    return importlib.util.find_spec('pyreft') is not None


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = super().__dir__()
        # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f'module {self.__name__} has no attribute {name}')

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        return importlib.import_module('.' + module_name, self.__name__)

    def __reduce__(self):
        return self.__class__, (self._name, self.__file__, self._import_structure)
