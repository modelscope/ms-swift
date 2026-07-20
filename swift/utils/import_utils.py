# Copyright (c) ModelScope Contributors. All rights reserved.
# Copyright 2023-present the HuggingFace Inc. team.

import importlib.util
import os
from itertools import chain
from types import ModuleType
from typing import Any


def is_vllm_available():
    return importlib.util.find_spec('vllm') is not None


def is_vllm_ascend_available():
    return importlib.util.find_spec('vllm_ascend') is not None


def is_vllm_metax_available():
    return importlib.util.find_spec('vllm_metax') is not None


def is_lmdeploy_available():
    return importlib.util.find_spec('lmdeploy') is not None


def is_liger_available():
    return importlib.util.find_spec('liger_kernel') is not None


def is_swanlab_available():
    return importlib.util.find_spec('swanlab') is not None


def is_megatron_available():
    return importlib.util.find_spec('megatron') is not None


def is_flash_attn_3_available():
    return (importlib.util.find_spec('flash_attn_3') is not None
            and importlib.util.find_spec('flash_attn_interface') is not None)


def is_flash_attn_2_available():
    return importlib.util.find_spec('flash_attn') is not None


def is_unsloth_available() -> bool:
    return importlib.util.find_spec('unsloth') is not None


def is_pyreft_available() -> bool:
    return importlib.util.find_spec('pyreft') is not None


def is_wandb_available() -> bool:
    return importlib.util.find_spec('wandb') is not None


def is_trl_available() -> bool:
    return importlib.util.find_spec('trl') is not None


def patch_trl_package_check() -> None:
    """Fix optional dependency checks in TRL <= 0.28 with Transformers 5."""
    try:
        import trl.import_utils as trl_import_utils
    except ImportError:
        return

    package_check = getattr(trl_import_utils, '_is_package_available', None)
    if package_check is None or not isinstance(package_check('trl'), tuple):
        return

    def compatible_package_check(package, return_version=False):
        result = package_check(package, return_version=return_version)
        if not return_version and isinstance(result, tuple):
            return result[0]
        return result

    trl_import_utils._is_package_available = compatible_package_check


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
