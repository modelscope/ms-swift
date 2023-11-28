# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2023-present the HuggingFace Inc. team.
import inspect
import os
import re
from inspect import Parameter, Signature, signature
from types import MethodType
from typing import Dict, List, Union

import json
import torch
from peft.utils import CONFIG_NAME
from peft.utils.other import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME
from torch import nn

from swift.hub.snapshot_download import snapshot_download
from swift.utils.constants import (DEFAULT_ADAPTER, PEFT_TYPE_KEY,
                                   SWIFT_TYPE_KEY)
from swift.utils.logger import get_logger
from .. import PeftConfig, PeftModel, get_peft_model
from .utils import SwiftConfig

logger = get_logger()


class SwiftModel(nn.Module):
    """The Swift wrapper model.

    Args:
        model (`Union[nn.Module, 'SwiftModel']`) A module to be tuned by Swift.
        config (`Union[SwiftConfig, Dict[str, SwiftConfig]]`) A config or a dict of {adapter_name: SwiftConfig}.
            If it's a config class, the adapter_name will be `default`
        extra_state_keys (`List[str]`, `optional`) A list of regex to match the extra state keys to be saved.
        inference_mode (bool, `optional`): Load model at inference mode, default False.
    """

    def __init__(self,
                 model: Union[nn.Module, 'SwiftModel'],
                 config: Union[SwiftConfig, Dict[str, SwiftConfig]],
                 extra_state_keys: List[str] = None,
                 inference_mode: bool = False,
                 **kwargs):
        super().__init__()
        self.adapters = {}
        if isinstance(model, SwiftModel):
            self.adapters = model.adapters
            extra_state_keys = extra_state_keys or []
            extra_state_keys.extend(model.extra_state_keys)
            model = model.base_model

        if (getattr(model, 'hf_device_map', None) is not None) and (
                len(set(model.hf_device_map.values()) & {'cpu', 'disk'}) > 0):
            from accelerate.hooks import remove_hook_from_submodules
            remove_hook_from_submodules(model)

        if isinstance(config, SwiftConfig):
            self.adapters[DEFAULT_ADAPTER] = self._prepare_model(
                model, config, DEFAULT_ADAPTER)
        elif isinstance(config, dict):
            assert (all(isinstance(c, SwiftConfig) for c in config.values()))
            for adapter_name, _config in config.items():
                self.adapters[adapter_name] = self._prepare_model(
                    model, _config, adapter_name)
        self.model = model

        self.extra_state_keys = extra_state_keys or []
        if (getattr(self, 'hf_device_map', None) is not None) and (
                len(set(self.hf_device_map.values()) & {'cpu', 'disk'}) > 0):
            from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
            from accelerate import dispatch_model, infer_auto_device_map
            from accelerate.utils import get_balanced_memory
            device_map = kwargs.get('device_map', 'auto')
            max_memory = kwargs.get('max_memory', None)
            offload_dir = kwargs.get('offload_folder', None)
            offload_index = kwargs.get('offload_index', None)

            dispatch_model_kwargs = {}
            # Safety checker for previous `accelerate` versions
            # `offload_index` was introduced in https://github.com/huggingface/accelerate/pull/873/
            if 'offload_index' in inspect.signature(dispatch_model).parameters:
                dispatch_model_kwargs['offload_index'] = offload_index

            no_split_module_classes = model._no_split_modules

            if device_map != 'sequential':
                max_memory = get_balanced_memory(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == 'balanced_low_0'),
                )
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes)
            dispatch_model(
                model,
                device_map=device_map,
                offload_dir=offload_dir,
                **dispatch_model_kwargs,
            )
            hook = AlignDevicesHook(io_same_device=True)
            add_hook_to_module(model, hook)

        def forward(self, *args, **kwargs):
            return self.base_model(*args, **kwargs)

        _parameters = [Parameter('self', Parameter.POSITIONAL_ONLY)]
        _parameters += list(
            signature(self.base_model.forward).parameters.values())
        forward.__signature__ = Signature(_parameters)
        self.forward = MethodType(forward, self)
        for adapter_name in self.adapters:
            self.activate_adapter(adapter_name)

        if inference_mode:
            self.eval()
        else:
            for output in self.adapters.values():
                output.mark_trainable_callback(model)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, False)

    def state_dict(self,
                   *args,
                   destination=None,
                   prefix='',
                   keep_vars=False,
                   adapter_name: str = None,
                   **kwargs):
        """
        Args:
            destination (`dict`, `optional`): If provided, the state of module will
                be updated into the dict and the same object is returned.
                Otherwise, an ``OrderedDict`` will be created and returned.
                Default: ``None``.
            prefix (`str`, `optional`): a prefix added to parameter and buffer
                names to compose the keys in state_dict. Default: ``''``.
            keep_vars (`bool`, `optional`): by default the :class:`~torch.Tensor` s
                returned in the state dict are detached from autograd. If it's
                set to ``True``, detaching will not be performed.
                Default: ``False``.
            adapter_name (`str`, `optional`): The name of the adapter's parameters to be saved,
                `None` input will save all adapters.
            **kwargs:
                save_adapter(`bool`): Save adapters or not, default True
                save_extra_states(`bool`): Save extra states or not, default True
        Returns:
            The state dict to be saved.
        """
        destination = self.model.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars)
        state_dicts = {}
        if kwargs.get('save_adapter', True):
            for name, output in self.adapters.items():
                if adapter_name == name or adapter_name is None:
                    state_dicts.update(
                        output.state_dict_callback(destination, name))
        if kwargs.get('save_extra_states', True):
            state_dicts.update({
                k: v
                for k, v in destination.items() if any(
                    re.fullmatch(extra_key, k)
                    for extra_key in self.extra_state_keys)
            })
        return state_dicts

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_model, name)

    @staticmethod
    def load_state_file(path):
        """Load a state dict file by the input path.

        Args:
            path: The local dir to load the state file.

        Returns:
            The state dict.
        """
        if os.path.exists(os.path.join(path, SAFETENSORS_WEIGHTS_NAME)):
            filename = os.path.join(path, SAFETENSORS_WEIGHTS_NAME)
            from safetensors.torch import load_file as safe_load_file
            return safe_load_file(
                filename,
                device='cuda' if torch.cuda.is_available() else 'cpu')
        elif os.path.exists(os.path.join(path, WEIGHTS_NAME)):
            filename = os.path.join(path, WEIGHTS_NAME)
            return torch.load(
                filename,
                map_location=torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu'))
        return None

    @classmethod
    def from_pretrained(cls,
                        model: Union[nn.Module, 'SwiftModel'],
                        model_id: str = None,
                        adapter_name: Union[str, List[str]] = None,
                        inference_mode: bool = False,
                        revision: str = None,
                        **kwargs):
        """Load a set of tuners and corresponding weights by a model_id.

        Args:
            model (`Union[torch.nn.Module, 'SwiftModel']`): The model to be tuned,
                if the model is already a `SwiftModel` it will be un-wrapped and re-wrapped..
            model_id (`str`): The model_id or a local model dir of tuners to use to tune the model.
            adapter_name (`Union[str, List[str]]`): The adapter_names saved in the model repo to load.
                Default `None`, means load all tuners saved in the model_id
            inference_mode (`bool`): Use in the inference mode or not.
            revision (`str`): The model revision to use.
            **kwargs:
                extra_state_keys (`List[str]`, `optional`) A list of regex to match the extra state keys to be saved.
                Other parameters will be passed to the device_map.
        Returns:
            The `SwiftModel` instance.
        """
        adapters = {}
        model_dir = model_id
        extra_state_keys = kwargs.pop('extra_state_keys', None)
        config_file = os.path.join(model_dir, CONFIG_NAME)
        if extra_state_keys is None and os.path.isfile(config_file):
            with open(config_file, 'r') as file:
                _json = json.load(file)
                extra_state_keys = _json.get('extra_state_keys')
        if os.path.isfile(model_dir):
            raise ValueError(
                f'Please pass in a local dir or a model id, not a local file: {model_id}'
            )
        if not os.path.exists(model_id):
            model_dir = snapshot_download(model_id, revision=revision)
        if adapter_name is None:
            adapter_name = [
                sub_dir for sub_dir in os.listdir(model_dir)
                if os.path.isdir(os.path.join(model_dir, sub_dir)) and
                os.path.isfile(os.path.join(model_dir, sub_dir, CONFIG_NAME))
            ]
        for _name in adapter_name if isinstance(adapter_name,
                                                list) else [adapter_name]:
            sub_folder = os.path.join(model_dir, _name)
            config_file = os.path.join(sub_folder, CONFIG_NAME)

            if not os.path.isfile(config_file):
                logger.warning(f'{_name} is not a valid tuner')
                continue

            with open(config_file, 'r') as file:
                json_object = json.load(file)

            if SWIFT_TYPE_KEY not in json_object:
                raise ValueError('Mixed using with peft is not allowed now.')
            else:
                adapters[_name] = SwiftConfig.from_pretrained(sub_folder)

        self = SwiftModel(model, adapters, extra_state_keys, inference_mode,
                          **kwargs)
        for _name in adapter_name if isinstance(adapter_name,
                                                list) else [adapter_name]:
            sub_folder = os.path.join(model_dir, _name)
            state_dict = cls.load_state_file(sub_folder)
            if state_dict is not None:
                model_is_qlora = len([
                    k for k in self.state_dict().keys()
                    if k.endswith('.lora_A.default.weight')
                    or k.endswith('.lora_B.default.weight')
                ])
                if not model_is_qlora:
                    # model is lora, state_dict: qlora->lora
                    state_dict = {
                        k[:-len('.default.weight') if k.
                          endswith('.lora_A.default.weight') or k.
                          endswith('.lora_B.default.weight') else None]: v
                        for k, v in state_dict.items()
                    }
                self.model.load_state_dict(state_dict, strict=False)
        state_dict = cls.load_state_file(model_dir)
        if state_dict is not None:
            self.model.load_state_dict(state_dict, strict=False)
        return self

    @classmethod
    def _prepare_model(
        cls,
        model: nn.Module,
        config: SwiftConfig,
        adapter_name: str,
    ):
        assert (hasattr(config, SWIFT_TYPE_KEY))
        from .mapping import SWIFT_MAPPING

        adatper_cls = SWIFT_MAPPING[config.swift_type][1]
        if adatper_cls.freeze_model() and not getattr(model, 'model_freezed',
                                                      False):
            for _, p in model.named_parameters():
                p.requires_grad = False
            model.model_freezed = True
        return adatper_cls.prepare_model(model, config, adapter_name)

    def create_or_update_model_card(self, output_dir: str):
        """
        Updates or create the model card.
        """
        if not os.path.exists(os.path.join(output_dir, 'README.md')):
            lines = []
        else:
            with open(os.path.join(output_dir, 'README.md'), 'r') as f:
                lines = f.readlines()

        quantization_config = None
        if hasattr(self.base_model, 'config') and hasattr(
                self.base_model.config, 'quantization_config'):
            quantization_config = self.base_model.config.quantization_config.to_dict(
            )
        training_config_text = ''
        # Adds quantization information if it was used
        if quantization_config is not None:
            training_config_text += '\nThe following `bitsandbytes` quantization config was used during training:\n'
            training_config_text += '\n'.join([
                f'- {name}: {value}'
                for name, value in quantization_config.items()
            ])
            training_config_text += '\n'

        training_procedure_heading = '## Training procedure\n'
        if training_procedure_heading in lines:
            lines.insert(
                lines.index(training_procedure_heading) + 2,
                training_config_text)
        else:
            lines.append(
                f'{training_procedure_heading}\n{training_config_text}')

        framework_block_heading = '### Framework versions\n'
        from swift.version import __version__
        if framework_block_heading in lines:
            lines.insert(
                lines.index(framework_block_heading) + 2,
                f'- SWIFT {__version__}\n')
        else:
            lines.append(
                f'{framework_block_heading}\n\n- SWIFT {__version__}\n')

        base_model_heading = '### Base model information\n'
        lines.append(
            f'{base_model_heading}\n\n- BaseModel Class {self.base_model.__class__.__name__}\n'
        )

        # write the lines back to README.md
        with open(os.path.join(output_dir, 'README.md'), 'w') as f:
            f.writelines(lines)

    def save_pretrained(self,
                        save_directory: str,
                        safe_serialization: bool = False,
                        adapter_name: Union[str, List[str]] = None,
                        **kwargs):
        """Save the adapters to a local directory.

        Args:
            save_directory (`str`): The directory to use.
            safe_serialization (`bool`): Use safe tensors to save the weights, default False.
            adapter_name(`Union[str, List[str]]`): The adapters to be saved, default is `None` to save all.
        """
        if os.path.isfile(save_directory):
            raise ValueError(
                f'Provided path ({save_directory}) should be a directory, not a file'
            )
        os.makedirs(save_directory, exist_ok=True)
        self.create_or_update_model_card(save_directory)

        adapter_names = adapter_name if isinstance(
            adapter_name, list) or adapter_name is None else [adapter_name]
        for adapter_name, output in self.adapters.items():
            if adapter_names is not None and adapter_name not in adapter_names:
                continue

            # save only the trainable weights
            output_state_dict = self.state_dict(
                adapter_name=adapter_name, save_extra_states=False)
            output_dir = os.path.join(save_directory, adapter_name)
            os.makedirs(output_dir, exist_ok=True)

            if safe_serialization:
                from safetensors.torch import save_file as safe_save_file
                safe_save_file(
                    output_state_dict,
                    os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME),
                    metadata={'format': 'pt'})
            else:
                torch.save(output_state_dict,
                           os.path.join(output_dir, WEIGHTS_NAME))
            output.config.save_pretrained(output_dir)

        output_state_dict = self.state_dict(
            save_extra_states=True, save_adapter=False)
        if len(output_state_dict) > 0:
            if safe_serialization:
                from safetensors.torch import save_file as safe_save_file
                safe_save_file(
                    output_state_dict,
                    os.path.join(save_directory, SAFETENSORS_WEIGHTS_NAME),
                    metadata={'format': 'pt'})
            else:
                torch.save(output_state_dict,
                           os.path.join(save_directory, WEIGHTS_NAME))
            with open(os.path.join(save_directory, CONFIG_NAME), 'w') as file:
                json.dump({'extra_state_keys': self.extra_state_keys}, file)

        if not os.path.exists(
                os.path.join(save_directory, 'configuration.json')):
            with open(os.path.join(save_directory, 'configuration.json'),
                      'w') as f:
                f.write('{}')

    @property
    def base_model(self):
        return self.model

    def set_active_adapters(self, adapter_names: Union[List[str], str]):
        if not adapter_names:
            return

        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        adapter_names = set(adapter_names)
        for adapter_name in (adapter_names & set(self.adapters.keys())):
            self.activate_adapter(adapter_name)

        for adapter_name in (set(self.adapters.keys()) - adapter_names):
            self.deactivate_adapter(adapter_name)

    def activate_adapter(self, adapter_name):
        if adapter_name not in self.adapters:
            logger.warning(
                f'{adapter_name} not in adapters: {self.adapters.keys()}')
            return

        from .mapping import SWIFT_MAPPING
        SWIFT_MAPPING[self.adapters[adapter_name].config.swift_type][1]\
            .activate_adapter(self.base_model, adapter_name, True)

    def deactivate_adapter(self, adapter_name):
        if adapter_name not in self.adapters:
            logger.warning(
                f'{adapter_name} not in adapters: {self.adapters.keys()}')
            return

        from .mapping import SWIFT_MAPPING
        SWIFT_MAPPING[self.adapters[adapter_name].config.swift_type][1]\
            .activate_adapter(self.base_model, adapter_name, False)

    def get_trainable_parameters(self):
        """
        Get the content of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.base_model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, 'ds_numel'):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return f'trainable params: {trainable_params:,d} || all params: {all_param:,d} ' \
               f'|| trainable%: {100 * trainable_params / all_param:.4f}' \
               '|| cuda memory: ' \
               f'{sum([torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())])/1024/1024/1024:.2f}' \
               'GB.'


class Swift:
    """The Wrapper to use both Peft and Swift tuners."""

    @staticmethod
    def prepare_model(model: Union[nn.Module, SwiftModel],
                      config: Union[SwiftConfig, PeftConfig,
                                    Dict[str, SwiftConfig]], **kwargs):
        """Prepare a model by the input config.

        Args:
            model(`Union[nn.Module, 'SwiftModel']`): The model to be tuned.
            config(`Union[SwiftConfig, PeftConfig, Dict[str, SwiftConfig]]`): The config or config dict, can be either
                SwiftConfigs or PeftConfigs
            **kwargs:
                Extra kwargs needed by SwiftModel or PeftModel.
        Returns:
            The model wrapped by SwiftModel or PeftModel.
        """

        if isinstance(config, (SwiftConfig, dict)):
            return SwiftModel(model, config, **kwargs)
        elif isinstance(
                config,
                PeftConfig) or config.__class__.__name__ == 'PeftWrapper':
            return get_peft_model(model, config, **kwargs)
        raise ValueError(f'Unsupported swift config type: {config.__class__}')

    @staticmethod
    def merge_and_unload(model: Union[PeftModel, SwiftModel], **kwargs):
        """Merge tuners into the base model and unload them.

        Args:
            model(`Union[PeftModel, SwiftModel]`): The model instance with tuners
            kwargs:
                adapter_name(`Union[str, List[str]]`): The adapter_name to unload, only supported in swift tuners.

        """
        from peft import PeftModel as _PeftModel
        if isinstance(model, _PeftModel):
            model.merge_and_unload()
        elif isinstance(model, SwiftModel):
            from swift import LoRAConfig
            from swift.tuners import LoRA
            adapter_name = kwargs.get('adapter_name', None)
            if isinstance(adapter_name, str):
                adapter_name = [adapter_name]
            for adapter, output in model.adapters.items():
                if isinstance(output.config,
                              LoRAConfig) and (adapter_name is None
                                               or adapter in adapter_name):
                    LoRA.unpatch_lora(model, output.config, adapter)

    @staticmethod
    def from_pretrained(model: Union[nn.Module, SwiftModel],
                        model_id: str = None,
                        adapter_name: Union[str, List[str]] = None,
                        revision: str = None,
                        **kwargs):
        """Prepare a model by a model_id in the ModelScope hub or a local dir.

        Args:
            model(`Union[nn.Module, 'SwiftModel']`): The model to be tuned.
            model_id(`str`): The model id of the modelhub or a local dir containing the configs/weights.
            adapter_name(`str`, `optional`): The adapter_name to use.
            revision(`str`, `optional`): The model revision if the model_id is a model id of the modelhub.
            **kwargs:
                Extra kwargs needed by ``SwiftModel.from_pretrained`` or ``PeftModel.from_pretrained``.
        Returns:
            The model wrapped by SwiftModel or PeftModel.
        """
        if not os.path.exists(model_id):
            model_id = snapshot_download(model_id, revision=revision)
        is_peft_model = False
        if os.path.exists(os.path.join(model_id, CONFIG_NAME)):
            with open(os.path.join(model_id, CONFIG_NAME), 'r') as f:
                _json = json.load(f)
            is_peft_model = SWIFT_TYPE_KEY not in _json

        _name = adapter_name if isinstance(
            adapter_name, str) or adapter_name is None else adapter_name[0]
        _name = _name or ''
        if os.path.exists(os.path.join(model_id, _name, CONFIG_NAME)):
            with open(os.path.join(model_id, _name, CONFIG_NAME), 'r') as f:
                _json = json.load(f)
            is_peft_model = SWIFT_TYPE_KEY not in _json
        if is_peft_model:
            return PeftModel.from_pretrained(
                model,
                model_id,
                revision=revision,
                adapter_name=adapter_name or 'default',
                **kwargs)
        else:
            return SwiftModel.from_pretrained(
                model,
                model_id,
                revision=revision,
                adapter_name=adapter_name,
                **kwargs)
