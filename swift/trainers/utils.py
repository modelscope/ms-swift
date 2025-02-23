# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import copy
import inspect
import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import repeat
from queue import Queue
from types import FunctionType, MethodType
from typing import List, Union

import torch
from torch.nn import Module

from swift.utils import get_logger

logger = get_logger()


def can_return_loss(model: Module) -> bool:
    """Check if a given model can return loss."""
    signature = inspect.signature(model.forward)
    for p in signature.parameters:
        if p == 'return_loss' and signature.parameters[p].default is True:
            return True
    return False


def find_labels(model: Module) -> List[str]:
    """Find the labels used by a given model."""
    model_name = model.__class__.__name__
    signature = inspect.signature(model.forward)
    if 'QuestionAnswering' in model_name:
        return [p for p in signature.parameters if 'label' in p or p in ('start_positions', 'end_positions')]
    else:
        return [p for p in signature.parameters if 'label' in p]


def get_function(method_or_function: Union[MethodType, FunctionType]) -> FunctionType:
    if isinstance(method_or_function, MethodType):
        method_or_function = method_or_function.__func__
    return method_or_function


def is_instance_of_ms_model(model: Module) -> bool:
    """avoid import modelscope: circular dependency problem"""
    for m_cls in model.__class__.__mro__:
        cls_name = m_cls.__name__
        cls_module = m_cls.__module__
        if cls_name == 'Model' and cls_module.startswith('modelscope'):
            return True
    return False


def _patch_lmdeploy():
    import lmdeploy
    assert lmdeploy.__version__ == '0.6.4'
    from lmdeploy import messages
    from lmdeploy.messages import TurbomindEngineConfig
    from lmdeploy.turbomind.deploy import loader
    from lmdeploy.turbomind.deploy.loader import create_loader
    from lmdeploy.turbomind.deploy.source_model import llama

    def _create_loader(model_path: str, pattern: str):
        if not isinstance(model_path, (str, os.PathLike)):

            def generate():
                generator = OrderedDict()
                model_dict = {}
                if not isinstance(model_path, dict):
                    for key, value in list(model_path):
                        model_dict[key] = value
                else:
                    model_dict = model_path
                for key, value in model_dict.items():
                    match = re.findall(pattern, key)
                    if not match:
                        if -1 not in generator:
                            generator[-1] = {}
                        generator[-1][key] = value
                    else:
                        layer = int(match[0])
                        if layer not in generator:
                            generator[layer] = {}
                        generator[layer][key] = value
                return generator

            return generate()
        else:
            return create_loader(model_path, pattern)

    loader.create_loader = _create_loader
    llama.create_loader = _create_loader

    TurbomindEngineConfig.devices = [0]

    from lmdeploy.turbomind.turbomind import TurboMind
    from lmdeploy.tokenizer import Tokenizer
    from lmdeploy.utils import get_max_batch_size, get_model
    from lmdeploy.turbomind.utils import ModelSource, get_model_source
    import _turbomind as _tm
    from lmdeploy.turbomind.supported_models import is_supported

    def __init__(self,
                 model_path: str,
                 model_name: str = None,
                 chat_template_name: str = None,
                 engine_config: TurbomindEngineConfig = None,
                 model_source='workspace',
                 **kwargs):
        self.model_name = model_name
        self.chat_template_name = chat_template_name

        _engine_config = copy.deepcopy(engine_config)
        if _engine_config is None:
            _engine_config = TurbomindEngineConfig()
        if _engine_config.max_batch_size is None:
            _engine_config.max_batch_size = get_max_batch_size('cuda')
        assert _engine_config.max_batch_size > 0, 'max_batch_size should be' \
                                                  f' greater than 0, but got {_engine_config.max_batch_size}'

        self.gpu_count = _engine_config.tp
        self.gpu_list = _engine_config.devices
        os.environ['LMDEPLOY_DEVICE_LIST'] = str(_engine_config.devices)

        if model_source == 'workspace':
            tokenizer_model_path = os.path.join(model_path, 'triton_models', 'tokenizer')
            self.tokenizer = Tokenizer(tokenizer_model_path)
            self.model_comm = self._from_workspace(model_path=model_path, engine_config=_engine_config)
        else:
            if not os.path.exists(model_path):
                model_path = get_model(model_path, _engine_config.download_dir, _engine_config.revision)
            self.tokenizer = Tokenizer(model_path)
            self.model_comm = self._from_hf(
                model_source=model_source, model_path=model_path, engine_config=_engine_config)

        with ThreadPoolExecutor(max_workers=self.gpu_count) as e:
            ranks = [self.node_id * self.gpu_count + device_id for device_id in range(self.gpu_count)]
            # This is for load_state_dict
            # process_weight will optimizer the kernel by col major matrix and pack_b
            # This will result in the failure of get_params
            # for _ in e.map(self.model_comm.process_weight,
            #                self.gpu_list, ranks):
            #     pass

            # implicit synchronization
            for _ in e.map(self.model_comm.create_engine, self.gpu_list, ranks, repeat(self.nccl_params)):
                pass

        self.session_len = self.config.session_len
        self.eos_id = self.tokenizer.eos_token_id

    def _create_weight(self, model_comm):
        """Allocate weight buffer, load params if from_workspace."""

        # TODO: support mpi
        self.node_id = 0
        self.node_num = 1
        self.nccl_params = model_comm.create_nccl_params(self.node_id)
        torch.cuda.synchronize()

        # create weight
        def _create_weight_func(index, device_id):
            rank = self.node_id * self.gpu_count + index
            model_comm.create_shared_weights(device_id, rank)

        with ThreadPoolExecutor(max_workers=self.gpu_count) as executor:
            futures = []
            for idx, device_id in enumerate(self.gpu_list):
                futures.append(executor.submit(_create_weight_func, idx, device_id))
            for future in futures:
                future.result()

    def _get_model_params(self, model_comm, tm_params):
        """Get turbomind model params when loading from hf."""

        def _get_params(idx, device_id, que):
            rank = self.node_id * self.gpu_count + idx
            out = model_comm.get_params(device_id, rank)
            que.put(out)

        que = Queue()
        with ThreadPoolExecutor(max_workers=self.gpu_count) as executor:
            futures = []
            for idx, device_id in enumerate(self.gpu_list):
                futures.append(executor.submit(_get_params, idx, device_id, que))
            for future in futures:
                future.result()

        for _ in range(self.gpu_count):
            tensor_map = que.get()
            for k, v in tensor_map.items():
                if k not in tm_params:
                    tm_params[k] = []
                tm_params[k].append(v)

    def _from_hf(self, model_source, model_path: str, engine_config: TurbomindEngineConfig):
        """Load model which is in hf format."""
        assert model_source == ModelSource.HF_MODEL, \
            f'{model_source} is not supported'
        assert is_supported(model_path), (f'turbomind does not support {model_path}. '
                                          'Plz try pytorch engine instead.')

        # convert transformers model into turbomind model
        from lmdeploy.turbomind.deploy.converter import get_tm_model
        tm_model = get_tm_model(model_path, self.model_name, self.chat_template_name, engine_config)

        self._postprocess_config(tm_model.tm_config, engine_config)
        import yaml
        model_comm = _tm.AbstractTransformerModel.create_llama_model(
            model_dir='',
            config=yaml.safe_dump(self.config_dict),
            tensor_para_size=self.gpu_count,
            data_type=self.config.model_config.weight_type)

        # create empty weight
        self._create_weight(model_comm)

        # copy hf model weight to turbomind weight
        tm_params = tm_model.tm_params
        self._get_model_params(model_comm, tm_params)
        logger.warning(f'get {len(tm_params)} model params')
        tm_model.export()
        self.tm_model = tm_model
        # there should be no left turbomind params.
        if len(tm_params) > 0:
            uninitialized = list(tm_params.keys())
            logger.warning('the model may not be loaded successfully '
                           f'with {len(tm_params)} uninitialized params:\n{uninitialized}')
        return model_comm

    def load_weights(self, state_dict):
        tm_params = self.tm_model.tm_params
        self._get_model_params(self.model_comm, tm_params)
        input_model = self.tm_model.input_model
        model_path = input_model.model_path
        input_model.model_path = state_dict
        self.tm_model.export()
        input_model.model_path = model_path

    TurboMind.__init__ = __init__
    TurboMind._create_weight = _create_weight
    TurboMind._get_model_params = _get_model_params
    TurboMind._from_hf = _from_hf
    TurboMind.load_weights = load_weights

    from lmdeploy.turbomind.turbomind import TurboMindInstance

    def _create_model_instance(self, device_id):
        rank = self.node_id * self.gpu_count + device_id
        model_inst = self.tm_model.model_comm.create_model_instance(
            eval(os.environ['LMDEPLOY_DEVICE_LIST'])[0], rank, self.cuda_stream_id, self.nccl_params)
        return model_inst

    TurboMindInstance._create_model_instance = _create_model_instance
