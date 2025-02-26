# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import contextlib
import inspect
import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from queue import Queue
from types import FunctionType, MethodType
from typing import List, Union

import torch
from packaging import version
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


def _patch_lmdeploy(load_weights=False):
    import lmdeploy
    assert version.parse(lmdeploy.__version__) >= version.parse('0.7.0')
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
    from lmdeploy.turbomind.utils import ModelSource, get_model_source

    @contextlib.contextmanager
    def patch_threadpool():
        ThreadPoolExecutor.map_origin = ThreadPoolExecutor.map
        ThreadPoolExecutor.map = lambda *args, **kwargs: []
        yield
        ThreadPoolExecutor.map = ThreadPoolExecutor.map_origin
        del ThreadPoolExecutor.map_origin

    @contextlib.contextmanager
    def tm_model_context(self):

        def _get_tm_model(model_path,
                          model_name,
                          chat_template_name,
                          engine_config: TurbomindEngineConfig,
                          group_size: int = None,
                          out_dir: str = None):
            from lmdeploy.turbomind.deploy.converter import get_tm_model_origin
            tm_model = get_tm_model_origin(model_path, model_name, chat_template_name, engine_config, group_size,
                                           out_dir)
            self.tm_model = tm_model
            return tm_model

        from lmdeploy.turbomind.deploy import converter
        converter.get_tm_model_origin = converter.get_tm_model
        converter.get_tm_model = _get_tm_model
        yield
        converter.get_tm_model = converter.get_tm_model_origin
        del converter.get_tm_model_origin

    def __init__(self,
                 model_path: str,
                 tokenizer: object,
                 model_name: str = None,
                 chat_template_name: str = None,
                 engine_config: TurbomindEngineConfig = None,
                 model_source: ModelSource = ModelSource.WORKSPACE,
                 **kwargs):
        self.gpu_list = engine_config.devices
        with patch_threadpool(), tm_model_context(self):
            self.__origin_init__(model_path, tokenizer, model_name, chat_template_name, engine_config, model_source,
                                 **kwargs)

        with ThreadPoolExecutor(max_workers=self.gpu_count) as e:
            ranks = [self.node_id * self.gpu_count + device_id for device_id in range(self.gpu_count)]
            if not load_weights:
                for _ in e.map(self.model_comm.process_weight, self.gpu_list, ranks):
                    pass
            for _ in e.map(self.model_comm.create_engine, self.gpu_list, ranks, repeat(self.nccl_params)):
                pass

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

    def _load_weights(self, state_dict):
        tm_params = self.tm_model.tm_params
        self._get_model_params(self.model_comm, tm_params)
        input_model = self.tm_model.input_model
        model_path = input_model.model_path
        input_model.model_path = state_dict
        self.tm_model.export()
        input_model.model_path = model_path

    from lmdeploy.turbomind.turbomind import TurboMindInstance

    def create_instance(self, cuda_stream_id=0):
        return TurboMindInstance(self, self.config, cuda_stream_id, self.gpu_list)

    TurboMind.__origin_init__ = TurboMind.__init__
    TurboMind.__init__ = __init__
    TurboMind._create_weight = _create_weight
    TurboMind._get_model_params = _get_model_params
    TurboMind.create_instance = create_instance
    if load_weights:
        TurboMind.load_weights = _load_weights

    def __init_ins__(self, tm_model, config, cuda_stream_id=0, gpu_list=None):
        if gpu_list is None:
            gpu_list = [0]
        self.gpu_list = gpu_list
        self.__origin_init__(tm_model, config, cuda_stream_id)

    def _create_model_instance(self, device_id):
        model_inst = self.tm_model.model_comm.create_model_instance(self.gpu_list[0])
        return model_inst

    TurboMindInstance.__origin_init__ = TurboMindInstance.__init__
    TurboMindInstance.__init__ = __init_ins__
    TurboMindInstance._create_model_instance = _create_model_instance
