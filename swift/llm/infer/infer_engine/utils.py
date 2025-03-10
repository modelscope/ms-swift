# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from itertools import repeat
from queue import Queue
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from packaging import version
from transformers import GenerationConfig, LogitsProcessor
from transformers.generation.streamers import BaseStreamer

from swift.llm.model.register import fix_do_sample_warning
from swift.utils import get_device, get_device_count, get_node_setting
from ..protocol import RequestConfig


@dataclass
class AdapterRequest:
    name: str
    path: str


class InferTools:

    @staticmethod
    def _is_chinese_char(cp: int) -> bool:
        """Checks whether CP is the codepoint of a CJK character."""
        # copy from transformers.generation.streamers.TextStreamer
        if ((0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0x20000 <= cp <= 0x2A6DF)
                or (0x2A700 <= cp <= 0x2B73F) or (0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF)
                or (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F)):
            return True

        return False


class InferStreamer(InferTools):

    def __init__(self, template, **decode_kwargs):
        self.template = template
        self.tokenizer = template.tokenizer

        self.cache_idx = 0  # token idx
        self.print_idx = 0
        self.decode_kwargs = decode_kwargs
        self.first_num_space = -1  # The number of whitespace characters before the first token.
        self.first_token = True

    def _align_blank_suffix(self, response: str) -> str:
        # Avoid the occurrence of repeated words in sentence.
        cur_num_space = len(response) - len(response.lstrip(' '))
        if self.first_num_space == -1:
            self.first_num_space = cur_num_space
        elif cur_num_space < self.first_num_space:
            response = ' ' * (self.first_num_space - cur_num_space) + response
        elif cur_num_space > self.first_num_space:
            response = response[cur_num_space - self.first_num_space:]
        return response

    def _get_response(self, response: str, is_finished: bool, token_len: int) -> str:
        # After the symbol for a new line, we flush the cache.
        if response.endswith('\n') or is_finished:
            printable_text = response[self.print_idx:]
            self.cache_idx += token_len
            self.first_num_space = -1
            self.print_idx = 0
        # If the last token is a CJK character, we print the characters.
        elif len(response) > 0 and self._is_chinese_char(ord(response[-1])):
            printable_text = response[self.print_idx:]
            self.print_idx += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = response[self.print_idx:response.rfind(' ') + 1]
            self.print_idx += len(printable_text)
        return printable_text

    def get_printable_text(self, raw_tokens: List[int], is_finished: bool) -> str:
        raw_tokens = raw_tokens[self.cache_idx:]
        response = self.template.decode(
            raw_tokens, is_finished, tokenizer_kwargs=self.decode_kwargs, first_token=self.first_token)
        self.first_token = False
        response = self._align_blank_suffix(response)
        return self._get_response(response, is_finished, len(raw_tokens))


class StreamerMixin:

    def __init__(self):
        self.queue = Queue()

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        value = self.queue.get()
        if value is None:
            raise StopIteration()
        else:
            return value


class TokensIteratorStreamer(StreamerMixin, BaseStreamer):

    def put(self, value: torch.Tensor) -> None:
        self.queue.put(value)

    def end(self) -> None:
        self.queue.put(None)


class LogitsStreamer(LogitsProcessor):

    def __init__(self):
        self.queue = Queue()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.queue.put(scores)
        return scores


def _set_generation_config_default_value(model_generation_config: GenerationConfig,
                                         generation_config: GenerationConfig) -> GenerationConfig:
    for k, v in model_generation_config.to_dict().items():
        new_v = getattr(generation_config, k, None)
        if k in ['max_length']:
            continue
        if k in ['no_repeat_ngram_size'] or v is not None and new_v is None:
            setattr(generation_config, k, v)
    return generation_config


def prepare_generation_config(model_generation_config: Optional[GenerationConfig], request_config: RequestConfig,
                              tokenizer) -> Optional[GenerationConfig]:
    if model_generation_config is None or request_config is None:
        return model_generation_config
    kwargs = {'max_new_tokens': request_config.max_tokens}
    # not use: 'n', 'best_of', 'frequency_penalty', 'presence_penalty'
    for key in ['length_penalty']:
        kwargs[key] = getattr(request_config, key)
    for key in ['temperature', 'top_k', 'top_p', 'repetition_penalty', 'num_beams']:
        new_value = getattr(request_config, key)
        if new_value is None:
            kwargs[key] = getattr(model_generation_config, key)
        else:
            kwargs[key] = new_value

    if not model_generation_config.do_sample and request_config.temperature in {0, None}:
        kwargs['temperature'] = 0
    if kwargs['temperature'] == 0:
        kwargs['do_sample'] = False
        kwargs['temperature'] = 1
        kwargs['top_p'] = 1
        kwargs['top_k'] = 50
    else:
        kwargs['do_sample'] = True
    generation_config = GenerationConfig(**kwargs)
    generation_config = _set_generation_config_default_value(model_generation_config, generation_config)
    fix_do_sample_warning(generation_config)

    if generation_config.eos_token_id is None:
        generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.pad_token_id = tokenizer.pad_token_id
    return generation_config


def patch_lmdeploy(load_weights=False):
    """This patch allows lmdeploy selects device and reload state_dict"""
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
    from lmdeploy.turbomind.utils import ModelSource

    @contextmanager
    def patch_threadpool_map():
        ThreadPoolExecutor.map_origin = ThreadPoolExecutor.map
        ThreadPoolExecutor.map = lambda *args, **kwargs: []
        yield
        ThreadPoolExecutor.map = ThreadPoolExecutor.map_origin
        del ThreadPoolExecutor.map_origin

    @contextmanager
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
        with patch_threadpool_map(), tm_model_context(self):
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


def patch_vllm(world_size=1, vllm_device: Optional[str] = None):

    @contextmanager
    def _get_context():
        from vllm.distributed.parallel_state import GroupCoordinator
        from unittest.mock import patch
        try:
            from vllm.worker.worker import Worker
            getattr(Worker, '_assert_memory_footprint_increased_during_profiling')
            profiling_patch = patch(
                'vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling', return_value=None)
        except (ImportError, AttributeError):
            profiling_patch = nullcontext()

        __origin_init__ = GroupCoordinator.__init__

        def get_world_size(group=None) -> int:
            if not group:
                # Given size
                return world_size
            else:
                return torch.distributed.get_world_size_origin(group)

        def __init__(self, group_ranks, local_rank, *args, **kwargs):
            node_rank, nnodes = get_node_setting()
            device_count = get_device_count()
            num_infer_workers = world_size // nnodes

            def map_rank_to_real_device(obj):
                # Use the last devices
                # world_size=4 gpus=8 [0,1,2,3] will map to [4,5,6,7]
                diff = device_count - num_infer_workers
                if diff < 0:
                    diff = 0
                if isinstance(obj, list):
                    return [map_rank_to_real_device(o) for o in obj]
                elif isinstance(obj, int):
                    return obj + diff
                else:
                    raise ValueError(f'Unsupported type: {obj}')

            if kwargs.get('group_name') == 'world':
                local_rank = local_rank + node_rank * num_infer_workers
            else:
                local_rank = map_rank_to_real_device(local_rank - node_rank * num_infer_workers)
            rank = dist.get_rank()
            if world_size == 1 and [rank] not in group_ranks:
                # for ddp inference
                group_ranks = [[rank]]
            return __origin_init__(self, group_ranks, local_rank, *args, **kwargs)

        GroupCoordinator.__init__ = __init__

        try:
            with profiling_patch, set_local_rank_context(vllm_device):
                torch.distributed.get_world_size_origin = torch.distributed.get_world_size
                torch.distributed.get_world_size = get_world_size
                yield
                torch.distributed.get_world_size = torch.distributed.get_world_size_origin
                del torch.distributed.get_world_size_origin
        finally:
            GroupCoordinator.__init__ = __origin_init__

    return _get_context() if dist.is_initialized() else nullcontext()


def patch_npu_vllm(vllm_device: str):
    if isinstance(vllm_device, int):
        vllm_device = get_device(vllm_device)
    device_type = vllm_device.split(':')[0]

    @contextmanager
    def new_group_context():
        original_new_group = torch.distributed.new_group
        try:
            torch.distributed.new_group = partial(original_new_group, use_local_synchronization=True)
            torch.npu.mem_get_info = partial(torch.npu.mem_get_info, device=vllm_device)
            yield
        finally:
            torch.distributed.new_group = original_new_group

    return new_group_context() if device_type == 'npu' else nullcontext()


@contextmanager
def set_device_context(device: Union[str, int]):
    original_device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    try:
        yield
    finally:
        torch.cuda.set_device(original_device)


@contextmanager
def set_local_rank_context(device: Union[str, int]):
    if isinstance(device, str):
        device = int(device.split(':')[-1])
    origin_local_rank = os.environ.get('LOCAL_RANK', None)
    os.environ['LOCAL_RANK'] = str(device)
    try:
        yield
    finally:
        if origin_local_rank is not None:
            os.environ['LOCAL_RANK'] = origin_local_rank
        else:
            del os.environ['LOCAL_RANK']
