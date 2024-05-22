# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import heapq
import importlib.util
import logging
import os
import shutil
from copy import deepcopy
from functools import partial, wraps
from queue import Empty, Queue
from tempfile import TemporaryDirectory
from threading import Thread
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import accelerate
import multiprocess
import numpy as np
import requests
import torch
import torch.distributed as dist
import transformers
from datasets import Dataset as HfDataset
from datasets.arrow_writer import SchemaInferenceError
from datasets.exceptions import DatasetGenerationError
from modelscope.utils.config_ds import MS_CACHE_HOME
from modelscope.utils.logger import get_logger as get_ms_logger
from torch import device as Device
from torch.nn import Linear, Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, IterableDataset
from tqdm.auto import tqdm
from transformers import (GenerationConfig, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase,
                          StoppingCriteriaList, TextStreamer, trainer)
from transformers.generation.streamers import BaseStreamer
from transformers.utils import is_torch_npu_available, strtobool

from swift.hub import ModelScopeConfig
from swift.tuners.module_mapping import MODEL_KEYS_MAPPING
from swift.utils import (get_dist_setting, get_logger, is_ddp_plus_mp, is_dist, is_local_master, safe_ddp_context,
                         stat_array, upper_bound, use_torchacc)
from .template import History, StopWords, StopWordsCriteria, Template

logger = get_logger()
ms_logger = get_ms_logger()

logger_format = logging.Formatter('[%(levelname)s:%(name)s] %(message)s')

logger.handlers[0].setFormatter(logger_format)
ms_logger.handlers[0].setFormatter(logger_format)
if is_local_master():
    logger.setLevel(logging.INFO)
    ms_logger.setLevel(logging.INFO)
else:
    logger.setLevel(logging.ERROR)
    ms_logger.setLevel(logging.ERROR)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def download_files(url: str, local_path: str, cookies) -> None:
    resp = requests.get(url, cookies=cookies, stream=True)
    with open(local_path, 'wb') as f:
        for data in tqdm(resp.iter_lines()):
            f.write(data)


def download_dataset(model_id: str, files: List[str], force_download: bool = False) -> str:
    assert isinstance(files, list)
    url = f'http://www.modelscope.cn/api/v1/datasets/{model_id}/repo?Revision=master&FilePath={{fpath}}'
    cache_dir = os.path.join(MS_CACHE_HOME, 'datasets', model_id, 'master')
    local_dir = os.path.join(cache_dir, 'raw')
    tmp_dir = os.path.join(cache_dir, 'tmp')
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    cookies = ModelScopeConfig.get_cookies()
    with TemporaryDirectory(dir=tmp_dir) as temp_dir:
        for remote_fpath in files:
            url = url.format(fpath=remote_fpath)
            temp_fpath = os.path.join(temp_dir, remote_fpath)
            local_fpath = os.path.join(local_dir, remote_fpath)
            if not force_download and os.path.exists(local_fpath):
                continue
            download_files(url, temp_fpath, cookies)
            shutil.copy2(temp_fpath, local_fpath)

    return local_dir


use_hf = strtobool(os.environ.get('USE_HF', 'False'))
if not use_hf:
    from modelscope import MsDataset
    _old_msdataset_load = MsDataset.load

    @wraps(_old_msdataset_load)
    def _msdataset_ddp_load(*args, **kwargs):
        with safe_ddp_context():
            dataset = _old_msdataset_load(*args, **kwargs)

        if is_dist():  # sync
            dist.barrier()
        return dataset

    # monkey patching
    MsDataset.load = _msdataset_ddp_load


def _get_max_memory(device_ids: List[int]) -> Dict[Union[int, str], int]:
    """add feat in accelerate to support DDP + MP"""
    import psutil
    # Make sure CUDA is initialized on each GPU to have the right memory info.
    for i in device_ids:
        _ = torch.tensor([0], device=i)

    device_ids_set = set(device_ids)
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        max_memory[i] = 0
        if i in device_ids_set:
            max_memory[i] = torch.cuda.mem_get_info(i)[0]
    max_memory['cpu'] = psutil.virtual_memory().available
    return max_memory


def _sync_max_memory(max_memory: Dict[Union[int, str], int]) -> Dict[Union[int, str], int]:
    """Make sure that the model structure of MP(device_map) is the same, when using DDP."""
    max_memory_list = [v for k, v in max_memory.items() if (v > 0 and k != 'cpu')]
    _, local_rank, world_size, _ = get_dist_setting()
    src_tensor = torch.tensor(max_memory_list).to(local_rank)
    tgt_tensor_list = [torch.zeros_like(src_tensor) for _ in range(world_size)]
    dist.all_gather(tgt_tensor_list, src_tensor)
    tgt_tensor = torch.stack(tgt_tensor_list, dim=0)
    new_max_memory_iter = iter(tgt_tensor.min(dim=0)[0].tolist())
    new_max_memory = {}
    for k, v in max_memory.items():
        new_max_memory[k] = v
        if v > 0 and k != 'cpu':
            new_max_memory[k] = next(new_max_memory_iter)
    return new_max_memory


class LLMDataset(Dataset):

    def __init__(self, data: List[Dict[str, Any]]) -> None:
        self.data = data

    def __getitem__(self, idx: Union[int, str]) -> Dict[str, Any]:
        if isinstance(idx, int):
            data, _ = self.data[idx]
            return data
        elif isinstance(idx, str):
            return [d[0][idx] for d in self.data]
        else:
            raise ValueError(f'idx: {idx}')

    def select(self, idx_list: List[int]) -> 'LLMDataset':
        data = [self.data[i] for i in idx_list]
        return self.__class__(data)

    def __len__(self) -> int:
        return len(self.data)


# Code borrowed from trl
class ConstantLengthDataset(IterableDataset):

    def __init__(
        self,
        template: 'Template',
        dataset,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        self.template = template

        self.concat_token_id = self.template.tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.append_concat_token = append_concat_token
        self.add_special_tokens = add_special_tokens

    @staticmethod
    def get_packed_dataset(template: 'Template',
                           dataset,
                           seq_length=1024,
                           num_of_sequences=1024,
                           chars_per_token=3.6,
                           append_concat_token=True,
                           add_special_tokens=True,
                           lazy_tokenize=False):
        constant_length_iterator = ConstantLengthDataset(template, dataset, seq_length, num_of_sequences,
                                                         chars_per_token, append_concat_token, add_special_tokens)

        if lazy_tokenize:
            return constant_length_iterator

        def data_generator(constant_length_iterator):
            yield from constant_length_iterator

        try:
            packed_dataset = HfDataset.from_generator(
                data_generator, gen_kwargs={'constant_length_iterator': constant_length_iterator})
        except (DatasetGenerationError, SchemaInferenceError) as exc:
            raise ValueError(
                'Error occurred while packing the dataset. '
                'Make sure that your dataset has enough samples to at least yield one packed sequence.') from exc
        return packed_dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    example = next(iterator)
                    lens = sum([len(value) if value else 0 for value in example.values()])
                    buffer.append(next(iterator))
                    buffer_len += lens
                except StopIteration:
                    more_examples = False
                    break

            packed_sequences = {}
            for example in buffer:
                input, _ = self.template.encode(example)
                for key in input.keys():
                    if key not in packed_sequences:
                        packed_sequences[key] = []
                    packed_sequences[key].extend(input[key])

            lens = len(packed_sequences[list(packed_sequences.keys())[0]])
            for i in range(0, lens, self.seq_length):
                example = {key: value[i:i + self.seq_length] for key, value in packed_sequences.items()}
                yield example


class LazyLLMDataset(Dataset):

    def __init__(self, dataset: HfDataset, template: Template, *, try_fetch_time: int = 20) -> None:
        self.dataset = dataset
        self.template = template
        self.try_fetch_time = min(try_fetch_time, len(self.dataset))
        assert self.try_fetch_time >= 1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        res = self._try_fetch(idx)
        if res is not None:
            data, _ = res
            return data
        raise ValueError('Please check if the max_length is appropriate.')

    def _try_fetch(self, first_idx: int) -> Optional[Dict[str, Any]]:
        idx = np.random.permutation(len(self))[:self.try_fetch_time - 1]
        for i in [first_idx] + idx.tolist():
            data = self.dataset[i]
            res = self.template.encode(data)
            if len(res[0]) > 0:
                return res

    def __len__(self) -> int:
        return len(self.dataset)


MapFunc = Callable[[Dict[str, Any]], Dict[str, Any]]


def _single_map(d: Dict[str, Any], map_func: MapFunc) -> Optional[Dict[str, Any]]:
    d = map_func(d)
    if len(d[0]) == 0:
        return None
    return d


def _map_mp_single(subset: HfDataset, map_func: MapFunc, queue: Queue, start_idx: int):
    for i, d in enumerate(subset, start=start_idx):
        queue.put((i, map_func(d)))  # idx, result


def _map_mp_i(dataset: HfDataset, map_func: MapFunc, num_proc: int) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with multiprocess.Pool(num_proc) as pool, multiprocess.Manager() as manager:
        queue = manager.Queue()
        async_results = []
        split_idx = np.linspace(0, len(dataset), num_proc + 1, dtype=np.int32)
        for i in range(num_proc):
            subset = dataset.select(range(split_idx[i], split_idx[i + 1]))
            async_results.append(pool.apply_async(_map_mp_single, args=(subset, map_func, queue, split_idx[i])))
        while True:
            try:
                yield queue.get(timeout=0.05)
            except Empty:
                if all(async_result.ready() for async_result in async_results) and queue.empty():
                    break


def _map_mp(dataset: HfDataset, map_func: MapFunc, num_proc: int) -> List[Dict[str, Any]]:
    # Solving the unordered problem
    data = [None] * len(dataset)
    num_proc = min(num_proc, len(dataset))
    for d in tqdm(_map_mp_i(dataset, map_func, num_proc), total=len(dataset)):
        data[d[0]] = d[1]
    return data


def dataset_map(dataset: HfDataset, map_func: MapFunc, num_proc: int = 1) -> Optional[LLMDataset]:
    single_map = partial(_single_map, map_func=map_func)
    if num_proc == 1:
        data = []
        for d in tqdm(dataset):
            d = single_map(d)
            data.append(d)
    else:
        assert num_proc > 1
        data = _map_mp(dataset, single_map, num_proc)
    data = [d for d in data if d is not None]
    if len(data) == 0:
        logger.warning('len(dataset): 0')
        return None
    return LLMDataset(data)


def stat_dataset(llm_dataset: Dataset) -> str:
    """Statistical analysis was performed on the dataset"""
    _token_len = []
    if isinstance(llm_dataset, HfDataset):
        input_ids = llm_dataset['input_ids']
        for ii in input_ids:
            _token_len.append(len(ii))
    else:
        for d in llm_dataset:
            _token_len.append(len(d['input_ids']))
    _, stat_str = stat_array(_token_len)
    logger.info(f'Dataset Token Length: {stat_str}')
    return stat_str


def safe_tokenizer_decode(tokenizer: PreTrainedTokenizerBase, input_ids: List[int], **tokenizer_kwargs) -> str:

    def _is_special(token: int) -> bool:
        if token < 0:
            return True
        if tokenizer.eos_token_id != tokenizer.pad_token_id:
            return token == tokenizer.pad_token_id
        return False

    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if len(input_ids) == 0:
        return ''
    result_str = ''
    for i in range(len(input_ids)):
        if i == 0:
            if _is_special(input_ids[i]):
                s = 0
            else:
                e = 0
            continue
        if _is_special(input_ids[i]) and not _is_special(input_ids[i - 1]):
            s = i
            result_str += tokenizer.decode(input_ids[e:s], **tokenizer_kwargs)
        if not _is_special(input_ids[i]) and _is_special(input_ids[i - 1]):
            e = i
            result_str += f'[{input_ids[i - 1]} * {e - s}]'
    if _is_special(input_ids[-1]):
        result_str += f'[{input_ids[i - 1]} * {len(input_ids) - s}]'
    else:
        result_str += tokenizer.decode(input_ids[e:], **tokenizer_kwargs)
    return result_str


def print_example(example: Dict[str, Any],
                  tokenizer: PreTrainedTokenizerBase,
                  tokenizer_kwargs: Optional[Dict[str, Any]] = None) -> None:
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}
    input_ids = example.get('input_ids')
    labels = example.get('labels')
    if input_ids is not None:
        logger.info(f'[INPUT_IDS] {input_ids}')
        input_str = safe_tokenizer_decode(tokenizer, input_ids, **tokenizer_kwargs)
        logger.info(f'[INPUT] {input_str}')
    if labels is not None:
        logger.info(f'[LABLES_IDS] {labels}')
        labels_str = safe_tokenizer_decode(tokenizer, labels, **tokenizer_kwargs)
        logger.info(f'[LABLES] {labels_str}')


def _find_layers(model: Module, module_cls: type) -> List[str]:
    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, module_cls):
            module_name = '.'.join(name.split('.')[-2:])
            module_names.add(module_name)
    return list(module_names)


def find_ln(model: Module) -> List[str]:
    # find_layer_norm
    module_names = set()
    for name, module in model.named_modules():
        module_cls_name = module.__class__.__name__.lower()
        if isinstance(module, torch.nn.LayerNorm) or 'rmsnorm' in module_cls_name:
            module_name = '.'.join(name.split('.')[-1:])
            module_names.add(module_name)
    return list(module_names)


def find_embedding(model: Module) -> List[str]:
    return _find_layers(model, torch.nn.Embedding)


def find_all_linears(model: Module, quantization_bit: int, model_type: str) -> List[str]:
    """ref: https://github.com/artidoro/qlora"""
    head_module_name = 'lm_head'
    if model_type in MODEL_KEYS_MAPPING:
        output = MODEL_KEYS_MAPPING[model_type].output
        idx = output.rfind('.')
        head_module_name = output[idx + 1:]
    if quantization_bit == 4:
        from bitsandbytes.nn import Linear4bit
        linear_cls = [Linear4bit]
    elif quantization_bit == 8:
        from bitsandbytes.nn import Linear8bitLt
        linear_cls = [Linear8bitLt]
    else:
        linear_cls = [Linear]
    if 'int4' in model_type or 'int8' in model_type:
        from peft.utils import get_auto_gptq_quant_linear, get_quantization_config
        gptq_quantization_config = get_quantization_config(model, 'gptq')
        AutoGPTQQuantLinear = get_auto_gptq_quant_linear(gptq_quantization_config)
        if AutoGPTQQuantLinear is None:
            from bitsandbytes.nn import Linear4bit
            linear_cls = [Linear4bit]
        else:
            linear_cls = [AutoGPTQQuantLinear]
    if 'awq' in model_type:
        from awq.modules.linear import WQLinear_GEMM
        linear_cls.append(WQLinear_GEMM)
    if 'aqlm' in model_type:
        from aqlm import QuantizedLinear
        linear_cls.append(QuantizedLinear)

    # The content of target_module_names cannot exist in inner_nodes.
    # O(n^2logn), n represents the number of nodes, n<1000.
    inner_nodes = set()
    for name, module in model.named_modules():
        if not isinstance(module, tuple(linear_cls)):
            inner_nodes.add(name)
    target_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, tuple(linear_cls)) and head_module_name not in name:
            module_name_list = name.split('.')
            module_name = module_name_list.pop()
            for inner_node in inner_nodes:
                while inner_node.endswith(module_name):
                    module_name = f'{module_name_list.pop()}.{module_name}'
            target_module_names.add(module_name)
    return list(target_module_names)


def sort_by_max_length(llm_dataset: LLMDataset, num_dataset: int) -> HfDataset:
    logger.info('sort by max length...')
    dataset_len = [len(d['input_ids']) for d in llm_dataset]
    idx = heapq.nlargest(num_dataset, range(len(dataset_len)), key=lambda i: dataset_len[i])
    return llm_dataset.select(idx)


def to_device(inputs: Any, device: Device) -> Any:
    if callable(getattr(inputs, 'to', None)):
        return inputs.to(device=device)

    if isinstance(inputs, Mapping):
        res = {}
        for k, v in inputs.items():
            res[k] = to_device(v, device)
    elif isinstance(inputs, Sequence) and not isinstance(inputs, str):
        res = []
        for b in inputs:
            res.append(to_device(b, device))
    elif isinstance(inputs, (int, float, str)) or inputs is None:
        res = inputs
    else:
        raise TypeError(f'inputs: {inputs}, {type(inputs)}')
    return res


class TokenListIteratorStreamer(BaseStreamer):

    def __init__(self, timeout: Optional[float] = None):
        self.token_queue = Queue()  # Queue[int]
        self.stop_signal = None
        self.timeout = timeout

    def put(self, value: torch.Tensor) -> None:
        if value.ndim > 1:
            value = value[0]
        value = value.tolist()
        for v in value:
            self.token_queue.put(v)

    def end(self) -> None:
        self.token_queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.token_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


@torch.inference_mode()
def inference_stream(model: PreTrainedModel,
                     template: Template,
                     query: str,
                     history: Optional[History] = None,
                     system: Optional[str] = None,
                     *,
                     generation_config: Optional[GenerationConfig] = None,
                     stop_words: Optional[StopWords] = None,
                     generation_info: Optional[Dict[str, int]] = None,
                     adapter_names: Optional[List[str]] = None,
                     **kwargs) -> Iterator[Tuple[str, History]]:
    """
    generation_config: Priority: generation_config > model.generation_config.
    """
    if stop_words is None:
        stop_words = []
    if history is None:
        history = []
    else:
        history = deepcopy(history)

    # agent support
    is_observation = history[-1][-1].endswith('Observation:') if history and history[-1][-1] else False
    if is_observation:
        history[-1][-1] = history[-1][-1] + query
        act_length = len(history[-1][-1])
        query = None

    example = {
        'query': query,
        'history': history,
        'system': system,
        'images': kwargs.pop('images', None)  # for vl. str.
    }
    template.model = model
    inputs, tokenizer_kwargs = template.encode(example)
    if len(inputs) == 0:
        raise ValueError('input_ids exceeds `max_length`. Please increase the value of `max_length`.')
    inputs.pop('labels', None)
    tokenizer = template.tokenizer
    device = next(model.parameters()).device
    if 'input_ids' in inputs:
        input_ids = torch.tensor(inputs['input_ids'])[None]
        inputs['input_ids'] = input_ids
        token_len = input_ids.shape[1]
    if 'inputs_embeds' in inputs:
        inputs_embeds = inputs['inputs_embeds'][None]
        inputs['inputs_embeds'] = inputs_embeds
        token_len = inputs_embeds.shape[1]

    inputs['attention_mask'] = torch.ones(token_len, dtype=torch.int64)[None]
    if 'token_type_ids' in inputs:
        inputs['token_type_ids'] = torch.tensor(inputs['token_type_ids'])[None]
    model.eval()
    if generation_config is None:
        generation_config = getattr(model, 'generation_config', None)
    generation_config = deepcopy(generation_config)
    if generation_config.num_beams != 1:
        error_msg = 'Streaming generation does not support beam search.'
        raise ValueError(error_msg)

    if tokenizer.eos_token_id is not None:
        generation_config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.bos_token_id is not None:
        generation_config.bos_token_id = tokenizer.bos_token_id
    if generation_config.max_new_tokens is not None:
        generation_config.max_length = 20  # fix max_length, max_new_tokens warning
        max_length = get_max_model_len(model.config)
        if max_length and token_len + generation_config.max_new_tokens > max_length:
            generation_config.max_new_tokens = max_length - token_len
            if generation_config.max_new_tokens <= 0:
                raise AssertionError('Current sentence length exceeds' f'the model max_length: {max_length}')
    if template.suffix[-1] not in stop_words:
        stop_words.append(template.suffix[-1])
    stopping_criteria = StoppingCriteriaList([StopWordsCriteria(tokenizer, stop_words, **tokenizer_kwargs)])
    inputs = to_device(inputs, device)
    if generation_info is not None:
        generation_info['num_prompt_tokens'] = token_len
    if 'inputs_embeds' in inputs:
        inputs.pop('input_ids', None)
    streamer = TokenListIteratorStreamer()
    if adapter_names is not None:
        inputs['adapter_names'] = adapter_names
    generation_kwargs = {
        'streamer': streamer,
        'generation_config': generation_config,
        'stopping_criteria': stopping_criteria,
        **inputs
    }
    _model_generate = model.generate
    if is_torch_npu_available():

        def _model_generate(*args, **kwargs):
            torch.npu.set_device(model.device)
            return model.generate(*args, **kwargs)

    thread = Thread(target=_model_generate, kwargs=generation_kwargs)
    thread.start()
    raw_generate_ids, generate_ids = [], []

    if not is_observation:
        history.append(None)  # dummy

    print_idx = [0]
    first_num_space = [-1]

    is_finished = False
    while not is_finished:
        try:
            token = next(streamer)
            raw_generate_ids.append(token)
        except StopIteration:
            is_finished = True
        generate_ids = template.get_generate_ids(torch.tensor(raw_generate_ids)[None], token_len)
        if generation_info is not None:
            generation_info['num_generated_tokens'] = len(generate_ids)
        response = template.generate_ids_to_response(
            generate_ids,
            is_finished,
            tokenizer_kwargs=tokenizer_kwargs,
            print_idx=print_idx,
            first_num_space=first_num_space)
        if not is_observation:
            history[-1] = [query, response]
        else:
            history[-1][-1] = history[-1][-1][:act_length] + response
        yield response, history


@torch.inference_mode()
def inference(model: PreTrainedModel,
              template: Template,
              query: str,
              history: Optional[History] = None,
              system: Optional[str] = None,
              *,
              generation_config: Optional[GenerationConfig] = None,
              stop_words: Optional[StopWords] = None,
              stream: bool = False,
              verbose: bool = False,
              prompt_prefix: str = '[PROMPT]',
              output_prefix: str = '[OUTPUT]',
              generation_info: Optional[Dict[str, int]] = None,
              adapter_names: Optional[List[str]] = None,
              **kwargs) -> Tuple[str, History]:
    """
    generation_config: Priority: generation_config > model.generation_config.
    """
    if stop_words is None:
        stop_words = []
    if history is None:
        history = []
    else:
        history = deepcopy(history)

    is_observation = history[-1][-1].endswith('Observation:') if history and history[-1][-1] else False
    if is_observation:
        history[-1][-1] = history[-1][-1] + query
        query = None

    example = {
        'query': query,
        'history': history,
        'system': system,
        'images': kwargs.pop('images', None)  # for vl. str.
    }
    template.model = model
    inputs, tokenizer_kwargs = template.encode(example)

    truncation_strategy = kwargs.pop('truncation_strategy', None)
    if len(inputs) == 0 and truncation_strategy == 'delete':
        return '', history

    inputs.pop('labels', None)
    tokenizer = template.tokenizer
    device = next(model.parameters()).device
    if 'input_ids' in inputs:
        input_ids = torch.tensor(inputs['input_ids'])[None]
        inputs['input_ids'] = input_ids
        token_len = input_ids.shape[1]
    if 'inputs_embeds' in inputs:
        inputs_embeds = inputs['inputs_embeds'][None]
        inputs['inputs_embeds'] = inputs_embeds
        token_len = inputs_embeds.shape[1]

    inputs['attention_mask'] = torch.ones(token_len)[None]
    if 'token_type_ids' in inputs:
        inputs['token_type_ids'] = torch.tensor(inputs['token_type_ids'])[None]
    model.eval()
    if generation_config is None:
        generation_config = getattr(model, 'generation_config', None)
    generation_config = deepcopy(generation_config)
    if stream is True and verbose is False:
        logger.warning('Please set verbose to True to support TextStreamer, or use `inference_stream.`')
        stream = False
    streamer = None
    if stream:
        streamer = TextStreamer(tokenizer, skip_prompt=True)
    if verbose:
        if 'input_ids' in inputs:
            input_ids = inputs['input_ids']
            print(
                f'{prompt_prefix}{safe_tokenizer_decode(tokenizer, input_ids[0], **tokenizer_kwargs)}{output_prefix}',
                end='')
        elif 'query' in example:
            query = example['query']
            print(f'[QUERY]{query}\n{output_prefix}', end='')
    if tokenizer.eos_token_id is not None:
        generation_config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.bos_token_id is not None:
        generation_config.bos_token_id = tokenizer.bos_token_id
    if generation_config.max_new_tokens is not None:
        generation_config.max_length = 20  # fix max_length, max_new_tokens warning
        max_length = get_max_model_len(model.config)
        if max_length and token_len + generation_config.max_new_tokens > max_length:
            generation_config.max_new_tokens = max_length - token_len
            if generation_config.max_new_tokens <= 0:
                raise AssertionError('Current sentence length exceeds' f'the model max_length: {max_length}')
    if template.suffix[-1] not in stop_words:
        stop_words.append(template.suffix[-1])
    stopping_criteria = StoppingCriteriaList([StopWordsCriteria(tokenizer, stop_words, **tokenizer_kwargs)])
    inputs = to_device(inputs, device)
    if generation_info is not None:
        generation_info['num_prompt_tokens'] = token_len
    if 'inputs_embeds' in inputs:
        inputs.pop('input_ids', None)
    if adapter_names is not None:
        inputs['adapter_names'] = adapter_names
    generate_ids = model.generate(
        streamer=streamer, generation_config=generation_config, stopping_criteria=stopping_criteria, **inputs)
    generate_ids = template.get_generate_ids(generate_ids, token_len)
    if generation_info is not None:
        generation_info['num_generated_tokens'] = len(generate_ids)
    response = None
    if verbose and stream is False:
        response = tokenizer.decode(generate_ids, **tokenizer_kwargs)
        print(response)
    response = template.generate_ids_to_response(generate_ids, tokenizer_kwargs=tokenizer_kwargs)
    if not is_observation:
        history.append([query, response])
    else:
        history[-1][-1] = history[-1][-1] + response
    return response, history


def limit_history_length(template: Template, query: str, history: Optional[History],
                         max_length: Optional[int]) -> Tuple[History, History]:
    """binary search"""
    if history is None:
        history = []
    if max_length is None:
        return [], history

    def compute_token_length(history_length: int) -> int:
        assert history_length != 0
        example = {'query': query, 'history': history[-history_length:]}
        input_ids = template.encode(example)[0]['input_ids']
        return len(input_ids)

    history_length = upper_bound(0, len(history), lambda mid: compute_token_length(mid) <= max_length)
    old_history = history[:len(history) - history_length]
    history = history[len(history) - history_length:]
    return old_history, history


Messages = List[Dict[str, str]]


def history_to_messages(history: Optional[History],
                        query: Optional[str] = None,
                        system: Optional[str] = None) -> Messages:
    if history is None:
        history = []
    messages = []
    if system is not None:
        messages.append({'role': 'system', 'content': system})
    for h in history:
        assert isinstance(h, (list, tuple))
        messages.append({'role': 'user', 'content': h[0]})
        messages.append({'role': 'assistant', 'content': h[1]})
    if query is not None:
        messages.append({'role': 'user', 'content': query})
    return messages


def messages_to_history(messages: Messages) -> Dict[str, Any]:
    system = None
    if messages[0]['role'] == 'system':
        system = messages[0]['content']
        messages = messages[1::]
    history = []
    for q, r in zip(messages[::2], messages[1::2]):
        history.append([q['content'], r['content']])
    query = None
    if len(messages) % 2 == 1:
        query = messages[-1]['content']
    return {
        'history': history,
        'query': query,
        'system': system,
    }


def set_generation_config(model: Module, generation_config: GenerationConfig) -> None:
    old_generation_config = getattr(model, 'generation_config', None)
    if old_generation_config is not None:
        for k, v in old_generation_config.__dict__.items():
            if k not in generation_config.__dict__:
                setattr(generation_config, k, v)
    model.generation_config = generation_config


def is_vllm_available():
    return importlib.util.find_spec('vllm') is not None


def is_xtuner_available():
    return importlib.util.find_spec('xtuner') is not None


def is_unsloth_available() -> bool:
    return importlib.util.find_spec('unsloth') is not None


def get_time_info(log_history: List[Dict[str, Any]], n_train_samples: Optional[int]) -> Optional[Dict[str, Any]]:
    time_info = None
    try:
        last_log_history = log_history[-1]
        train_runtime = last_log_history['train_runtime']
        train_samples_per_second = n_train_samples / train_runtime
        time_info = {
            'train_runtime': train_runtime,
            'n_train_samples': n_train_samples,
            'train_samples_per_second': train_samples_per_second,
        }
    except Exception:
        pass
    return time_info


def get_max_model_len(config: PretrainedConfig) -> Optional[int]:
    INF = int(1e9)
    max_model_len = INF
    possible_keys = [
        'seq_length',  # qwen, chatglm
        'max_position_embeddings',  # qwen1.5, llama2
        'n_positions',  # polylm, phi-2
        'model_max_length',  # baichuan2
        # others
        'seq_len',
        'max_seq_len',
        'max_sequence_length',
        'max_seq_length',
    ]
    for key in possible_keys:
        max_len_key = getattr(config, key, None)
        if max_len_key is not None:
            max_model_len = min(max_model_len, max_len_key)
    if max_model_len == INF:
        max_model_len = None
    return max_model_len


if is_ddp_plus_mp():
    from accelerate.utils.modeling import (get_balanced_memory, infer_auto_device_map)

    @wraps(infer_auto_device_map)
    def _infer_auto_device_map_patch(model: Module,
                                     max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
                                     **kwargs) -> Dict[str, Union[int, str, Device]]:
        """The auxiliary function for supports DDP+MP. Monkey Patching.
        add feat in accelerate to support DDP + MP"""
        verbose = kwargs.pop('verbose', False)
        n_gpu = torch.cuda.device_count()
        _, local_rank, _, local_world_size = get_dist_setting()
        device_ids = list(range(local_rank, n_gpu, local_world_size))
        max_memory = _get_max_memory(device_ids)
        max_memory = _sync_max_memory(max_memory)
        max_memory = get_balanced_memory(model, max_memory, low_zero=False, **kwargs)
        max_memory = {k: v for k, v in max_memory.items() if v > 0}
        return infer_auto_device_map(model, max_memory, verbose=verbose, **kwargs)

    _old_ddp_init = DDP.__init__
    accelerate.accelerator.torch.nn.parallel.DistributedDataParallel.__init__ = (
        lambda self, model, device_ids, output_device, *args, **kwargs: _old_ddp_init(self, model, *args, **kwargs))
    transformers.modeling_utils.get_balanced_memory = lambda *args, **kwargs: None
    transformers.modeling_utils.infer_auto_device_map = _infer_auto_device_map_patch

if is_ddp_plus_mp() or use_torchacc():
    _old_accelerator_init = trainer.Accelerator.__init__
    trainer.Accelerator.__init__ = (lambda self, device_placement=False, *args, **kwargs: _old_accelerator_init(
        self, device_placement=device_placement, *args, **kwargs))
    trainer.Accelerator.verify_device_map = lambda *args, **kwargs: False
