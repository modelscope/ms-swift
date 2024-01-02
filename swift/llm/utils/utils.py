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
from typing import (Any, Callable, Dict, Iterator, List, Optional, Tuple,
                    TypeVar, Union)

import accelerate
import multiprocess
import numpy as np
import requests
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from accelerate.utils.modeling import (get_balanced_memory,
                                       infer_auto_device_map)
from datasets import Dataset as HfDataset
from modelscope import MsDataset
from modelscope.utils.config_ds import MS_CACHE_HOME
from modelscope.utils.logger import get_logger as get_ms_logger
from torch import device as Device
from torch.nn import Linear, Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import (GenerationConfig, PreTrainedModel,
                          PreTrainedTokenizerBase, StoppingCriteriaList,
                          TextStreamer, trainer)

from swift.hub import ModelScopeConfig
from swift.utils import (get_dist_setting, get_logger, is_ddp_plus_mp, is_dist,
                         is_local_master, is_master, stat_array, upper_bound)
from .template import (History, StopWords, StopWordsCriteria, Template,
                       get_audio_info)

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


def download_dataset(model_id: str,
                     files: List[str],
                     force_download: bool = False) -> str:
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


_old_msdataset_load = MsDataset.load


@wraps(_old_msdataset_load)
def _msdataset_ddp_load(*args, **kwargs):
    if is_dist() and not is_local_master():
        dist.barrier()
    dataset = _old_msdataset_load(*args, **kwargs)
    if is_dist() and is_local_master():
        dist.barrier()

    if is_dist():
        dist.barrier()
    return dataset


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


def _sync_max_memory(
        max_memory: Dict[Union[int, str], int]) -> Dict[Union[int, str], int]:
    """Make sure that the model structure of MP(device_map) is the same, when using DDP."""
    max_memory_list = [
        v for k, v in max_memory.items() if (v > 0 and k != 'cpu')
    ]
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


@wraps(infer_auto_device_map)
def _infer_auto_device_map_patch(
        model: Module,
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
    max_memory = get_balanced_memory(
        model, max_memory, low_zero=False, **kwargs)
    max_memory = {k: v for k, v in max_memory.items() if v > 0}
    return infer_auto_device_map(model, max_memory, verbose=verbose, **kwargs)


class LLMDataset(Dataset):

    def __init__(self, data: List[Dict[str, Any]]) -> None:
        self.data = data

    def __getitem__(self, idx: Union[int, str]) -> Dict[str, Any]:
        if isinstance(idx, int):
            return self.data[idx]
        elif isinstance(idx, str):
            return [d[idx] for d in self.data]
        else:
            raise ValueError(f'idx: {idx}')

    def select(self, idx_list: List[int]) -> 'LLMDataset':
        new_data = np.array(self.data)
        new_data = new_data[idx_list].tolist()
        return self.__class__(new_data)

    def __len__(self) -> int:
        return len(self.data)


class LazyLLMDataset(Dataset):

    def __init__(self,
                 dataset: HfDataset,
                 template: Template,
                 *,
                 try_fetch_time: int = 20) -> None:
        self.dataset = dataset
        self.template = template
        self.try_fetch_time = min(try_fetch_time, len(self.dataset))
        assert self.try_fetch_time >= 1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        res = self._try_fetch(idx)
        if res is not None:
            return res
        raise ValueError('Please check if the max_length is appropriate.')

    def _try_fetch(self, first_idx: int) -> Optional[Dict[str, Any]]:
        idx = np.random.permutation(len(self))[:self.try_fetch_time - 1]
        for i in [first_idx] + idx.tolist():
            data = self.dataset[i]
            res = self.template.encode(data)
            if res is not None:
                return res

    def __len__(self) -> int:
        return len(self.dataset)


MapFunc = Callable[[Dict[str, Any]], Dict[str, Any]]


def _single_map(d: Dict[str, Any],
                map_func: MapFunc) -> Optional[Dict[str, Any]]:
    d = map_func(d)
    if d is None:
        return None
    audio_info = d.get('audio_info')
    if audio_info is not None:
        audio_info.pop('input_audios', None)
    return d


def _map_mp_single(subset: HfDataset, map_func: MapFunc, queue: Queue,
                   start_idx: int):
    for i, d in enumerate(subset, start=start_idx):
        queue.put((i, map_func(d)))  # idx, result


def _map_mp_i(dataset: HfDataset, map_func: MapFunc,
              num_proc: int) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with multiprocess.Pool(
            num_proc) as pool, multiprocess.Manager() as manager:
        queue = manager.Queue()
        async_results = []
        split_idx = np.linspace(0, len(dataset), num_proc + 1, dtype=np.int32)
        for i in range(num_proc):
            subset = dataset.select(range(split_idx[i], split_idx[i + 1]))
            async_results.append(
                pool.apply_async(
                    _map_mp_single,
                    args=(subset, map_func, queue, split_idx[i])))
        while True:
            try:
                yield queue.get(timeout=0.05)
            except Empty:
                if all(async_result.ready()
                       for async_result in async_results) and queue.empty():
                    break


def _map_mp(dataset: HfDataset, map_func: MapFunc,
            num_proc: int) -> List[Dict[str, Any]]:
    # Solving the unordered problem
    data = [None] * len(dataset)
    num_proc = min(num_proc, len(dataset))
    for d in tqdm(_map_mp_i(dataset, map_func, num_proc), total=len(dataset)):
        data[d[0]] = d[1]
    return data


def dataset_map(dataset: HfDataset,
                map_func: MapFunc,
                num_proc: int = 1) -> LLMDataset:
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
        logger.info('len(dataset): 0')
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


def data_collate_fn(batch: List[Dict[str, Any]],
                    tokenizer: PreTrainedTokenizerBase,
                    padding_to: Optional[int] = None) -> Dict[str, Any]:
    """
    Args:
        batch(`List[Dict[str, Any]]`): The input data in batch
        tokenizer(`PreTrainedTokenizerBase`): The tokenizer of the model
        padding_to(`int`, optional): Whether padding the batch to a fixed length, if none, the batch
            will be padded to the `longest`
    """
    assert tokenizer.pad_token_id is not None
    input_ids = [torch.tensor(b['input_ids']) for b in batch]
    labels = [torch.tensor(b['labels']) for b in batch]
    attention_mask = [
        torch.ones(len(input_ids[i]), dtype=torch.int64)
        for i in range(len(input_ids))
    ]

    if padding_to is not None and padding_to > input_ids[0].shape[-1]:
        input_ids[0] = F.pad(input_ids[0],
                             (0, padding_to - input_ids[0].shape[-1]),
                             'constant', tokenizer.pad_token_id)
        labels[0] = F.pad(labels[0], (0, padding_to - labels[0].shape[-1]),
                          'constant', -100)
        attention_mask[0] = F.pad(
            attention_mask[0], (0, padding_to - attention_mask[0].shape[-1]),
            'constant', 0)

    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(
        attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    res = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }
    if batch[0].get('audio_info') is not None:
        res['audio_info'] = [
            get_audio_info(tokenizer, audio_info=b['audio_info'])
            for b in batch
        ]
    if batch[0].get('images') is not None:
        res['images'] = [b['images'] for b in batch]
    if batch[0].get('cross_images') is not None:
        res['cross_images'] = [b['cross_images'] for b in batch]
    if batch[0].get('token_type_ids') is not None:
        res['token_type_ids'] = torch.stack(
            [b['token_type_ids'] for b in batch])
    return res


def _get_labels_str(labels: List[int], tokenizer: PreTrainedTokenizerBase,
                    **decode_kwargs) -> str:
    if len(labels) == 0:
        return ''
    labels_str = ''
    for i in range(len(labels)):
        if i == 0:
            if labels[i] == -100:
                s = 0
            else:
                e = 0
            continue
        if labels[i] == -100 and labels[i - 1] != -100:
            s = i
            labels_str += tokenizer.decode(labels[e:s], **decode_kwargs)
        if labels[i] != -100 and labels[i - 1] == -100:
            e = i
            labels_str += f'[-100 * {e - s}]'
    if labels[-1] == -100:
        labels_str += f'[-100 * {len(labels) - s}]'
    else:
        labels_str += tokenizer.decode(labels[e:], **decode_kwargs)
    return labels_str


def print_example(example: Dict[str, Any],
                  tokenizer: PreTrainedTokenizerBase) -> None:
    input_ids, labels = example['input_ids'], example.get('labels')
    logger.info(f'[INPUT_IDS] {input_ids}')
    decode_kwargs = {}
    # Compatible with qwen-audio
    if 'audio_info' in example:
        decode_kwargs['audio_info'] = example['audio_info']
    logger.info(f'[INPUT] {tokenizer.decode(input_ids, **decode_kwargs)}')
    if labels is not None:
        logger.info(f'[LABLES_IDS] {labels}')
        labels_str = _get_labels_str(labels, tokenizer, **decode_kwargs)
        logger.info(f'[LABLES] {labels_str}')


def find_all_linear_for_lora(model: Module, quantization_bit: int,
                             model_type: str) -> List[str]:
    """ref: https://github.com/artidoro/qlora"""
    head_module_name = 'lm_head'
    if model_type.startswith('chatglm'):
        head_module_name = 'output_layer'
    if quantization_bit == 4:
        from bitsandbytes.nn import Linear4bit
        linear_cls = Linear4bit
    elif quantization_bit == 8:
        from bitsandbytes.nn import Linear8bitLt
        linear_cls = Linear8bitLt
    else:
        linear_cls = Linear
    if 'int4' in model_type or 'int8' in model_type:
        from bitsandbytes.nn import Linear4bit
        from peft.utils import get_auto_gptq_quant_linear, get_quantization_config
        gptq_quantization_config = get_quantization_config(model, 'gptq')
        AutoGPTQQuantLinear = get_auto_gptq_quant_linear(
            gptq_quantization_config)
        linear_cls = Linear4bit
        if AutoGPTQQuantLinear is not None:
            linear_cls = (Linear4bit, AutoGPTQQuantLinear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls):
            module_name = name.split('.')[-1]
            if head_module_name not in module_name:
                lora_module_names.add(module_name)
    return list(lora_module_names)


def sort_by_max_length(llm_dataset: LLMDataset, num_dataset: int) -> HfDataset:
    logger.info('sort by max length...')
    dataset_len = [len(d['input_ids']) for d in llm_dataset]
    idx = heapq.nlargest(
        num_dataset, range(len(dataset_len)), key=lambda i: dataset_len[i])
    return llm_dataset.select(idx)


def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # copy from transformers.generation.streamers.TextStreamer
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or (cp >= 0x3400 and cp <= 0x4DBF)
            or (cp >= 0x20000 and cp <= 0x2A6DF)
            or (cp >= 0x2A700 and cp <= 0x2B73F)
            or (cp >= 0x2B740 and cp <= 0x2B81F)
            or (cp >= 0x2B820 and cp <= 0x2CEAF)
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)):
        return True

    return False


def inference_stream(model: PreTrainedModel,
                     template: Template,
                     query: str,
                     history: Optional[History] = None,
                     system: Optional[str] = None,
                     *,
                     generation_config: Optional[GenerationConfig] = None,
                     stop_words: Optional[List[StopWords]] = None,
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
    example = {'query': query, 'history': history, 'system': system}
    image = kwargs.pop('image', None)
    if image is not None:
        example['image'] = image
    inputs = template.encode(example)
    audio_info = inputs.get('audio_info')  # Compatible with qwen-audio
    input_ids = inputs['input_ids']
    tokenizer = template.tokenizer
    device = next(model.parameters()).device
    input_ids = torch.tensor(input_ids)[None].to(device)
    if 'attention_mask' not in inputs:
        attention_mask = torch.ones_like(input_ids).to(device)
    else:
        attention_mask = inputs['attention_mask'].to(device)
    model.eval()
    if generation_config is None:
        generation_config = getattr(model, 'generation_config', None)
    generation_config = deepcopy(generation_config)
    if generation_config.num_beams != 1:
        error_msg = 'Streaming generation does not support beam search.'
        raise ValueError(error_msg)
    from transformers_stream_generator.main import NewGenerationMixin, StreamGenerationConfig
    model.__class__.generate_stream = NewGenerationMixin.generate
    model.__class__.sample_stream = NewGenerationMixin.sample_stream
    stream_config = StreamGenerationConfig(
        **generation_config.to_dict(), do_stream=True)
    if tokenizer.eos_token_id is not None:
        stream_config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        stream_config.pad_token_id = tokenizer.pad_token_id
    if stream_config.max_new_tokens is not None:
        stream_config.max_length = 20  # fix max_length, max_new_tokens warning
    stream_config.do_sample = True  # avoid is_greedy_gen_mode = True
    if template.suffix[-1] not in stop_words:
        stop_words.append(template.suffix[-1])
    decode_kwargs = {}
    model_kwargs = {}
    # Compatible with cogagent
    if 'token_type_ids' in inputs:
        model_kwargs['token_type_ids'] = inputs['token_type_ids'].to(device)
    if 'images' in inputs:
        model_kwargs['images'] = [[
            inputs['images'][0][0].to(device).to(torch.float16)
        ]]
    if 'cross_images' in inputs:
        model_kwargs['cross_images'] = [[
            inputs['cross_images'][0][0].to(device).to(torch.float16)
        ]]
    # Compatible with qwen-audio
    if audio_info is not None:
        audio_info = get_audio_info(tokenizer, audio_info=audio_info)
        decode_kwargs['audio_info'] = audio_info
        model_kwargs['audio_info'] = audio_info
    stopping_criteria = StoppingCriteriaList(
        [StopWordsCriteria(tokenizer, stop_words, **decode_kwargs)])
    gen = model.generate_stream(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=stream_config,
        stopping_criteria=stopping_criteria,
        **model_kwargs,
        seed=-1)
    generate_ids = []
    response = ''
    print_idx = 0
    history.append(None)  # dummy
    for token in gen:
        generate_ids.append(token.item())
        response = tokenizer.decode(generate_ids, True, **decode_kwargs)
        if response.endswith('\n') or len(response) > 0 and _is_chinese_char(
                ord(response[-1])):
            print_idx = len(response)
        else:
            print_idx = max(response.rfind(' ') + 1, print_idx)
        # avoid printing incomplete words
        safe_response = response[:print_idx]
        history[-1] = (query, safe_response)
        yield safe_response, history
    history[-1] = (query, response)
    yield response, history


def inference(model: PreTrainedModel,
              template: Template,
              query: str,
              history: Optional[History] = None,
              system: Optional[str] = None,
              *,
              generation_config: Optional[GenerationConfig] = None,
              stop_words: Optional[List[StopWords]] = None,
              stream: bool = False,
              verbose: bool = False,
              prompt_prefix: str = '[PROMPT]',
              output_prefix: str = '[OUTPUT]',
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
    example = {'query': query, 'history': history, 'system': system}
    inputs = template.encode(example)
    audio_info = inputs.get('audio_info')  # Compatible with qwen-audio
    input_ids = inputs['input_ids']
    tokenizer = template.tokenizer
    device = next(model.parameters()).device
    input_ids = torch.tensor(input_ids)[None].to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    model.eval()
    if generation_config is None:
        generation_config = getattr(model, 'generation_config', None)
    generation_config = deepcopy(generation_config)
    if stream is True and verbose is False:
        logger.warning(
            'Please set verbose to True to support TextStreamer, or use `inference_stream.`'
        )
        stream = False
    streamer = None
    decode_kwargs = {}
    model_kwargs = {}
    if audio_info is not None:
        audio_info = get_audio_info(tokenizer, audio_info=audio_info)
        decode_kwargs['audio_info'] = audio_info
        model_kwargs['audio_info'] = audio_info
    if stream:
        streamer = TextStreamer(tokenizer, skip_prompt=True)
    if verbose:
        print(
            f'{prompt_prefix}{tokenizer.decode(input_ids[0], False, **decode_kwargs)}{output_prefix}',
            end='')
    if tokenizer.eos_token_id is not None:
        generation_config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        generation_config.pad_token_id = tokenizer.pad_token_id
    if generation_config.max_new_tokens is not None:
        generation_config.max_length = 20  # fix max_length, max_new_tokens warning
    if template.suffix[-1] not in stop_words:
        stop_words.append(template.suffix[-1])
    stopping_criteria = StoppingCriteriaList(
        [StopWordsCriteria(tokenizer, stop_words, **decode_kwargs)])
    generate_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        generation_config=generation_config,
        stopping_criteria=stopping_criteria,
        **model_kwargs)
    response = tokenizer.decode(generate_ids[0, len(input_ids[0]):], True,
                                **decode_kwargs)
    if verbose and stream is False:
        print(
            tokenizer.decode(generate_ids[0, len(input_ids[0]):], False,
                             **decode_kwargs))
    history.append((query, response))
    return response, history


def limit_history_length(template: Template, query: str,
                         history: Optional[History], max_length: int) -> int:
    """binary search"""
    if history is None:
        history = []

    def compute_token_length(history_length: int) -> int:
        assert history_length != 0
        example = {'query': query, 'history': history[-history_length:]}
        input_ids = template.encode(example)['input_ids']
        return len(input_ids)

    history_length = upper_bound(
        0, len(history), lambda mid: compute_token_length(mid) <= max_length)
    old_history = history[:len(history) - history_length]
    history = history[len(history) - history_length:]
    return old_history, history


Messages = List[Dict[str, str]]


def history_to_messages(history: History,
                        query: Optional[str] = None,
                        system: Optional[str] = None) -> Messages:
    messages = []
    if system is not None:
        messages.append({'role': 'system', 'content': system})
    for h in history:
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


def set_generation_config(model: Module,
                          generation_config: GenerationConfig) -> None:
    if hasattr(model, 'generation_config'):
        old_generation_config = model.generation_config
        for k, v in old_generation_config.__dict__.items():
            if k not in generation_config.__dict__:
                setattr(generation_config, k, v)
    model.generation_config = generation_config


def fix_fp16_trainable_bug(model: Module) -> None:
    # fix peft==0.7 bug
    is_logging = False
    for p in model.parameters():
        if p.requires_grad and p.dtype == torch.float16:
            if not is_logging:
                logger.info('Convert trainable parameters from fp16 to fp32.')
                is_logging = True
            p.data = p.data.to(dtype=torch.float32)


def is_vllm_available():
    return importlib.util.find_spec('vllm') is not None


def get_time_info(log_history: List[Dict[str, Any]],
                  n_train_samples: Optional[int]) -> Optional[Dict[str, Any]]:
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


# monkey patching
MsDataset.load = _msdataset_ddp_load
if is_ddp_plus_mp():
    _old_ddp_init = DDP.__init__
    accelerate.accelerator.torch.nn.parallel.DistributedDataParallel.__init__ = (
        lambda self, model, device_ids, output_device, *args, **kwargs:
        _old_ddp_init(self, model, *args, **kwargs))
    transformers.modeling_utils.get_balanced_memory = lambda *args, **kwargs: None
    transformers.modeling_utils.infer_auto_device_map = _infer_auto_device_map_patch
    _old_accelerator_init = trainer.Accelerator.__init__
    trainer.Accelerator.__init__ = (
        lambda self, device_placement=False, *args, **kwargs:
        _old_accelerator_init(
            self, device_placement=device_placement, *args, **kwargs))
    trainer.Accelerator.verify_device_map = lambda *args, **kwargs: False
