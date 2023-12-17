import inspect
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

import torch
from modelscope import GenerationConfig, snapshot_download
from torch import dtype as Dtype
from vllm import EngineArgs, LLMEngine, SamplingParams

from swift.utils import get_logger
from .model import MODEL_MAPPING, get_model_tokenizer
from .template import Template
from .utils import _is_chinese_char

logger = get_logger()


def get_vllm_engine(model_type: str,
                    torch_dtype: Optional[Dtype] = None,
                    gpu_memory_utilization: float = 0.9,
                    tensor_parallel_size: int = 1,
                    pipeline_parallel_size: int = 1,
                    engine_kwargs: Optional[Dict[str, Any]] = None,
                    **kwargs) -> LLMEngine:
    if engine_kwargs is None:
        engine_kwargs = {}
    model_info = MODEL_MAPPING[model_type]
    model_id_or_path = model_info['model_id_or_path']
    ignore_file_pattern = model_info['ignore_file_pattern']
    model_dir = kwargs.get('model_dir', None)
    if model_dir is None:
        model_dir = model_id_or_path
        if model_id_or_path is not None and not os.path.exists(
                model_id_or_path):
            revision = model_info['revision']
            model_dir = snapshot_download(
                model_id_or_path,
                revision,
                ignore_file_pattern=ignore_file_pattern)
    model_dir = os.path.expanduser(model_dir)
    assert os.path.isdir(model_dir)

    dtype_mapping = {
        torch.float16: 'float16',
        torch.bfloat16: 'bfloat16',
        torch.float32: 'float32',
        None: 'auto'
    }
    disable_log_stats = engine_kwargs.pop('disable_log_stats', True)
    engine_args = EngineArgs(
        model=model_dir,
        trust_remote_code=True,
        dtype=dtype_mapping[torch_dtype],
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        disable_log_stats=disable_log_stats,
        **engine_kwargs)
    llm_engine = LLMEngine.from_engine_args(engine_args)
    llm_engine.model_dir = model_dir
    llm_engine.model_type = model_type
    llm_engine.tokenizer = get_model_tokenizer(model_type, load_model=False)[1]
    generation_config_path = os.path.join(model_dir, 'generation_config.json')
    if os.path.isfile(generation_config_path):
        generation_config = GenerationConfig.from_pretrained(model_dir)
        kwargs = generation_config.to_dict()
        parameters = inspect.signature(
            VllmGenerationConfig.__init__).parameters
        for k in kwargs.copy().keys():
            if k not in parameters:
                kwargs.pop(k)
        llm_engine.generation_config = VllmGenerationConfig(**kwargs)
    else:
        llm_engine.generation_config = VllmGenerationConfig()
    return llm_engine


class VllmGenerationConfig(SamplingParams):

    def __init__(
        self,
        max_length: int = 20,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.,
        top_k: int = 50,  # -1: all
        top_p: float = 1.0,
        repetition_penalty: float = 1.,
        length_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs,
    ):
        # The parameter design is similar to transformers.GenerationConfig.
        if top_k == 0:
            top_k = -1
        self.max_new_tokens = max_new_tokens
        kwargs['max_tokens'] = max_length
        kwargs['temperature'] = temperature
        kwargs['top_k'] = top_k
        kwargs['top_p'] = top_p
        kwargs['repetition_penalty'] = repetition_penalty
        kwargs['length_penalty'] = length_penalty
        kwargs['stop'] = stop
        parameters = inspect.signature(SamplingParams.__init__).parameters
        for k in kwargs.copy().keys():
            if k not in parameters:
                logger.info(
                    f'The VLLM version is too old and does not support the parameter: {k}.'
                )
                kwargs.pop(k)
        super().__init__(**kwargs)

    @property
    def max_length(self) -> int:
        return self.max_tokens

    @max_length.setter
    def max_length(self, value: int) -> None:
        self.max_tokens = value


def inference_stream_vllm(
    llm_engine: LLMEngine,
    template: Template,
    request_list: List[Dict[str, Any]],
    *,
    generation_config: Optional[VllmGenerationConfig] = None
) -> List[Dict[str, Any]]:
    """
    request_list: e.g. [{'query': 'hello!'}].
        The keys that can be included are: 'query', 'history', 'system'.
    generation_config: Priority: generation_config > model.generation_config.
    return: e.g. [{'response': 'hi!', 'history': [('hello!', 'hi!')]}].
        The keys to be included will be: 'response', 'history'.
    """
    if generation_config is None:
        generation_config = getattr(llm_engine, 'generation_config',
                                    VllmGenerationConfig())
    assert isinstance(generation_config, VllmGenerationConfig)
    request_list = deepcopy(request_list)
    generation_config = deepcopy(generation_config)
    for i, request in enumerate(request_list):
        history = request.get('history', None)
        if history is None:
            history = []
        request['history'] = history
        inputs = template.encode(request)
        input_ids = inputs['input_ids']
        tokenizer = template.tokenizer
        if tokenizer.eos_token is not None and tokenizer.eos_token not in generation_config.stop:
            generation_config.stop.append(tokenizer.eos_token)
        if generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + len(
                input_ids)
        llm_engine.add_request(str(i), None, generation_config, input_ids)

    batch_size = len(request_list)
    response_list = [None] * batch_size
    while llm_engine.has_unfinished_requests():
        step_outputs = llm_engine.step()
        for output in step_outputs:
            i = int(output.request_id)
            request = request_list[i]
            response = tokenizer.decode(output.outputs[0].token_ids, True)
            if output.finished or response.endswith(
                    '\n') or len(response) > 0 and _is_chinese_char(
                        ord(response[-1])):
                print_idx = len(response)
            else:
                print_idx = max(response.rfind(' ') + 1, print_idx)
            # avoid printing incomplete words
            safe_response = response[:print_idx]
            query = request['query']
            history = request['history']
            if response_list[i] is None:
                history.append(None)
            history[-1] = (query, safe_response)
            response_list[i] = {'response': safe_response, 'history': history}
        yield response_list


def inference_vllm(llm_engine: LLMEngine,
                   template: Template,
                   request_list: List[Dict[str, Any]],
                   *,
                   generation_config: Optional[VllmGenerationConfig] = None,
                   verbose: bool = False,
                   prompt_prefix: str = '[PROMPT]',
                   output_prefix: str = '[OUTPUT]') -> List[Dict[str, Any]]:
    """
    request_list: e.g. [{'query': 'hello!'}].
        The keys that can be included are: 'query', 'history', 'system'.
    generation_config: Priority: generation_config > model.generation_config.
    return: e.g. [{'response': 'hi!', 'history': [('hello!', 'hi!')]}].
        The keys to be included will be: 'response', 'history'.
    """
    if generation_config is None:
        generation_config = getattr(llm_engine, 'generation_config',
                                    VllmGenerationConfig())
    assert isinstance(generation_config, VllmGenerationConfig)
    request_list = deepcopy(request_list)
    generation_config = deepcopy(generation_config)
    for i, request in enumerate(request_list):
        history = request.get('history', None)
        if history is None:
            history = []
        request['history'] = history
        inputs = template.encode(request)
        input_ids = inputs['input_ids']
        tokenizer = template.tokenizer
        if tokenizer.eos_token is not None and tokenizer.eos_token not in generation_config.stop:
            generation_config.stop.append(tokenizer.eos_token)
        if generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + len(
                input_ids)
        llm_engine.add_request(str(i), None, generation_config, input_ids)
    outputs = []
    while llm_engine.has_unfinished_requests():
        step_outputs = llm_engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)

    batch_size = len(request_list)
    response_list = [None] * batch_size
    for output in outputs:
        i = int(output.request_id)
        request = request_list[i]
        response = tokenizer.decode(output.outputs[0].token_ids, True)
        query = request['query']
        history = request['history']
        history.append((query, response))
        response_list[i] = {'response': response, 'history': history}
        if verbose:
            print(
                f'{prompt_prefix}{tokenizer.decode(output.prompt_token_ids, False)}{output_prefix}',
                end='')
            print(tokenizer.decode(output.outputs[0].token_ids, False))
    return response_list
