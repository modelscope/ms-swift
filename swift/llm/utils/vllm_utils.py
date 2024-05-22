import asyncio
import inspect
import os
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import vllm
from modelscope import GenerationConfig
from packaging import version
from torch import dtype as Dtype
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from vllm import AsyncEngineArgs, AsyncLLMEngine, EngineArgs, LLMEngine, SamplingParams

from swift.utils import get_logger
from .argument import InferArguments
from .model import get_model_tokenizer
from .template import Template, get_template

try:
    from vllm.lora.request import LoRARequest
except ImportError:
    pass

logger = get_logger()


def get_vllm_engine(model_type: str,
                    torch_dtype: Optional[Dtype] = None,
                    *,
                    model_id_or_path: Optional[str] = None,
                    revision: Optional[str] = None,
                    gpu_memory_utilization: float = 0.9,
                    tensor_parallel_size: int = 1,
                    max_model_len: Optional[int] = None,
                    engine_kwargs: Optional[Dict[str, Any]] = None,
                    use_async: bool = False,
                    enable_lora: bool = False,
                    max_loras: int = 1,
                    max_lora_rank: int = 16,
                    **kwargs) -> LLMEngine:
    model_dir = kwargs.pop('model_dir', None)  # compat with swift<1.7
    tokenizer = get_model_tokenizer(
        model_type,
        load_model=False,
        model_id_or_path=model_id_or_path,
        model_dir=model_dir,
        revision=revision,
        download_model=True)[1]
    model_dir = tokenizer.model_dir

    if engine_kwargs is None:
        engine_kwargs = {}
    dtype_mapping = {torch.float16: 'float16', torch.bfloat16: 'bfloat16', torch.float32: 'float32', None: 'auto'}
    dtype = dtype_mapping[torch_dtype]
    disable_log_stats = engine_kwargs.pop('disable_log_stats', True)

    if use_async:
        engine_args_cls = AsyncEngineArgs
        llm_engine_cls = AsyncLLMEngine
        engine_kwargs['disable_log_requests'] = True
    else:
        engine_args_cls = EngineArgs
        llm_engine_cls = LLMEngine

    parameters = inspect.signature(engine_args_cls.__init__).parameters
    if 'enable_lora' in parameters and enable_lora:
        engine_kwargs['enable_lora'] = enable_lora
        engine_kwargs['max_loras'] = max_loras
        engine_kwargs['max_lora_rank'] = max_lora_rank
    else:
        assert not enable_lora, ('The current version of VLLM does not support `enable_lora`. Please upgrade VLLM.')

    engine_args = engine_args_cls(
        model=model_dir,
        trust_remote_code=True,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        disable_log_stats=disable_log_stats,
        **engine_kwargs)
    try:
        from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except ImportError:
        pass
    # fix HTTPError bug (use model_dir)
    os.environ.pop('VLLM_USE_MODELSCOPE', None)
    llm_engine = llm_engine_cls.from_engine_args(engine_args)
    llm_engine.engine_args = engine_args
    llm_engine.model_dir = model_dir
    llm_engine.model_type = model_type

    if use_async:
        _engine = llm_engine.engine
    else:
        _engine = llm_engine
    # compatible with vllm==0.3.*
    if version.parse(vllm.__version__) >= version.parse('0.3'):
        assert isinstance(_engine.tokenizer.tokenizer, PreTrainedTokenizerBase)
        _engine.tokenizer.tokenizer = tokenizer
    else:
        assert isinstance(_engine.tokenizer, PreTrainedTokenizerBase)
        _engine.tokenizer = tokenizer

    llm_engine.hf_tokenizer = tokenizer
    generation_config_path = os.path.join(model_dir, 'generation_config.json')
    if os.path.isfile(generation_config_path):
        generation_config = GenerationConfig.from_pretrained(model_dir)
        kwargs = generation_config.to_dict()
        parameters = inspect.signature(VllmGenerationConfig.__init__).parameters
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
        max_new_tokens: Optional[int] = 64,  # max_tokens
        temperature: float = 1.,
        top_k: int = 50,  # -1: all
        top_p: float = 1.,
        repetition_penalty: float = 1.,
        num_beams: int = 1,
        *,
        n: int = 1,
        length_penalty: float = 1.,
        stop: Optional[List[str]] = None,
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> None:
        # The parameter design is similar to transformers.GenerationConfig.
        if max_new_tokens is None:
            max_new_tokens = 64
        if num_beams > 1:
            top_k = -1
            top_p = 1
            temperature = 0
            logger.warning(
                'The output of num_beams in vllm may not be consistent with the output of num_beams in transformers.')
        if top_k == 0:
            top_k = -1
        if stop is None:
            stop = []
        kwargs['max_tokens'] = max_new_tokens
        kwargs['temperature'] = temperature
        kwargs['top_k'] = top_k
        kwargs['top_p'] = top_p
        kwargs['repetition_penalty'] = repetition_penalty
        if num_beams > 1:
            best_of = kwargs.get('best_of')
            assert 'use_beam_search' not in kwargs and best_of is None
            kwargs['use_beam_search'] = True
            kwargs['best_of'] = num_beams
        kwargs['n'] = n
        kwargs['length_penalty'] = length_penalty
        kwargs['stop'] = stop
        kwargs['skip_special_tokens'] = skip_special_tokens
        parameters = inspect.signature(SamplingParams.__init__).parameters
        for k in kwargs.copy().keys():
            if k not in parameters:
                logger.info(f'The VLLM version is too old and does not support the parameter: {k}.')
                kwargs.pop(k)
        self._temperature = temperature
        super().__init__(**kwargs)

    def __setattr__(self, key: str, value: str) -> None:
        if key == 'max_new_tokens':
            self.max_tokens = value
        elif key == 'do_sample':
            assert value in {True, False}
            if value:
                self.temperature = self._temperature
            else:
                self.temperature = 0.
        elif key == 'max_length':
            raise ValueError('`max_length` is not supported, please use `max_new_tokens` for setting.')
        else:
            super().__setattr__(key, value)


@torch.inference_mode()
def inference_stream_vllm(llm_engine: LLMEngine,
                          template: Template,
                          request_list: List[Dict[str, Any]],
                          *,
                          generation_config: Optional[VllmGenerationConfig] = None,
                          lora_request: Optional['LoRARequest'] = None,
                          use_tqdm: bool = False,
                          **kwargs) -> Iterator[List[Dict[str, Any]]]:
    """
    request_list: e.g. [{'query': 'hello!'}].
        The keys that can be included are: 'query', 'history', 'system'.
    generation_config: Priority: generation_config > model.generation_config.
    return: e.g. [{'response': 'hi!', 'history': [('hello!', 'hi!')]}].
        The keys to be included will be: 'response', 'history'.
    """
    if generation_config is None:
        generation_config = getattr(llm_engine, 'generation_config', VllmGenerationConfig())
    assert isinstance(generation_config, VllmGenerationConfig)
    request_list = deepcopy(request_list)
    generation_config = deepcopy(generation_config)
    if generation_config.use_beam_search is True:
        error_msg = 'Streaming generation does not support beam search.'
        raise ValueError(error_msg)

    tokenizer = template.tokenizer
    if tokenizer.eos_token is not None and tokenizer.eos_token not in generation_config.stop:
        generation_config.stop.append(tokenizer.eos_token)
    if isinstance(template.suffix[-1], str) and template.suffix[-1] not in generation_config.stop:
        generation_config.stop.append(template.suffix[-1])
    if isinstance(template.suffix[-1], list):
        token_str = tokenizer.decode(template.suffix[-1])
        if token_str not in generation_config.stop:
            generation_config.stop.append(token_str)

    parameters = inspect.signature(llm_engine.add_request).parameters
    add_request_kwargs = {}
    if 'lora_request' in parameters:
        add_request_kwargs['lora_request'] = lora_request
    else:
        assert lora_request is None, (
            'The current version of VLLM does not support `lora_request`. Please upgrade VLLM.')
    request_temp = []
    for i, request in enumerate(request_list):
        history = request.get('history', None)
        if history is None:
            history = []

        # agent support
        is_observation = history[-1][-1].endswith('Observation:') if history and history[-1][-1] else False
        act_length = None
        if is_observation:
            history[-1][-1] = history[-1][-1] + request['query']
            act_length = len(history[-1][-1])
            request['query'] = None
        request_temp.append((is_observation, act_length))

        request['history'] = history
        inputs = template.encode(request)[0]
        if len(inputs) == 0:
            raise ValueError('input_ids exceeds `max_length`. Please increase the value of `max_length`.')
        input_ids = inputs['input_ids']
        llm_engine.add_request(str(i), None, generation_config, input_ids, **add_request_kwargs)

    resp_list = [None] * len(request_list)
    print_idx_list = [[0] for _ in range(len(request_list))]
    prog_bar = tqdm(total=len(request_list), dynamic_ncols=True, disable=not use_tqdm)
    while llm_engine.has_unfinished_requests():
        step_outputs = llm_engine.step()
        for output in step_outputs:
            i = int(output.request_id)
            request = request_list[i]
            generate_ids = output.outputs[0].token_ids
            safe_response = template.generate_ids_to_response(
                generate_ids, output.finished, print_idx=print_idx_list[i])
            query = request['query']
            history = request['history']
            if resp_list[i] is None and not request_temp[i][0]:
                history.append(None)
            if not request_temp[i][0]:
                history[-1] = [query, safe_response]
            else:
                history[-1][-1] = history[-1][-1][:request_temp[i][1]] + safe_response
            resp_list[i] = {'response': safe_response, 'history': history}
            if output.finished:
                prog_bar.update()
        yield resp_list


@torch.inference_mode()
def inference_vllm(llm_engine: LLMEngine,
                   template: Template,
                   request_list: List[Dict[str, Any]],
                   *,
                   generation_config: Optional[VllmGenerationConfig] = None,
                   lora_request: Optional['LoRARequest'] = None,
                   use_tqdm: bool = False,
                   verbose: bool = False,
                   prompt_prefix: str = '[PROMPT]',
                   output_prefix: str = '[OUTPUT]',
                   **kwargs) -> List[Dict[str, Any]]:
    """
    request_list: e.g. [{'query': 'hello!'}].
        The keys that can be included are: 'query', 'history', 'system'.
    generation_config: Priority: generation_config > model.generation_config.
    return: e.g. [{'response': 'hi!', 'history': [('hello!', 'hi!')]}].
        The keys to be included will be: 'response', 'history'.
    """
    if generation_config is None:
        generation_config = getattr(llm_engine, 'generation_config', VllmGenerationConfig())
    assert isinstance(generation_config, VllmGenerationConfig)
    request_list = deepcopy(request_list)
    generation_config = deepcopy(generation_config)

    tokenizer = template.tokenizer
    if tokenizer.eos_token is not None and tokenizer.eos_token not in generation_config.stop:
        generation_config.stop.append(tokenizer.eos_token)
    if isinstance(template.suffix[-1], str) and template.suffix[-1] not in generation_config.stop:
        generation_config.stop.append(template.suffix[-1])
    if isinstance(template.suffix[-1], list):
        token_str = tokenizer.decode(template.suffix[-1])
        if token_str not in generation_config.stop:
            generation_config.stop.append(token_str)

    parameters = inspect.signature(llm_engine.add_request).parameters
    add_request_kwargs = {}
    if 'lora_request' in parameters:
        add_request_kwargs['lora_request'] = lora_request
    else:
        assert lora_request is None, (
            'The current version of VLLM does not support `lora_request`. Please upgrade VLLM.')

    for i, request in enumerate(request_list):
        history = request.get('history', None)
        if history is None:
            history = []

        is_observation = history[-1][-1].endswith('Observation:') if history and history[-1][-1] else False
        if is_observation:
            history[-1][-1] = history[-1][-1] + request['query']
            request['query'] = None
        request['history'] = history
        inputs = template.encode(request)[0]
        if len(inputs) == 0:
            raise ValueError('input_ids exceeds `max_length`. Please increase the value of `max_length`.')
        input_ids = inputs['input_ids']
        llm_engine.add_request(str(i), None, generation_config, input_ids, **add_request_kwargs)

    if use_tqdm is True:
        assert verbose is False
    prog_bar = tqdm(total=len(request_list), dynamic_ncols=True, disable=not use_tqdm)
    outputs = []
    while llm_engine.has_unfinished_requests():
        step_outputs = llm_engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                prog_bar.update()

    resp_list = [None] * len(request_list)
    for output in outputs:
        i = int(output.request_id)
        request = request_list[i]
        generate_ids = output.outputs[0].token_ids
        response = template.generate_ids_to_response(generate_ids)
        query = request['query']
        history = request['history']
        if not is_observation:
            history.append([query, response])
        else:
            history[-1][-1] = history[-1][-1] + response
        resp_list[i] = {'response': response, 'history': history}
        if verbose:
            print(f'{prompt_prefix}{tokenizer.decode(output.prompt_token_ids, False)}{output_prefix}', end='')
            print(tokenizer.decode(output.outputs[0].token_ids, False))
    return resp_list


def prepare_vllm_engine_template(args: InferArguments, use_async: bool = False) -> Tuple[LLMEngine, Template]:
    logger.info(f'device_count: {torch.cuda.device_count()}')

    assert args.quantization_bit == 0, 'not support bnb'
    assert not (args.sft_type == 'lora' and not args.vllm_enable_lora), 'you need to merge lora'
    # Loading Model and Tokenizer
    model_id_or_path = None
    if args.sft_type == 'full' and args.ckpt_dir is not None:
        model_id_or_path = args.ckpt_dir
    elif args.model_id_or_path is not None:
        model_id_or_path = args.model_id_or_path
    llm_engine = get_vllm_engine(
        args.model_type,
        args.torch_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        use_async=use_async,
        model_id_or_path=model_id_or_path,
        enable_lora=args.vllm_enable_lora,
        max_loras=min(len(args.lora_modules), 1),
        max_lora_rank=args.vllm_max_lora_rank)
    tokenizer = llm_engine.hf_tokenizer
    if use_async:
        model_config = asyncio.run(llm_engine.get_model_config())
        llm_engine.model_config = model_config
    else:
        model_config = llm_engine.model_config
    logger.info(f'model_config: {model_config.hf_config}')

    if not args.do_sample:
        args.temperature = 0
    generation_config = VllmGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop=args.stop_words,
        repetition_penalty=args.repetition_penalty,
        num_beams=args.num_beams)
    logger.info(f'generation_config: {generation_config}')
    llm_engine.generation_config = generation_config
    template: Template = get_template(args.template_type, tokenizer, args.system, args.max_length,
                                      args.truncation_strategy)
    args.system = template.default_system
    logger.info(f'system: {args.system}')
    return llm_engine, template
