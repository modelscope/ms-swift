import concurrent.futures
import inspect
import os
import time
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import vllm
from packaging import version
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, PreTrainedTokenizerBase
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


@contextmanager
def _patch_auto_tokenizer(tokenizer):
    _old_from_pretrained = AutoTokenizer.from_pretrained

    @wraps(_old_from_pretrained)
    def _from_pretrained(self, *args, **kwargs):
        return tokenizer

    AutoTokenizer.from_pretrained = _from_pretrained
    yield
    AutoTokenizer.from_pretrained = _old_from_pretrained


def get_vllm_engine(
        model_type: str,
        torch_dtype: Optional[torch.dtype] = None,
        *,
        model_id_or_path: Optional[str] = None,
        revision: Optional[str] = None,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        max_num_seqs: int = 256,
        max_model_len: Optional[int] = None,
        disable_custom_all_reduce: bool = True,  # Default values different from vllm
        enforce_eager: bool = False,
        limit_mm_per_prompt: Optional[Dict[str, Any]] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        use_async: bool = False,
        # lora
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
        assert not enable_lora, 'The current version of VLLM does not support `enable_lora`. Please upgrade VLLM.'

    if 'limit_mm_per_prompt' in parameters and limit_mm_per_prompt:
        engine_kwargs['limit_mm_per_prompt'] = limit_mm_per_prompt
    else:
        assert not limit_mm_per_prompt, (
            'The current version of VLLM does not support `limit_mm_per_prompt`. Please upgrade VLLM.')

    engine_args = engine_args_cls(
        model=model_dir,
        trust_remote_code=True,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        disable_log_stats=disable_log_stats,
        disable_custom_all_reduce=disable_custom_all_reduce,
        enforce_eager=enforce_eager,
        **engine_kwargs)
    try:
        from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except ImportError:
        pass
    # fix HTTPError bug (use model_dir)
    os.environ.pop('VLLM_USE_MODELSCOPE', None)
    if version.parse(vllm.__version__) >= version.parse('0.5.1'):
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    with _patch_auto_tokenizer(tokenizer):
        llm_engine = llm_engine_cls.from_engine_args(engine_args)
    llm_engine.engine_args = engine_args
    llm_engine.model_dir = model_dir
    llm_engine.model_type = model_type

    if use_async:
        _engine = llm_engine.engine
    else:
        _engine = llm_engine
    model_config = _engine.model_config
    llm_engine.model_config = model_config
    llm_engine.dtype = model_config.dtype  # compat with pt
    llm_engine.max_model_len = model_config.max_model_len
    llm_engine.is_multimodal = tokenizer.is_multimodal
    # compatible with vllm==0.3.*
    if version.parse(vllm.__version__) >= version.parse('0.3'):
        assert isinstance(_engine.tokenizer.tokenizer, PreTrainedTokenizerBase)
        _engine.tokenizer.tokenizer = tokenizer

        # fix vllm==0.4 bug (very slow)
        if version.parse(vllm.__version__) >= version.parse('0.4'):
            _tokenizer_len = len(tokenizer)
            __old_len__ = tokenizer.__class__.__len__

            def __len__(self) -> int:
                if self is tokenizer:
                    return _tokenizer_len
                else:
                    return __old_len__(self)

            tokenizer.__class__.__len__ = __len__

    else:
        assert isinstance(_engine.tokenizer, PreTrainedTokenizerBase)
        _engine.tokenizer = tokenizer

    llm_engine.hf_tokenizer = tokenizer
    generation_config_path = os.path.join(model_dir, 'generation_config.json')
    if os.path.isfile(generation_config_path):
        generation_config = GenerationConfig.from_pretrained(model_dir)
        kwargs = generation_config.to_dict()
        max_new_tokens = kwargs.get('max_new_tokens')
        if max_new_tokens is not None:
            kwargs['max_tokens'] = max_new_tokens
        if version.parse(vllm.__version__) < version.parse('0.5.5'):
            parameters = inspect.signature(VllmGenerationConfig.__init__).parameters
        else:
            parameters = VllmGenerationConfig.__annotations__
        for k, v in kwargs.copy().items():
            if k not in parameters or v is None:
                kwargs.pop(k)
        llm_engine.generation_config = VllmGenerationConfig(**kwargs)
    else:
        llm_engine.generation_config = VllmGenerationConfig()
    return llm_engine


class _VllmGenerationConfigMixin:

    def __setattr__(self, key: str, value: str) -> None:
        if key == 'max_new_tokens':
            self.max_tokens = value
        elif key == 'do_sample' and hasattr(self, '_temperature'):
            assert value in {True, False}
            super().__setattr__('temperature', self._temperature if value else 0)
        elif key == 'max_length':
            raise ValueError('`max_length` is not supported, please use `max_new_tokens` for setting.')
        else:
            if key == 'temperature':
                self._temperature = value
            super().__setattr__(key, value)


if version.parse(vllm.__version__) < version.parse('0.5.5'):

    class VllmGenerationConfig(_VllmGenerationConfigMixin, SamplingParams):

        def __init__(
            self,
            max_tokens: int = 64,  # max_tokens
            temperature: float = 1.,
            top_k: int = 50,  # -1: all
            top_p: float = 1.,
            repetition_penalty: float = 1.,
            *,
            n: int = 1,
            logprobs: Optional[int] = None,
            seed: Optional[int] = None,
            length_penalty: float = 1.,
            stop: Optional[List[str]] = None,
            skip_special_tokens: bool = False,
            **kwargs,
        ) -> None:
            # compat
            max_new_tokens = kwargs.pop('max_new_tokens', None)
            if max_new_tokens is not None:
                max_tokens = max_new_tokens
            if top_k == 0:
                top_k = -1
            if stop is None:
                stop = []
            kwargs['max_tokens'] = max_tokens
            kwargs['temperature'] = temperature
            kwargs['top_k'] = top_k
            kwargs['top_p'] = top_p
            kwargs['repetition_penalty'] = repetition_penalty
            kwargs['n'] = n
            kwargs['logprobs'] = logprobs
            kwargs['seed'] = seed
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

else:

    class VllmGenerationConfig(_VllmGenerationConfigMixin, SamplingParams):
        max_tokens: int = 64
        temperature: float = 1.
        top_k: int = 50  # -1: all
        top_p: float = 1.
        repetition_penalty: float = 1.
        n: int = 1
        logprobs: Optional[int] = None
        seed: Optional[int] = None
        length_penalty: float = 1.
        stop: Optional[List[str]] = None
        skip_special_tokens: bool = False

        def __post_init__(self):
            if self.top_k == 0:
                self.top_k = -1
            if self.stop is None:
                self.stop = []
            self._temperature = self.temperature
            super().__post_init__()


def add_vllm_request(llm_engine: Union[LLMEngine, AsyncLLMEngine], inputs: Dict[str, Any], *, request_id: str,
                     generation_config: VllmGenerationConfig, **kwargs):
    input_ids = inputs['input_ids']
    if version.parse(vllm.__version__) >= version.parse('0.4.3'):
        llm_inputs = {'prompt_token_ids': input_ids}
        mm_data = {}
        for key in ['images', 'audios', 'videos']:
            meida_data = inputs.get(key) or []
            if meida_data:
                if version.parse(vllm.__version__) < version.parse('0.6'):
                    assert len(meida_data) == 1, (
                        f'The current version of vllm only supports single {key}. Please upgrade to vllm >= 0.6.0')
                    mm_data = {key.rstrip('s'): meida_data[0]}
                else:
                    mm_data = {key.rstrip('s'): meida_data[0] if len(meida_data) == 1 else meida_data}
        if mm_data:
            llm_inputs['multi_modal_data'] = mm_data
        if llm_engine.__class__.__name__ == 'LLMEngine':
            result_generator = llm_engine.add_request(request_id, llm_inputs, generation_config, **kwargs)
        else:
            result_generator = llm_engine.generate(llm_inputs, generation_config, request_id, **kwargs)
    else:
        if llm_engine.__class__.__name__ == 'LLMEngine':
            result_generator = llm_engine.add_request(request_id, None, generation_config, input_ids, **kwargs)
        else:
            result_generator = llm_engine.generate(None, generation_config, request_id, input_ids, **kwargs)
    return result_generator


def _prepare_vllm_request(llm_engine: LLMEngine,
                          template: Template,
                          request_list: List[Dict[str, Any]],
                          *,
                          generation_config: VllmGenerationConfig,
                          generation_info: Dict[str, Any],
                          lora_request: Optional['LoRARequest'] = None,
                          use_tqdm: bool = False,
                          **kwargs) -> Tuple[List[Optional[Dict[str, Any]]], List[Tuple[bool, int]]]:
    for key in ['num_prompt_tokens', 'num_generated_tokens', 'num_samples']:
        if key not in generation_info:
            generation_info[key] = 0

    template.model = llm_engine
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

    resp_list: List[Optional[Dict[str, Any]]] = [None] * len(request_list)
    agent_state = []
    is_multimodal = getattr(llm_engine, 'is_multimodal', False)
    max_workers = os.cpu_count()
    if not is_multimodal:
        use_tqdm = False
        max_workers = 1

    prog_bar = tqdm(request_list, dynamic_ncols=True, disable=not use_tqdm)

    def _prepare_inputs(request: Dict[str, Any]) -> Dict[str, Any]:
        history = request.get('history') or []
        # agent support
        is_observation = history[-1][-1].endswith('Observation:') if history and history[-1][-1] else False
        act_length = None
        if is_observation:
            history[-1][-1] = history[-1][-1] + request['query']
            act_length = len(history[-1][-1])
            request['query'] = None
        agent_state.append((is_observation, act_length))
        request['history'] = history

        inputs = template.encode(request)[0]
        prog_bar.update()
        return inputs

    with template.vllm_context(), concurrent.futures.ThreadPoolExecutor(
            max_workers=min(max_workers, len(request_list))) as executor:
        futures = [executor.submit(_prepare_inputs, request) for request in request_list]
        concurrent.futures.wait(futures)
        inputs_list = [future.result() for future in futures]
    prog_bar.close()

    for i, (inputs, request) in enumerate(zip(inputs_list, request_list)):
        truncation_strategy = kwargs.pop('truncation_strategy', 'delete')
        if len(inputs) == 0 and truncation_strategy == 'delete':
            # input_ids exceeds `max_length`. Please increase the value of `max_length`.
            resp_list[i] = {'response': '', 'history': request['history']}
            continue
        generation_info['num_prompt_tokens'] += len(inputs['input_ids'])
        generation_info['num_samples'] += 1
        add_vllm_request(
            llm_engine, inputs, request_id=str(i), generation_config=generation_config, **add_request_kwargs)
    return resp_list, agent_state


@torch.inference_mode()
def inference_stream_vllm(
        llm_engine: LLMEngine,
        template: Template,
        request_list: List[Dict[str, Any]],
        *,
        generation_config: Optional[VllmGenerationConfig] = None,
        generation_info: Optional[Dict[str, Any]] = None,
        lora_request: Optional['LoRARequest'] = None,
        use_tqdm: bool = False,
        flush_steps: Optional[int] = None,  # Ensuring efficiency
        **kwargs) -> Iterator[List[Dict[str, Any]]]:
    """
    request_list: e.g. [{'query': 'hello!'}].
        The keys that can be included are: 'query', 'history', 'system', 'images'.
    generation_config: Priority: generation_config > model.generation_config.
    return: e.g. [{'response': 'hi!', 'history': [('hello!', 'hi!')]}].
        The keys to be included will be: 'response', 'history'.
    """
    if len(request_list) == 0:
        return
    start_runtime = time.perf_counter()
    if generation_config is None:
        generation_config = getattr(llm_engine, 'generation_config', None) or VllmGenerationConfig()
    assert isinstance(generation_config, VllmGenerationConfig)
    request_list = deepcopy(request_list)
    generation_config = deepcopy(generation_config)
    if generation_info is None:
        generation_info = {}
    else:
        generation_info.clear()

    resp_list, agent_state = _prepare_vllm_request(
        llm_engine,
        template,
        request_list,
        generation_config=generation_config,
        generation_info=generation_info,
        lora_request=lora_request,
        use_tqdm=use_tqdm,
        **kwargs)

    n_finished = 0
    n_steps = 0
    if flush_steps is None:
        flush_steps = min(10, generation_info['num_samples'])
    print_idx_list = [[0] for _ in range(len(request_list))]
    num_generated_tokens = [0] * len(request_list)
    prog_bar = tqdm(total=generation_info['num_samples'], dynamic_ncols=True, disable=not use_tqdm)
    while llm_engine.has_unfinished_requests():
        is_flush = False
        n_steps += 1
        step_outputs = llm_engine.step()
        for output in step_outputs:
            if not output.finished and n_steps % flush_steps != 0:
                continue
            is_flush = True
            i = int(output.request_id)
            request = request_list[i]
            generate_ids = output.outputs[0].token_ids
            logprobs = output.outputs[0].logprobs
            safe_response = template.generate_ids_to_response(
                generate_ids, output.finished, print_idx=print_idx_list[i])
            query = request['query']
            history = request['history']
            if resp_list[i] is None and not agent_state[i][0]:
                history.append(None)
            if not agent_state[i][0]:
                history[-1] = [query, safe_response]
            else:
                history[-1][-1] = history[-1][-1][:agent_state[i][1]] + safe_response

            n_gen_tokens = sum(len(_output.token_ids) for _output in output.outputs)
            generation_info['num_generated_tokens'] += n_gen_tokens - num_generated_tokens[i]
            num_generated_tokens[i] = n_gen_tokens

            resp_list[i] = {'response': safe_response, 'history': history}
            if logprobs is not None:
                resp_list[i]['logprobs'] = logprobs
            if output.finished:
                n_finished += 1
                prog_bar.update()
        if not is_flush:
            continue
        runtime = time.perf_counter() - start_runtime
        generation_info['runtime'] = runtime
        generation_info['samples/s'] = n_finished / runtime
        generation_info['tokens/s'] = generation_info['num_generated_tokens'] / runtime
        yield resp_list
    prog_bar.close()


@torch.inference_mode()
def inference_vllm(llm_engine: LLMEngine,
                   template: Template,
                   request_list: List[Dict[str, Any]],
                   *,
                   generation_config: Optional[VllmGenerationConfig] = None,
                   generation_info: Optional[Dict[str, Any]] = None,
                   max_batch_size: Optional[int] = None,
                   lora_request: Optional['LoRARequest'] = None,
                   use_tqdm: bool = False,
                   verbose: bool = False,
                   prompt_prefix: str = '[PROMPT]',
                   output_prefix: str = '[OUTPUT]',
                   **kwargs) -> List[Dict[str, Any]]:
    """
    request_list: e.g. [{'query': 'hello!'}].
        The keys that can be included are: 'query', 'history', 'system', 'images'.
    generation_config: Priority: generation_config > model.generation_config.
    return: e.g. [{'response': 'hi!', 'history': [('hello!', 'hi!')]}].
        The keys to be included will be: 'response', 'history'.
    """
    if len(request_list) == 0:
        return []
    runtime = time.perf_counter()

    is_multimodal = getattr(llm_engine, 'is_multimodal', False)
    if is_multimodal and max_batch_size is None:
        max_batch_size = 512

    _inner_call = kwargs.get('_inner_call', False)
    if generation_info is None:
        generation_info = {}
    elif not _inner_call:
        generation_info.clear()
    if max_batch_size is not None and len(request_list) > max_batch_size:
        i = 0
        resp_list = []
        kwargs['_inner_call'] = True
        while i < len(request_list):
            resp_list += inference_vllm(
                llm_engine,
                template,
                request_list[i:i + max_batch_size],
                generation_config=generation_config,
                generation_info=generation_info,
                max_batch_size=max_batch_size,
                lora_request=lora_request,
                use_tqdm=use_tqdm,
                verbose=verbose,
                prompt_prefix=prompt_prefix,
                output_prefix=output_prefix,
                **kwargs)
            i += max_batch_size
        runtime = time.perf_counter() - runtime
        generation_info['runtime'] = runtime
        generation_info['samples/s'] = generation_info['num_samples'] / runtime
        generation_info['tokens/s'] = generation_info['num_generated_tokens'] / runtime
        return resp_list

    if generation_config is None:
        generation_config = getattr(llm_engine, 'generation_config', None) or VllmGenerationConfig()
    assert isinstance(generation_config, VllmGenerationConfig)
    request_list = deepcopy(request_list)
    generation_config = deepcopy(generation_config)

    old_num_samples = generation_info.get('num_samples', 0)
    resp_list, agent_state = _prepare_vllm_request(
        llm_engine,
        template,
        request_list,
        generation_config=generation_config,
        generation_info=generation_info,
        lora_request=lora_request,
        use_tqdm=use_tqdm,
        **kwargs)

    tokenizer = template.tokenizer
    if use_tqdm:
        assert verbose is False
    prog_bar = tqdm(total=generation_info['num_samples'] - old_num_samples, dynamic_ncols=True, disable=not use_tqdm)
    outputs = []
    while llm_engine.has_unfinished_requests():
        step_outputs = llm_engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                prog_bar.update()
    prog_bar.close()

    for output in outputs:
        i = int(output.request_id)
        request = request_list[i]
        generate_ids = output.outputs[0].token_ids
        logprobs = output.outputs[0].logprobs
        response = template.generate_ids_to_response(generate_ids)
        query = request['query']
        history = request['history']
        if not agent_state[i][0]:
            history.append([query, response])
        else:
            history[-1][-1] = history[-1][-1] + response

        generation_info['num_generated_tokens'] += sum(len(_output.token_ids) for _output in output.outputs)
        resp_list[i] = {'response': response, 'history': history}
        if logprobs is not None:
            resp_list[i]['logprobs'] = logprobs
        if verbose:
            print(f'{prompt_prefix}{tokenizer.decode(output.prompt_token_ids, False)}{output_prefix}', end='')
            print(tokenizer.decode(output.outputs[0].token_ids, False))
    runtime = time.perf_counter() - runtime
    generation_info['runtime'] = runtime
    generation_info['samples/s'] = generation_info['num_samples'] / runtime
    generation_info['tokens/s'] = generation_info['num_generated_tokens'] / runtime
    return resp_list


def prepare_vllm_engine_template(args: InferArguments, use_async: bool = False) -> Tuple[LLMEngine, Template]:
    logger.info(f'device_count: {torch.cuda.device_count()}')

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
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        enforce_eager=args.enforce_eager,
        limit_mm_per_prompt=args.limit_mm_per_prompt,
        use_async=use_async,
        model_id_or_path=model_id_or_path,
        enable_lora=args.vllm_enable_lora,
        max_loras=max(len(args.lora_modules), 1),
        max_lora_rank=args.vllm_max_lora_rank)
    setattr(llm_engine.generation_config, 'max_tokens', args.max_new_tokens)
    for k in ['temperature', 'do_sample', 'top_k', 'top_p', 'repetition_penalty']:
        val = getattr(args, k, None)
        if val is not None:
            setattr(llm_engine.generation_config, k, val)
    logger.info(f'llm_engine.generation_config: {llm_engine.generation_config}')

    tokenizer = llm_engine.hf_tokenizer
    template: Template = get_template(
        args.template_type,
        tokenizer,
        args.system,
        args.max_length,
        args.truncation_strategy,
        model=llm_engine,
        tools_prompt=args.tools_prompt)
    logger.info(f'system: {template.default_system}')
    return llm_engine, template
