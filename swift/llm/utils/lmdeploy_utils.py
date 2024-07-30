import asyncio
import concurrent.futures
import inspect
import os
import time
from contextlib import contextmanager
from copy import deepcopy
from queue import Queue
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from lmdeploy import EngineGenerationConfig as _LmdeployGenerationConfig
from lmdeploy import PytorchEngineConfig, TurbomindEngineConfig, VisionConfig, pipeline
from lmdeploy.api import autoget_backend_config
from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.serve.vl_async_engine import VLAsyncEngine
from tqdm import tqdm
from transformers import AutoConfig, GenerationConfig

from swift.utils import get_logger
from .argument import InferArguments
from .model import get_model_tokenizer
from .template import Template, get_template
from .utils import get_max_model_len

logger = get_logger()


def get_lmdeploy_engine(
        model_type: str,
        # TODO: https://github.com/InternLM/lmdeploy/issues/1846
        # torch_dtype: Optional[Dtype] = None,
        *,
        model_id_or_path: Optional[str] = None,
        revision: Optional[str] = None,
        tp: int = 1,
        cache_max_entry_count: float = 0.8,
        vision_batch_size: int = 1,  # max_batch_size in VisionConfig
        engine_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs) -> Union[AsyncEngine, VLAsyncEngine]:
    model_dir = kwargs.pop('model_dir', None)
    tokenizer = get_model_tokenizer(
        model_type,
        load_model=False,
        model_id_or_path=model_id_or_path,
        model_dir=model_dir,
        revision=revision,
        download_model=True)[1]
    model_dir = tokenizer.model_dir
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    if engine_kwargs is None:
        engine_kwargs = {}
    engine_kwargs['tp'] = tp
    engine_kwargs['cache_max_entry_count'] = cache_max_entry_count

    backend_config = TurbomindEngineConfig(**engine_kwargs)
    backend_config = autoget_backend_config(model_dir, backend_config)
    if isinstance(backend_config, PytorchEngineConfig):
        backend_config.thread_safe = True
    logger.info(f'backend_config: {backend_config}')
    pipeline_kwargs = {}
    is_multimodal = tokenizer.is_multimodal
    if is_multimodal:
        vision_config = VisionConfig(max_batch_size=vision_batch_size)
        pipeline_kwargs['vision_config'] = vision_config
        logger.info(f'vision_config: {vision_config}')

    lmdeploy_engine = pipeline(model_dir, backend_config=backend_config, **pipeline_kwargs)
    lmdeploy_engine.model_dir = model_dir
    lmdeploy_engine.model_type = model_type
    lmdeploy_engine.is_multimodal = is_multimodal
    lmdeploy_engine.hf_tokenizer = tokenizer
    lmdeploy_engine.model_config = model_config
    lmdeploy_engine.max_model_len = get_max_model_len(model_config)

    generation_config_path = os.path.join(model_dir, 'generation_config.json')
    if os.path.isfile(generation_config_path):
        generation_config = GenerationConfig.from_pretrained(model_dir)
        kwargs = generation_config.to_dict()
        parameters = inspect.signature(LmdeployGenerationConfig.__init__).parameters
        for k in kwargs.copy().keys():
            if k not in parameters:
                kwargs.pop(k)
        lmdeploy_engine.generation_config = LmdeployGenerationConfig(**kwargs)
    else:
        lmdeploy_engine.generation_config = LmdeployGenerationConfig()

    return lmdeploy_engine


@contextmanager
def lmdeploy_context(self: Template):
    self._is_lmdeploy = True
    yield
    self._is_lmdeploy = False


class LmdeployGenerationConfig(_LmdeployGenerationConfig):

    def __init__(
        self,
        max_new_tokens: Optional[int] = 64,
        temperature: float = 1.,
        top_k: int = 50,  # -1: all
        top_p: float = 1.,
        repetition_penalty: float = 1.,
        *,
        n: int = 1,
        stop_words: Optional[List[int]] = None,
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> None:
        if stop_words is None:
            stop_words = []
        super().__init__(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            n=n,
            stop_words=stop_words,
            skip_special_tokens=skip_special_tokens,
            **kwargs)


def _add_stop_word(stop_words: List[int], token: Union[List[int], int, str, None], tokenizer=None) -> None:
    if token is None:
        return
    elif isinstance(token, int):
        stop_words.append(token)
    elif isinstance(token, str) and tokenizer is not None:
        token_list = tokenizer.encode(token, add_special_tokens=False)
        if len(token_list) == 1 and token_list[0] not in stop_words:
            stop_words.append(token_list[0])
    elif isinstance(token, list) and len(token) == 1 and token[0] not in stop_words:
        stop_words.append(token[0])


def _prepare_lmdeploy_request(lmdeploy_engine: Union[AsyncEngine, VLAsyncEngine],
                              template: Template,
                              request_list: List[Dict[str, Any]],
                              *,
                              generation_config: LmdeployGenerationConfig,
                              generation_info: Dict[str, Any],
                              use_tqdm: bool = False,
                              **kwargs):
    for key in ['num_prompt_tokens', 'num_generated_tokens', 'num_samples']:
        generation_info[key] = 0

    if hasattr(lmdeploy_engine, 'vl_encoder'):
        lmdeploy_engine.vl_encoder._loop_task = None

    template.model = lmdeploy_engine
    tokenizer = template.tokenizer

    _add_stop_word(generation_config.stop_words, tokenizer.eos_token_id, tokenizer=tokenizer)
    _add_stop_word(generation_config.stop_words, template.suffix[-1], tokenizer=tokenizer)

    resp_list: List[Optional[Dict[str, Any]]] = [None] * len(request_list)
    generators = []
    is_multimodal = getattr(lmdeploy_engine, 'is_multimodal', False)
    max_workers = os.cpu_count()
    if not is_multimodal:
        use_tqdm = False
        max_workers = 1

    prog_bar = tqdm(request_list, dynamic_ncols=True, disable=not use_tqdm)

    def _prepare_inputs(request: Dict[str, Any]) -> Dict[str, Any]:
        request['history'] = request.get('history') or []
        inputs = template.encode(request)[0]
        prog_bar.update()
        return inputs

    with lmdeploy_context(template), concurrent.futures.ThreadPoolExecutor(
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
        generator = lmdeploy_engine.get_generator(False, i)
        generators.append((i, inputs, generator))

    generation_info['num_samples'] = len(generators)
    return resp_list, generators


@torch.inference_mode()
def inference_stream_lmdeploy(lmdeploy_engine: Union[AsyncEngine, VLAsyncEngine],
                              template: Template,
                              request_list: List[Dict[str, Any]],
                              *,
                              generation_config: Optional[LmdeployGenerationConfig] = None,
                              generation_info: Optional[Dict[str, Any]] = None,
                              use_tqdm: bool = False,
                              **kwargs) -> List[Dict[str, Any]]:
    start_runtime = time.perf_counter()
    if generation_config is None:
        generation_config = getattr(lmdeploy_engine, 'generation_config', LmdeployGenerationConfig())
    assert isinstance(generation_config, LmdeployGenerationConfig)
    request_list = deepcopy(request_list)
    generation_config = deepcopy(generation_config)
    if generation_info is None:
        generation_info = {}
    else:
        generation_info.clear()

    resp_list, generators = _prepare_lmdeploy_request(
        lmdeploy_engine,
        template,
        request_list,
        generation_config=generation_config,
        generation_info=generation_info,
        use_tqdm=use_tqdm,
        **kwargs)

    n_finished = 0
    print_idx_list = [[0] for _ in range(len(request_list))]
    outputs = [None] * len(request_list)
    num_generated_tokens = [0] * len(request_list)
    prog_bar = tqdm(total=len(generators), dynamic_ncols=True, disable=not use_tqdm)
    queue = Queue()

    async def _inner_infer(i: int, inputs: Dict[str, Any], generator) -> None:
        generator = await generator
        images = inputs.pop('images', None) or []
        if len(images) > 0:
            inputs['images'] = await lmdeploy_engine.vl_encoder.async_infer(images)
            await template.prepare_lmdeploy_inputs(inputs)
        generation_info['num_prompt_tokens'] += len(inputs['input_ids'])
        async with lmdeploy_engine.safe_run(i):
            async for output in generator.async_stream_infer(
                    session_id=i, **inputs, stream_output=True, gen_config=generation_config):
                queue.put((i, output))
            queue.put((i, None))

    async def _batch_infer() -> None:
        tasks = [_inner_infer(i, inputs, generator) for i, inputs, generator in generators]
        await asyncio.gather(*tasks)

    thread = Thread(target=lambda: asyncio.run(_batch_infer()))
    thread.start()

    while n_finished < len(generators):
        i, output = queue.get()
        is_finished = False
        if output is None:
            is_finished = True
            n_finished += 1
            prog_bar.update()
            output = outputs[i]  # old value
        outputs[i] = output
        request = request_list[i]
        safe_response = template.generate_ids_to_response(output.token_ids, is_finished, print_idx=print_idx_list[i])
        query = request['query']
        history = request['history']
        if resp_list[i] is None:
            history.append(None)
        history[-1] = [query, safe_response]
        n_gen_tokens = len(output.token_ids)
        generation_info['num_generated_tokens'] += n_gen_tokens - num_generated_tokens[i]
        num_generated_tokens[i] = n_gen_tokens
        resp_list[i] = {'response': safe_response, 'history': history}

        runtime = time.perf_counter() - start_runtime
        generation_info['runtime'] = runtime
        generation_info['samples/s'] = n_finished / runtime
        generation_info['tokens/s'] = generation_info['num_generated_tokens'] / runtime
        yield resp_list
    prog_bar.close()


@torch.inference_mode()
def inference_lmdeploy(lmdeploy_engine: Union[AsyncEngine, VLAsyncEngine],
                       template: Template,
                       request_list: List[Dict[str, Any]],
                       *,
                       generation_config: Optional[LmdeployGenerationConfig] = None,
                       generation_info: Optional[Dict[str, Any]] = None,
                       use_tqdm: bool = False,
                       verbose: bool = False,
                       prompt_prefix: str = '[PROMPT]',
                       output_prefix: str = '[OUTPUT]',
                       **kwargs) -> List[Dict[str, Any]]:
    runtime = time.perf_counter()
    if generation_config is None:
        generation_config = getattr(lmdeploy_engine, 'generation_config', LmdeployGenerationConfig())
    assert isinstance(generation_config, LmdeployGenerationConfig)
    request_list = deepcopy(request_list)
    generation_config = deepcopy(generation_config)
    if generation_info is None:
        generation_info = {}
    else:
        generation_info.clear()

    resp_list, generators = _prepare_lmdeploy_request(
        lmdeploy_engine,
        template,
        request_list,
        generation_config=generation_config,
        generation_info=generation_info,
        use_tqdm=use_tqdm,
        **kwargs)

    tokenizer = template.tokenizer
    if use_tqdm:
        assert verbose is False
    prog_bar = tqdm(total=len(generators), dynamic_ncols=True, disable=not use_tqdm)

    async def _inner_infer(i: int, inputs: Dict[str, Any], generator) -> None:
        generator = await generator
        images = inputs.pop('images', None) or []
        if len(images) > 0:
            inputs['images'] = await lmdeploy_engine.vl_encoder.async_infer(images)
            await template.prepare_lmdeploy_inputs(inputs)
        generation_info['num_prompt_tokens'] += len(inputs['input_ids'])
        async with lmdeploy_engine.safe_run(i):
            async for output in generator.async_stream_infer(
                    session_id=i, **inputs, stream_output=False, gen_config=generation_config):
                pass
        request = request_list[i]
        input_ids = inputs['input_ids']
        response = template.generate_ids_to_response(output.token_ids)
        query = request['query']
        history = request['history']
        history.append([query, response])

        generation_info['num_generated_tokens'] += len(output.token_ids)
        resp_list[i] = {'response': response, 'history': history}
        if verbose:
            print(f'{prompt_prefix}{tokenizer.decode(input_ids, False)}{output_prefix}', end='')
            print(tokenizer.decode(output.token_ids, False))
        prog_bar.update()

    async def _batch_infer() -> None:
        tasks = [_inner_infer(i, inputs, generator) for i, inputs, generator in generators]
        await asyncio.gather(*tasks)

    asyncio.run(_batch_infer())
    prog_bar.close()
    runtime = time.perf_counter() - runtime
    generation_info['runtime'] = runtime
    generation_info['samples/s'] = len(generators) / runtime
    generation_info['tokens/s'] = generation_info['num_generated_tokens'] / runtime
    return resp_list


def prepare_lmdeploy_engine_template(args: InferArguments) -> Tuple[Union[AsyncEngine, VLAsyncEngine], Template]:
    logger.info(f'device_count: {torch.cuda.device_count()}')

    assert args.quantization_bit == 0, 'not support bnb'
    assert not args.sft_type == 'lora', 'you need to merge lora'
    # Loading Model and Tokenizer
    model_id_or_path = None
    if args.sft_type == 'full' and args.ckpt_dir is not None:
        model_id_or_path = args.ckpt_dir
    elif args.model_id_or_path is not None:
        model_id_or_path = args.model_id_or_path
    lmdeploy_engine = get_lmdeploy_engine(
        args.model_type, tp=args.tp, vision_batch_size=args.vision_batch_size, model_id_or_path=model_id_or_path)
    tokenizer = lmdeploy_engine.hf_tokenizer

    if not args.do_sample:
        args.temperature = 0

    stop_words = []
    for stop_word in args.stop_words:
        _add_stop_word(stop_words, stop_word, tokenizer=tokenizer)
    generation_config = LmdeployGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop_words=stop_words,
        repetition_penalty=args.repetition_penalty)
    logger.info(f'generation_config: {generation_config}')
    lmdeploy_engine.generation_config = generation_config
    template: Template = get_template(
        args.template_type,
        tokenizer,
        args.system,
        args.max_length,
        args.truncation_strategy,
        model=lmdeploy_engine,
        tools_prompt=args.tools_prompt)
    args.system = template.default_system
    logger.info(f'system: {args.system}')
    return lmdeploy_engine, template
