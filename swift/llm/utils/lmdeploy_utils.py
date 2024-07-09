import asyncio
import inspect
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import torch
from lmdeploy import EngineGenerationConfig as _LmdeployGenerationConfig
from lmdeploy import TurbomindEngineConfig, pipeline
from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.serve.vl_async_engine import VLAsyncEngine
from torch import dtype as Dtype
from tqdm import tqdm
from transformers import GenerationConfig

#
from swift.llm import Template, get_model_tokenizer


def get_lmdeploy_engine(model_type: str,
                        torch_dtype: Optional[Dtype] = None,
                        *,
                        model_id_or_path: Optional[str] = None,
                        revision: Optional[str] = None,
                        tp: int = 1,
                        cache_max_entry_count: float = 0.8,
                        engine_kwargs: Optional[Dict[str, Any]] = None,
                        **kwargs) -> Union[AsyncEngine, VLAsyncEngine]:
    tokenizer = get_model_tokenizer(
        model_type, load_model=False, model_id_or_path=model_id_or_path, revision=revision, download_model=True)[1]
    model_dir = tokenizer.model_dir

    if engine_kwargs is None:
        engine_kwargs = {}
    engine_kwargs['tp'] = tp
    engine_kwargs['cache_max_entry_count'] = cache_max_entry_count

    backend_config = TurbomindEngineConfig(**engine_kwargs)
    lmdeploy_engine = pipeline(model_dir, backend_config=backend_config)
    lmdeploy_engine.model_dir = model_dir
    lmdeploy_engine.model_type = model_type
    lmdeploy_engine.hf_tokenizer = tokenizer

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


def _prepare_lmdeploy_request(lmdeploy_engine: Union[AsyncEngine, VLAsyncEngine],
                          template: Template,
                          request_list: List[Dict[str, Any]],
                          *,
                          generation_config: LmdeployGenerationConfig,
                          use_tqdm: bool = False,
                          **kwargs):
    pass

@torch.inference_mode()
def inference_lmdeploy(lmdeploy_engine: Union[AsyncEngine, VLAsyncEngine],
                       template: Template,
                       request_list: List[Dict[str, Any]],
                       *,
                       generation_config: Optional[LmdeployGenerationConfig] = None,
                       generation_info: Optional[Dict[str, int]] = None,
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

    tokenizer = template.tokenizer
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in generation_config.stop_words:
        generation_config.stop_words.append(tokenizer.eos_token_id)
    if isinstance(template.suffix[-1], str):
        token_list = tokenizer.encode(template.suffix[-1], skip_special_token=True)
        if len(token_list) == 1 and token_list not in generation_config.stop_words:
            generation_config.stop_words.append(token_list[0])
    if isinstance(template.suffix[-1], list) and len(
            template.suffix[-1]) == 1 and template.suffix[-1] not in generation_config.stop_words:
        generation_config.stop_words.append(template.suffix[-1])

    resp_list: List[Optional[Dict[str, Any]]] = [None] * len(request_list)
    generators = []
    for i, request in enumerate(tqdm(request_list, dynamic_ncols=True, disable=not use_tqdm)):
        history = request.get('history', None)
        if history is None:
            history = []

        request['history'] = history
        inputs = template.encode(request)[0]
        truncation_strategy = kwargs.pop('truncation_strategy', 'delete')
        if len(inputs) == 0 and truncation_strategy == 'delete':
            # input_ids exceeds `max_length`. Please increase the value of `max_length`.
            resp_list[i] = {'response': '', 'history': history}
            continue
        generator = lmdeploy_engine.get_generator(False, i)
        generators.append((i, inputs, generator))

    if generation_info is None:
        generation_info = {}
    else:
        generation_info.clear()
    for key in ['num_prompt_tokens', 'num_generated_tokens']:
        generation_info[key] = 0

    if use_tqdm:
        assert verbose is False
    prog_bar = tqdm(total=len(request_list), dynamic_ncols=True, disable=not use_tqdm)

    async def _inner_infer(i: int, inputs: Dict[str, Any], generator) -> None:
        async with lmdeploy_engine.safe_run(i):
            async for output in generator.async_stream_infer(
                    session_id=i, **inputs, sequence_end=True,
                    stream_output=False, gen_config=generation_config):
                pass
        request = request_list[i]
        input_ids = inputs['input_ids']
        response = template.generate_ids_to_response(output.token_ids)
        query = request['query']
        history = request['history']
        history.append([query, response])
        generation_info['num_prompt_tokens'] += len(input_ids)
        generation_info['num_generated_tokens'] += len(output.token_ids)
        resp_list[i] = {'response': response, 'history': history}
        if verbose:
            print(f'{prompt_prefix}{tokenizer.decode(input_ids, False)}{output_prefix}', end='')
            print(tokenizer.decode(output.token_ids, False))

    async def _batch_infer() -> None:
        tasks = [_inner_infer(i, inputs, await generator) for i, inputs, generator in generators]
        for coro in asyncio.as_completed(tasks):
            await coro
            prog_bar.update()

    asyncio.run(_batch_infer())
    runtime = time.perf_counter() - runtime
    generation_info['runtime'] = runtime
    generation_info['samples/s'] = len(generators) / runtime
    generation_info['tokens/s'] = generation_info['num_generated_tokens'] / runtime

    return resp_list


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    from swift.llm import get_default_template_type, get_template
    model_type = 'qwen-7b-chat'
    request_list = [{'query': '晚上睡不着觉怎么办'} for i in range(100)]

if __name__ == '__main__':
    lmdeploy_engine = get_lmdeploy_engine(model_type, torch.float16)
    template_type = get_default_template_type(model_type)
    template = get_template(template_type, lmdeploy_engine.hf_tokenizer)
    lmdeploy_engine.generation_config.max_new_tokens = 256
    generation_info = {}
    resp_list = inference_lmdeploy(lmdeploy_engine, template, request_list[:2], 
                                   generation_info=generation_info, verbose=True)
    print(generation_info)

# if __name__ == '__main__':
#     from swift.llm import inference_vllm, get_vllm_engine
#     vllm_engine = get_vllm_engine(model_type, torch.float16)
#     template_type = get_default_template_type(model_type)
#     template = get_template(template_type, vllm_engine.hf_tokenizer)
#     vllm_engine.generation_config.max_new_tokens = 256
#     generation_info = {}
#     resp_list = inference_vllm(vllm_engine, template, request_list[:2], generation_info=generation_info, verbose=True)
#     print(generation_info)