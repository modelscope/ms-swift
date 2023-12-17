from copy import deepcopy
from typing import *

import torch
from torch import dtype as Dtype
from vllm import EngineArgs, LLMEngine, SamplingParams

from swift.llm import *
from swift.llm.utils.model import *


def get_vllm_engine(
    model_type: str,
    torch_dtype: Optional[Dtype] = None,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    engine_kwargs: Optional[Dict[str, Any]] = None,
) -> LLMEngine:
    if engine_kwargs is None:
        engine_kwargs = {}
    model_info = MODEL_MAPPING[model_type]
    model_id_or_path = model_info['model_id_or_path']
    revision = model_info['revision']
    ignore_file_pattern = model_info['ignore_file_pattern']
    model_dir = snapshot_download(
        model_id_or_path, revision, ignore_file_pattern=ignore_file_pattern)
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
        stop: Optional[Union[str, List[str]]] = None,
        skip_special_tokens: bool = True,
        **kwargs,
    ):
        # The parameter design is similar to transformers.GenerationConfig.
        self.max_new_tokens = max_new_tokens
        super().__init__(
            max_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            stop=stop,
            skip_special_tokens=skip_special_tokens,
            **kwargs)

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
        generation_config = getattr(model, 'generation_config',
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

    response_list = []
    for output, request in zip(outputs, request_list):
        response = output.outputs[0].text
        query = request['query']
        history = request['history']
        history.append((query, response))
        response_list.append({'response': response, 'history': history})
        if verbose:
            print(
                f'{prompt_prefix}{tokenizer.decode(output.prompt_token_ids, False)}{output_prefix}',
                end='')
            print(tokenizer.decode(output.outputs[0].token_ids, False))
    return response_list


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
        generation_config = getattr(model, 'generation_config',
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

    response_list = []
    for output, request in zip(outputs, request_list):
        response = output.outputs[0].text
        query = request['query']
        history = request['history']
        history.append((query, response))
        response_list.append({'response': response, 'history': history})
        if verbose:
            print(
                f'{prompt_prefix}{tokenizer.decode(output.prompt_token_ids, False)}{output_prefix}',
                end='')
            print(tokenizer.decode(output.outputs[0].token_ids, False))
    return response_list


if __name__ == '__main__':
    model_type = ModelType.qwen_7b_chat
    llm_engine = get_vllm_engine(model_type, torch.float16)
    template_type = get_default_template_type(model_type)
    template = get_template(template_type, llm_engine.tokenizer)
    request_list = [{'query': '浙江的省会在哪？'}, {'query': '你好!'}]
    response_list = inference_vllm(llm_engine, template, request_list)
    for response in response_list:
        print(response)

    gen = inference_stream_vllm(llm_engine, template, request_list)
    for response_list in gen:
        print(response_list[0]['history'])
        print(response_list[1]['history'])
