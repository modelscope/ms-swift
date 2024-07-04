import inspect
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import torch
from lmdeploy import GenerationConfig as _LmdeployGenerationConfig
from lmdeploy import TurbomindEngineConfig, pipeline
from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.serve.vl_async_engine import VLAsyncEngine
from torch import dtype as Dtype
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
        stop_words: Optional[List[str]] = None,
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            n=n,
            stop_words=stop_words,
            skip_special_token=skip_special_tokens,
            **kwargs)


@torch.inference_mode()
def inference_lmdeploy(lmdeploy_engine: Union[AsyncEngine, VLAsyncEngine],
                       template: Template,
                       request_list: List[Dict[str, Any]],
                       *,
                       generation_config: Optional[LmdeployGenerationConfig] = None,
                       use_tqdm: bool = False,
                       verbose: bool = False,
                       prompt_prefix: str = '[PROMPT]',
                       output_prefix: str = '[OUTPUT]',
                       **kwargs) -> List[Dict[str, Any]]:
    if generation_config is None:
        generation_config = getattr(lmdeploy_engine, 'generation_config', LmdeployGenerationConfig())
    assert isinstance(generation_config, LmdeployGenerationConfig)
    request_list = deepcopy(request_list)
    generation_config = deepcopy(generation_config)

    tokenizer = template.tokenizer
    if use_tqdm:
        assert verbose is False


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    from swift.llm import get_default_template_type, get_template

    model_type = 'qwen-7b-chat'
    lmdeploy_engine = get_lmdeploy_engine(model_type, torch.float16)
    template_type = get_default_template_type(model_type)
    template = get_template(template_type, lmdeploy_engine.hf_tokenizer)
    lmdeploy_engine.generation_config.max_new_tokens = 256

    request_list = [{'query': '你好!'}, {'query': '浙江的省会在哪？'}]
    resp_list = inference_lmdeploy(lmdeploy_engine, template, request_list)
    for request, resp in zip(request_list, resp_list):
        print(f"query: {request['query']}")
        print(f"response: {resp['response']}")
