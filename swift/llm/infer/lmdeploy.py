import asyncio
import inspect
import os
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

import torch
from lmdeploy import GenerationConfig as LmdeployGenerationConfig
from lmdeploy import PytorchEngineConfig, TurbomindEngineConfig, VisionConfig, pipeline
from lmdeploy.api import autoget_backend_config
from transformers import GenerationConfig, PreTrainedTokenizerBase

from swift.utils import get_logger
from ..template import Template
from .base import InferEngine
from .patch import patch_auto_config, patch_auto_tokenizer
from .protocol import (ChatCompletionMessageToolCall, ChatCompletionResponse, ChatCompletionResponseChoice,
                       ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse, ChatMessage, DeltaMessage,
                       Function, InferRequest, RequestConfig, UsageInfo, random_uuid)
from .utils import InferStreamer, InferTools

logger = get_logger()


class LMDeployEngine(InferEngine):

    def __init__(
            self,
            model_id_or_path: str,
            torch_dtype: Optional[torch.dtype] = None,
            *,
            model_type: Optional[str] = None,
            # engine_kwargs
            tp: int = 1,
            cache_max_entry_count: float = 0.8,
            quant_policy: int = 0,  # e.g. 4, 8
            vision_batch_size: int = 1,  # max_batch_size in VisionConfig
            engine_kwargs: Optional[Dict[str, Any]] = None,  # extra
            **kwargs) -> None:

        self._prepare_model_tokenizer(model_id_or_path, torch_dtype, False, model_type=model_type, **kwargs)
        self._prepare_engine_kwargs(
            tp=tp,
            cache_max_entry_count=cache_max_entry_count,
            quant_policy=quant_policy,
            vision_batch_size=vision_batch_size,
            engine_kwargs=engine_kwargs)

        self._prepare_engine()
        self._load_generation_config()

    def _prepare_engine_kwargs(self,
                               tp: int = 1,
                               cache_max_entry_count: float = 0.8,
                               quant_policy: int = 0,
                               vision_batch_size: int = 1,
                               engine_kwargs: Optional[Dict[str, Any]] = None):
        if engine_kwargs is None:
            engine_kwargs = {}
        engine_kwargs['tp'] = tp
        engine_kwargs['cache_max_entry_count'] = cache_max_entry_count
        engine_kwargs['quant_policy'] = quant_policy

        backend_config = TurbomindEngineConfig(**engine_kwargs)
        backend_config = autoget_backend_config(self.model_dir, backend_config)
        if isinstance(backend_config, PytorchEngineConfig):
            backend_config.thread_safe = True
        self.backend_config = backend_config
        logger.info(f'backend_config: {backend_config}')

        pipeline_kwargs = {}
        if self.is_multimodal:
            vision_config = VisionConfig(max_batch_size=vision_batch_size)
            pipeline_kwargs['vision_config'] = vision_config
            logger.info(f'vision_config: {vision_config}')
        self.pipeline_kwargs = pipeline_kwargs

    def _patch_pipeline(self):
        from lmdeploy.serve import async_engine
        _old_best_match_model = async_engine.best_match_model

        def _best_match_model(*args, **kwargs) -> Optional[str]:
            return self.model_type

        async_engine.best_match_model = _best_match_model
        yield
        async_engine.best_match_model = _old_best_match_model

    def _prepare_engine(self):
        with patch_auto_tokenizer(self.tokenizer), patch_auto_config(self.config), self._patch_pipeline():
            engine = pipeline(self.model_dir, backend_config=self.backend_config, **self.pipeline_kwargs)
        self.engine = engine

    def _load_generation_config(self):
        generation_config_path = os.path.join(self.model_dir, 'generation_config.json')
        if os.path.isfile(generation_config_path):
            generation_config = GenerationConfig.from_pretrained(self.model_dir)
            kwargs = generation_config.to_dict()
            max_new_tokens = kwargs.get('max_new_tokens')
            if max_new_tokens is None:
                kwargs.pop('max_new_tokens', None)
            parameters = inspect.signature(LmdeployGenerationConfig.__init__).parameters
            for k, v in kwargs.copy().items():
                if k not in parameters or v is None:
                    kwargs.pop(k)
            self.generation_config = LmdeployGenerationConfig(**kwargs)
        else:
            self.generation_config = LmdeployGenerationConfig()

    def _prepare_generation_config(self):
        pass

    def _add_stop_words(self):
        pass

    @staticmethod
    def _get_logprobs(tokenizer: PreTrainedTokenizerBase,
                      logprobs_list: Optional[List[Dict[int, float]]],
                      token_ids: List[int],
                      top_logprobs: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if logprobs_list is None:
            return None
        res = []
        for logprobs, token_id in zip(logprobs_list, token_ids):
            token = tokenizer.decode(token_id)
            _res = {'token': token, 'logprob': logprobs[token_id], 'bytes': list(token.encode('utf8'))}
            if top_logprobs is not None:
                res_top_logprobs = []
                for k, logprob in logprobs.items():
                    if k == token_id:  # TODO
                        continue
                    token = tokenizer.decode(k)
                    res_top_logprobs.append({'token': token, 'logprob': logprob, 'bytes': list(token.encode('utf8'))})
                _res['top_logprobs'] = res_top_logprobs
            res.append(_res)
        return {'content': res}

    @torch.inference_mode()
    async def infer_async(
            self,
            template: Template,
            infer_request: InferRequest,
            request_config: Optional[RequestConfig] = None,
            *,
            request_id: Optional[str] = None
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        if request_id is None:
            request_id = f'chatcmpl-{random_uuid()}'
        created_time = int(time.time())
        request_config = request_config or RequestConfig()

        inputs = template.encode(infer_request)
        assert len(inputs) >= 0
        generation_config = self._prepare_generation_config(request_config)
        self._add_stop_words(generation_config, request_config, template)
        result_generator = self._add_request(inputs, generation_config, request_id)
        infer_args = (template, result_generator, generation_config, request_id, created_time)
        if request_config.stream:
            return self._infer_stream_async(*infer_args)
        else:
            return await self._infer_async(*infer_args)
