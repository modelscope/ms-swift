# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import os
import time
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import torch
from lmdeploy import GenerationConfig as LmdeployGenerationConfig
from lmdeploy import PytorchEngineConfig, TurbomindEngineConfig, VisionConfig, pipeline
from lmdeploy.api import autoget_backend_config
from lmdeploy.serve import async_engine
from transformers import GenerationConfig

from swift.llm import InferRequest, Template, TemplateMeta, get_model_tokenizer
from swift.plugin import Metric
from swift.utils import get_logger, get_seed
from ..protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
                        ChatCompletionStreamResponse, ChatMessage, DeltaMessage, RequestConfig)
from .infer_engine import InferEngine
from .patch import patch_auto_config, patch_auto_tokenizer
from .utils import InferStreamer

logger = get_logger()


class LmdeployEngine(InferEngine):

    def __init__(
            self,
            model_id_or_path: str,
            torch_dtype: Optional[torch.dtype] = None,
            *,
            model_type: Optional[str] = None,
            use_hf: Optional[bool] = None,
            revision: Optional[str] = None,
            # engine_kwargs
            tp: int = 1,
            session_len: Optional[int] = None,
            cache_max_entry_count: float = 0.8,
            quant_policy: int = 0,  # e.g. 4, 8
            vision_batch_size: int = 1,  # max_batch_size in VisionConfig
            engine_kwargs: Optional[Dict[str, Any]] = None) -> None:

        self.processor = get_model_tokenizer(
            model_id_or_path,
            torch_dtype,
            load_model=False,
            download_model=True,
            model_type=model_type,
            use_hf=use_hf,
            revision=revision)[1]
        self._post_init()

        if self.max_model_len is not None:
            self.max_model_len -= 1
        self._prepare_engine_kwargs(
            tp=tp,
            session_len=session_len,
            cache_max_entry_count=cache_max_entry_count,
            quant_policy=quant_policy,
            vision_batch_size=vision_batch_size,
            engine_kwargs=engine_kwargs)

        self.config.torch_dtype = torch_dtype
        self._prepare_engine()
        self._load_generation_config()

    def _prepare_engine_kwargs(self,
                               tp: int = 1,
                               session_len: Optional[int] = None,
                               cache_max_entry_count: float = 0.8,
                               quant_policy: int = 0,
                               vision_batch_size: int = 1,
                               engine_kwargs: Optional[Dict[str, Any]] = None):
        if engine_kwargs is None:
            engine_kwargs = {}
        engine_kwargs['tp'] = tp
        engine_kwargs['session_len'] = session_len
        engine_kwargs['cache_max_entry_count'] = cache_max_entry_count
        engine_kwargs['quant_policy'] = quant_policy

        backend_config = TurbomindEngineConfig(**engine_kwargs)
        backend_config = autoget_backend_config(self.model_dir, backend_config)
        if isinstance(backend_config, PytorchEngineConfig):
            backend_config.thread_safe = True
        self.backend_config = backend_config
        logger.info(f'backend_config: {backend_config}')

        pipeline_kwargs = {}
        is_multimodal = self.model_meta.is_multimodal
        if is_multimodal:
            vision_config = VisionConfig(max_batch_size=vision_batch_size)
            pipeline_kwargs['vision_config'] = vision_config
            logger.info(f'vision_config: {vision_config}')
        self.pipeline_kwargs = pipeline_kwargs

    @contextmanager
    def _patch_pipeline(self):
        _old_best_match_model = async_engine.best_match_model

        def _best_match_model(*args, **kwargs) -> Optional[str]:
            return self.model_info.model_type

        async_engine.best_match_model = _best_match_model
        try:
            yield
        finally:
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

    def _get_stop_token_ids(self, stop_words: List[Union[str, List[int], None]]) -> List[int]:
        stop_token_ids: List[int] = []
        for stop_word in stop_words:
            if stop_word is None:
                continue
            if isinstance(stop_word, str):
                stop_word = self.tokenizer.encode(stop_word, add_special_tokens=False)
            if isinstance(stop_word, list):
                if len(stop_word) != 1:
                    continue
                else:
                    stop_token = stop_word[0]
            elif isinstance(stop_word, int):
                stop_token = stop_word
            assert isinstance(stop_token, int)
            if stop_token not in stop_token_ids:
                stop_token_ids.append(stop_token)
        return stop_token_ids

    def _add_stop_words(self, generation_config: LmdeployGenerationConfig, request_config: RequestConfig,
                        template_meta: TemplateMeta) -> None:
        stop_words = (request_config.stop or []) + (self.generation_config.stop_words or []) + template_meta.stop_words
        generation_config.stop_token_ids = self._get_stop_token_ids(stop_words)

    def _prepare_generation_config(self, request_config: RequestConfig) -> LmdeployGenerationConfig:
        kwargs = {'max_new_tokens': request_config.max_tokens}
        for key in ['temperature', 'top_k', 'top_p', 'repetition_penalty']:
            new_value = getattr(request_config, key)
            if new_value is None:
                kwargs[key] = getattr(self.generation_config, key)
            else:
                kwargs[key] = new_value
        if request_config.seed is None:
            request_config.seed = get_seed()
        kwargs['random_seed'] = request_config.seed

        if request_config.logprobs:
            kwargs['logprobs'] = 1
            if request_config.top_logprobs is not None:
                kwargs['logprobs'] = max(1, request_config.top_logprobs)

        return LmdeployGenerationConfig(**kwargs)

    async def _infer_stream_async(
            self, template: Template, inputs: Dict[str, Any],
            generation_config: LmdeployGenerationConfig) -> AsyncIterator[ChatCompletionStreamResponse]:
        session_id = time.time_ns()
        generator = await self.engine.get_generator(False, session_id)

        infer_streamer = InferStreamer(template)
        token_idx = 0
        async with self.engine.safe_run(session_id):
            async_iter = generator.async_stream_infer(
                session_id=session_id, **inputs, stream_output=True, gen_config=generation_config).__aiter__()
            is_finished = False
            while not is_finished:
                try:
                    output = await async_iter.__anext__()
                except StopAsyncIteration:
                    is_finished = True
                delta_text = infer_streamer.get_printable_text(output.token_ids, is_finished)
                if not delta_text and not is_finished:
                    continue

                logprobs = self._get_logprobs(template.tokenizer, output.logprobs, output.token_ids[token_idx:],
                                              generation_config.logprobs)
                token_idx = len(output.token_ids)

                usage_info = self._get_usage_info(len(inputs['input_ids']), output.num_token)
                toolcall = None
                if is_finished:
                    toolcall = self._get_toolcall(template.decode(output.token_ids))
                finish_reason = self._get_finish_reason(generation_config.max_new_tokens, output.num_token,
                                                        output.status.name == 'FINISH')
                choices = [
                    ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(role='assistant', content=delta_text, tool_calls=toolcall),
                        finish_reason=finish_reason,
                        logprobs=logprobs)
                ]
                yield ChatCompletionStreamResponse(model=self.model_name, choices=choices, usage=usage_info)

    async def _infer_full_async(self, template: Template, inputs: Dict[str, Any],
                                generation_config: LmdeployGenerationConfig) -> ChatCompletionResponse:
        session_id = time.time_ns()
        generator = await self.engine.get_generator(False, session_id)

        async with self.engine.safe_run(session_id):
            async for output in generator.async_stream_infer(
                    session_id=session_id, **inputs, stream_output=False, gen_config=generation_config):
                pass

        response = template.decode(output.token_ids)
        logprobs = self._get_logprobs(template.tokenizer, output.logprobs, output.token_ids, generation_config.logprobs)

        usage_info = self._get_usage_info(len(inputs['input_ids']), output.num_token)
        toolcall = self._get_toolcall(response)
        finish_reason = self._get_finish_reason(generation_config.max_new_tokens, output.num_token,
                                                output.status.name == 'FINISH')
        choices = [
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),
                finish_reason=finish_reason,
                logprobs=logprobs)
        ]
        return ChatCompletionResponse(model=self.model_name, choices=choices, usage=usage_info)

    @torch.inference_mode()
    async def infer_async(self,
                          infer_request: InferRequest,
                          request_config: Optional[RequestConfig] = None,
                          *,
                          template: Optional[Template] = None,
                          **kwargs) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        request_config = deepcopy(request_config or RequestConfig())
        if template is None:
            template = self.default_template

        template.set_mode('lmdeploy')
        if request_config.seed is None:
            request_config.seed = get_seed()

        inputs = template.encode(infer_request)
        images = inputs.pop('images', None)
        if images:
            inputs['images'] = await self.engine.vl_encoder.async_infer(images)
            await template.prepare_lmdeploy_inputs(inputs)

        self.set_default_max_tokens(request_config, inputs)
        generation_config = self._prepare_generation_config(request_config)
        self._add_stop_words(generation_config, request_config, template.template_meta)
        kwargs.update({'template': template, 'inputs': inputs, 'generation_config': generation_config})
        for pre_infer_hook in self.pre_infer_hooks:
            kwargs = pre_infer_hook(kwargs)
        if request_config.stream:
            return self._infer_stream_async(**kwargs)
        else:
            return await self._infer_full_async(**kwargs)

    @torch.inference_mode()
    def infer(
        self,
        infer_requests: List[InferRequest],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        template: Optional[Template] = None,
        use_tqdm: Optional[bool] = None,
    ) -> Union[List[ChatCompletionResponse], Iterator[List[Optional[ChatCompletionStreamResponse]]]]:
        if hasattr(self.engine, 'vl_encoder'):
            self.engine.vl_encoder._loop_task = None
        return super().infer(infer_requests, request_config, metrics, template=template, use_tqdm=use_tqdm)
