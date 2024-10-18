import inspect
import os
import time
from contextlib import contextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

import torch
from lmdeploy import GenerationConfig as LmdeployGenerationConfig
from lmdeploy import PytorchEngineConfig, TurbomindEngineConfig, VisionConfig, pipeline
from lmdeploy.api import autoget_backend_config
from lmdeploy.serve import async_engine
from transformers import GenerationConfig, PreTrainedTokenizerBase

from swift.llm import Template
from swift.utils import get_logger, get_seed
from ..patch import patch_auto_config, patch_auto_tokenizer
from ..protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
                        ChatCompletionStreamResponse, ChatMessage, DeltaMessage, RequestConfig, UsageInfo, random_uuid)
from ..utils import InferStreamer, InferTools
from .infer_engine import InferEngine

logger = get_logger()


class LmdeployEngine(InferEngine):

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

    @contextmanager
    def _patch_pipeline(self):
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

    def _add_stop_words(self, generation_config: LmdeployGenerationConfig, request_config: RequestConfig,
                        template: Template) -> None:
        stop_words = (request_config.stop or []) + (self.generation_config.stop_words or []) + template.stop_words
        stop_words += [template.suffix[-1], self.tokenizer.eos_token]
        generation_config.stop_words = self._get_stop_words(stop_words)

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

    async def _add_request(self, template: Template, inputs: Dict[str, Any], session_id: int):

        generator = await self.engine.get_generator(False, session_id)
        images = inputs.pop('images', None) or []
        if len(images) > 0:
            inputs['images'] = await self.engine.vl_encoder.async_infer(images)
            await template.prepare_lmdeploy_inputs(inputs)
        return generator

    @staticmethod
    def _get_finish_reason(output, generation_config: LmdeployGenerationConfig):
        if output.status.name == 'FINISH':
            if output.num_token >= generation_config.max_new_tokens:
                finish_reason = 'length'
            else:
                finish_reason = 'stop'
        else:
            finish_reason = None
        return finish_reason

    async def _infer_stream_async(
            self, template: Template, inputs: Dict[str, Any],
            generation_config: LmdeployGenerationConfig) -> AsyncIterator[ChatCompletionStreamResponse]:
        session_id = time.time_ns()
        request_id = f'chatcmpl-{random_uuid()}'
        generator = await self._add_request(template, inputs, session_id)

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
                toolcall = self._get_toolcall(output.token_ids, is_finished)
                finish_reason = self._get_finish_reason(output, generation_config)
                choices = [
                    ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(role='assistant', content=delta_text, tool_calls=toolcall),
                        finish_reason=finish_reason,
                        logprobs=logprobs)
                ]
                yield ChatCompletionStreamResponse(
                    model=self.model_dir, choices=choices, usage=usage_info, id=request_id)

    async def _infer_full_async(self, template: Template, inputs: Dict[str, Any],
                                generation_config: LmdeployGenerationConfig) -> ChatCompletionResponse:
        session_id = time.time_ns()
        generator = await self._add_request(template, inputs, session_id)

        async with self.engine.safe_run(session_id):
            async for output in generator.async_stream_infer(
                    session_id=session_id, **inputs, stream_output=False, gen_config=generation_config):
                pass

        response = InferTools.safe_decode(template, output.token_ids, True)
        logprobs = self._get_logprobs(template.tokenizer, output.logprobs, output.token_ids, generation_config.logprobs)

        usage_info = self._get_usage_info(len(inputs['input_ids']), output.num_token)
        toolcall = self._get_toolcall(response, True)
        finish_reason = self._get_finish_reason(output, generation_config)
        choices = [
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),
                finish_reason=finish_reason,
                logprobs=logprobs)
        ]
        return ChatCompletionResponse(model=self.model_dir, choices=choices, usage=usage_info)
