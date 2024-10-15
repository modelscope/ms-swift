import asyncio
import concurrent.futures
import inspect
import os
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from queue import Queue
from threading import Thread
from .protocol import (ChatCompletionMessageToolCall, ChatCompletionResponse, ChatCompletionResponseChoice,
                       ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse, ChatMessage, DeltaMessage,
                       Function, InferRequest, RequestConfig, UsageInfo, random_uuid)
from .utils import InferStreamer, InferTools
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
from lmdeploy import PytorchEngineConfig, TurbomindEngineConfig, VisionConfig, pipeline
from lmdeploy.api import autoget_backend_config
from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.serve.vl_async_engine import VLAsyncEngine
from .patch import patch_auto_config, patch_auto_tokenizer
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from .base import InferEngine
from ..model import get_model_tokenizer, HfConfigFactory
from ..template import Template, get_template
from swift.utils import get_logger, get_seed

from lmdeploy import GenerationConfig as LmdeployGenerationConfig

logger = get_logger()


class LMDeployEngine(InferEngine):

    def __init__(self,
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

    def _get_stop_words(self, request_config: RequestConfig, template: Template) -> List[str]:
        stop_words = (request_config.stop or []) + (
            self.generation_config.stop_words or []) + template.stop_words + [template.suffix[-1], self.tokenizer.eos_token]
        

    def _add_stop_word(self, stop_words: List[int], token: Union[List[int], int, str, None]) -> None:
        if token is None:
            return
        elif isinstance(token, int):
            stop_words.append(token)
        elif isinstance(token, str) and self.tokenizer is not None:
            token_list = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_list) == 1 and token_list[0] not in stop_words:
                stop_words.append(token_list[0])
        elif isinstance(token, list) and len(token) == 1 and token[0] not in stop_words:
            stop_words.append(token[0])
