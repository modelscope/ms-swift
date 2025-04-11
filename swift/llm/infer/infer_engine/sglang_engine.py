import os
from copy import deepcopy
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import sglang as sgl
import torch
from transformers import GenerationConfig

from swift.llm import InferRequest, Template, TemplateMeta, get_model_tokenizer
from swift.plugin import Metric
from ..protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
                        ChatCompletionStreamResponse, ChatMessage, DeltaMessage, RequestConfig, random_uuid)
from .infer_engine import InferEngine


class SglangEngine(InferEngine):
    sampling_parameters = {
        'max_new_tokens', 'stop', 'temperature', 'top_p', 'top_k', 'min_p', 'frequency_penalty', 'presence_penalty',
        'repetition_penalty', 'min_new_tokens', 'n'
    }

    def __init__(
        self,
        model_id_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        *,
        model_type: Optional[str] = None,
        use_hf: Optional[bool] = None,
        hub_token: Optional[str] = None,
        revision: Optional[str] = None,
    ):
        self.processor = get_model_tokenizer(
            model_id_or_path,
            torch_dtype,
            load_model=False,
            download_model=True,
            model_type=model_type,
            use_hf=use_hf,
            hub_token=hub_token,
            revision=revision)[1]
        self._post_init()
        self.engine = sgl.Engine(model_path=model_id_or_path, dtype=torch_dtype)
        self._load_generation_config()

    def _load_generation_config(self) -> None:
        generation_config_path = os.path.join(self.model_dir, 'generation_config.json')
        if os.path.isfile(generation_config_path):
            generation_config = GenerationConfig.from_pretrained(self.model_dir)
            kwargs = generation_config.to_dict()
            top_k = kwargs.get('top_k')
            if top_k == 0:
                kwargs['top_k'] = -1

            for k, v in kwargs.copy().items():
                if k not in self.sampling_parameters or v is None:
                    kwargs.pop(k)
            self.generation_config = kwargs
        else:
            self.generation_config = {}

    def _prepare_generation_config(self, request_config: RequestConfig) -> Dict[str, Any]:
        kwargs = {'max_new_tokens': request_config.max_tokens}
        for key in ['temperature', 'top_k', 'top_p', 'repetition_penalty']:
            new_value = getattr(request_config, key)
            if new_value is None:
                kwargs[key] = getattr(self.generation_config, key)
            else:
                kwargs[key] = new_value
        for key in ['n', 'frequency_penalty', 'presence_penalty']:
            kwargs[key] = getattr(request_config, key)

        return kwargs

    def _add_stop_words(self, generation_config: Dict[str, Any], request_config: RequestConfig,
                        template_meta: TemplateMeta) -> None:
        stop_words = (request_config.stop or []) + (self.generation_config.get('stop') or []) + template_meta.stop_words
        generation_config['stop'] = self._get_stop_words(stop_words)

    def infer(self,
              infer_requests: List[InferRequest],
              request_config: Optional[RequestConfig] = None,
              metrics: Optional[List[Metric]] = None,
              *,
              template: Optional[Template] = None,
              **kwargs) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
        request_config = deepcopy(request_config or RequestConfig())
        if template is None:
            template = self.default_template
        batched_inputs, error_list = self._batch_encode(
            infer_requests, template=template, strict=getattr(self, 'strict', True))
        self.set_default_max_tokens(request_config, batched_inputs)
        generation_config = self._prepare_generation_config(request_config)
        self._add_stop_words(generation_config, request_config, template.template_meta)
        input_ids = [inputs['input_ids'] for inputs in batched_inputs]
        outputs = self.engine.generate(input_ids=input_ids, sampling_params=generation_config)
        print()

    async def infer_async(
        self,
        infer_request: InferRequest,
        request_config: Optional[RequestConfig] = None,
        *,
        template: Optional[Template] = None,
        pre_infer_hook=None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        pass
