import asyncio
import inspect
import os
from copy import deepcopy
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import sglang as sgl
import torch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from transformers import GenerationConfig

from swift.llm import InferRequest, Template, TemplateMeta, get_model_tokenizer
from swift.plugin import Metric
from ..protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
                        ChatCompletionStreamResponse, ChatMessage, DeltaMessage, RequestConfig, random_uuid)
from .infer_engine import InferEngine


class SglangEngine(InferEngine):

    def __init__(
        self,
        model_id_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        *,
        model_type: Optional[str] = None,
        use_hf: Optional[bool] = None,
        hub_token: Optional[str] = None,
        revision: Optional[str] = None,
        # engine kwargs
        tp_size: int = 1,
        pp_size: int = 1,
        dp_size: int = 1,
        ep_size: int = 1,
        mem_fraction_static: Optional[float] = None,
        context_length: Optional[int] = None,
        disable_cuda_graph: bool = False,
        quantization: Optional[str] = None,
        kv_cache_dtype: str = 'auto',
        enable_dp_attention: bool = False,
        disable_custom_all_reduce: bool = True,
        log_level='error',
        engine_kwargs: Optional[Dict[str, Any]] = None,
        template: Optional[Template] = None,
    ):
        if engine_kwargs is None:
            engine_kwargs = {}
        self.processor = get_model_tokenizer(
            model_id_or_path,
            torch_dtype,
            load_model=False,
            download_model=True,
            model_type=model_type,
            use_hf=use_hf,
            hub_token=hub_token,
            revision=revision)[1]
        self._post_init(template)
        if self.max_model_len is not None:
            self.max_model_len -= 1
        parameters = inspect.signature(ServerArgs).parameters
        if 'pp_size' in parameters:
            engine_kwargs['pp_size'] = pp_size
        self.server_args = ServerArgs(
            model_path=self.model_dir,
            dtype=self.model_info.torch_dtype,
            tp_size=tp_size,
            dp_size=dp_size,
            ep_size=ep_size,
            mem_fraction_static=mem_fraction_static,
            context_length=context_length,
            disable_cuda_graph=disable_cuda_graph,
            quantization=quantization,
            kv_cache_dtype=kv_cache_dtype,
            enable_dp_attention=enable_dp_attention,
            disable_custom_all_reduce=disable_custom_all_reduce,
            log_level=log_level,
            **engine_kwargs,
        )
        self.engine = sgl.Engine(server_args=self.server_args)
        self._load_generation_config()

    def _load_generation_config(self) -> None:
        generation_config_path = os.path.join(self.model_dir, 'generation_config.json')
        if os.path.isfile(generation_config_path):
            generation_config = GenerationConfig.from_pretrained(self.model_dir)
            kwargs = generation_config.to_dict()
            top_k = kwargs.get('top_k')
            if top_k == 0:
                kwargs['top_k'] = -1

            parameters = inspect.signature(SamplingParams).parameters
            for k, v in kwargs.copy().items():
                if k not in parameters or v is None:
                    kwargs.pop(k)
            self.generation_config = kwargs
        else:
            self.generation_config = {}

    def _prepare_generation_config(self, request_config: RequestConfig) -> Dict[str, Any]:
        kwargs = {'max_new_tokens': request_config.max_tokens}
        for key in ['temperature', 'top_k', 'top_p', 'repetition_penalty']:
            new_value = getattr(request_config, key)
            if new_value is None:
                kwargs[key] = self.generation_config.get(key)
            else:
                kwargs[key] = new_value
        for key in ['n', 'frequency_penalty', 'presence_penalty']:
            kwargs[key] = getattr(request_config, key)

        return kwargs

    def _add_stop_words(self, generation_config: Dict[str, Any], request_config: RequestConfig,
                        template_meta: TemplateMeta) -> None:
        stop_words = (request_config.stop or []) + (self.generation_config.get('stop') or []) + template_meta.stop_words
        generation_config['stop'] = self._get_stop_words(stop_words)

    def _create_chat_completion_response(self, output, template):
        assert output is not None
        meta_info = output['meta_info']
        usage_info = self._get_usage_info(meta_info['prompt_tokens'], meta_info['completion_tokens'])
        response = output['text']
        toolcall = self._get_toolcall(response, template)
        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),
            finish_reason=meta_info['finish_reason']['type'],
            logprobs=None)
        return ChatCompletionResponse(model=self.model_name, choices=[choice], usage=usage_info, id=random_uuid())

    def infer(
        self,
        infer_requests: List[InferRequest],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        template: Optional[Template] = None,
        use_tqdm: Optional[bool] = None,
    ) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
        return super().infer(infer_requests, request_config, metrics, template=template, use_tqdm=use_tqdm)

    async def infer_async(self,
                          infer_request: InferRequest,
                          request_config: Optional[RequestConfig] = None,
                          *,
                          template: Optional[Template] = None,
                          pre_infer_hook=None,
                          **kwargs) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        request_config = deepcopy(request_config or RequestConfig())
        if template is None:
            template = self.default_template

        template.set_mode('pt')
        loop = asyncio.get_running_loop()
        with torch.inference_mode():
            inputs = await loop.run_in_executor(None, template.encode, infer_request)

        self.set_default_max_tokens(request_config, inputs)
        generation_config = self._prepare_generation_config(request_config)
        self._add_stop_words(generation_config, request_config, template.template_meta)
        kwargs.update({'template': template, 'inputs': inputs, 'generation_config': generation_config})
        if pre_infer_hook:
            kwargs = pre_infer_hook(kwargs)
        if request_config.stream:
            return self._infer_stream_async(**kwargs)
        else:
            return await self._infer_full_async(**kwargs)

    async def _infer_full_async(self, template: Template, inputs: Dict[str, Any],
                                generation_config: Dict[str, Any]) -> ChatCompletionResponse:
        output = await self.engine.async_generate(**inputs, sampling_params=generation_config)
        return self._create_chat_completion_response(output, template)

    async def _infer_stream_async(self, template: Template, inputs: Dict[str, Any],
                                  generation_config: Dict[str, Any]) -> AsyncIterator[ChatCompletionStreamResponse]:
        result_generator = await self.engine.async_generate(**inputs, sampling_params=generation_config, stream=True)
        idx = [0]
        async for output in result_generator:
            res = self._create_chat_completion_stream_response(output, template, generation_config, idx)
            if res is None:
                continue
            yield res

    def _create_chat_completion_stream_response(self, output, template, generation_config,
                                                idx) -> Optional[ChatCompletionStreamResponse]:
        assert output is not None
        response = output['text']
        meta_info = output['meta_info']
        finish_reason = meta_info['finish_reason']
        delta_text = response[idx[0]:]
        idx[0] = len(response)
        if not delta_text:
            return
        if finish_reason:
            finish_reason = finish_reason['type']
            toolcall = self._get_toolcall(response, template)
        else:
            toolcall = None
        meta_info = output['meta_info']
        usage_info = self._get_usage_info(meta_info['prompt_tokens'], meta_info['completion_tokens'])
        # TODO: logprobs
        choice = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(role='assistant', content=delta_text, tool_calls=toolcall),
            finish_reason=finish_reason,
            logprobs=None)
        return ChatCompletionStreamResponse(model=self.model_name, choices=[choice], usage=usage_info)
