# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import inspect
import os
from copy import deepcopy
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import sglang as sgl
import torch
from PIL import Image
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from transformers import GenerationConfig

from swift.metrics import Metric
from swift.model import get_model_info_meta, get_processor
from swift.template import Template
from swift.utils import get_logger
from .infer_engine import InferEngine
from .protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
                       ChatCompletionStreamResponse, ChatMessage, DeltaMessage, EmbeddingResponse,
                       EmbeddingResponseData, InferRequest, RequestConfig, random_uuid)
from .utils import InferStreamer

logger = get_logger()


class SglangEngine(InferEngine):

    def __init__(
        self,
        model_id_or_path: str,
        *,
        template: Optional[Template] = None,
        torch_dtype: Optional[torch.dtype] = None,
        model_type: Optional[str] = None,
        use_hf: Optional[bool] = None,
        hub_token: Optional[str] = None,
        revision: Optional[str] = None,
        # engine kwargs
        tp_size: int = 1,
        pp_size: int = 1,
        dp_size: int = 1,
        ep_size: int = 1,
        enable_ep_moe: bool = False,
        mem_fraction_static: Optional[float] = None,
        context_length: Optional[int] = None,
        disable_cuda_graph: bool = False,
        quantization: Optional[str] = None,
        task_type: Optional[str] = None,
        kv_cache_dtype: str = 'auto',
        enable_dp_attention: bool = False,
        disable_custom_all_reduce: bool = True,
        speculative_algorithm: Optional[str] = None,
        speculative_num_steps: Optional[int] = None,
        speculative_eagle_topk: Optional[int] = None,
        speculative_num_draft_tokens: Optional[int] = None,
        log_level='error',
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model_id_or_path = model_id_or_path
        self.torch_dtype = torch_dtype
        self.model_type = model_type
        self.use_hf = use_hf
        self.hub_token = hub_token
        self.revision = revision
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = dp_size
        self.ep_size = ep_size
        self.enable_ep_moe = enable_ep_moe
        self.mem_fraction_static = mem_fraction_static
        self.context_length = context_length
        self.disable_cuda_graph = disable_cuda_graph
        self.quantization = quantization
        self.task_type = task_type
        self.kv_cache_dtype = kv_cache_dtype
        self.enable_dp_attention = enable_dp_attention
        self.disable_custom_all_reduce = disable_custom_all_reduce
        self.speculative_algorithm = speculative_algorithm
        self.speculative_num_steps = speculative_num_steps
        self.speculative_eagle_topk = speculative_eagle_topk
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        self.log_level = log_level
        if template is None:
            processor = self._get_processor()
            template = self._get_template(processor)
        else:
            get_model_info_meta(
                model_id_or_path, hub_token=hub_token, use_hf=use_hf, revision=revision, download_model=True)
        super().__init__(template)
        self._prepare_server_args(engine_kwargs)
        self.engine = sgl.Engine(server_args=self.server_args)
        self._load_generation_config()
        if speculative_num_draft_tokens is not None:
            self.max_tokens_offset = -speculative_num_draft_tokens

    def _get_processor(self):
        return get_processor(
            model_id_or_path=self.model_id_or_path,
            torch_dtype=self.torch_dtype,
            download_model=True,
            model_type=self.model_type,
            use_hf=self.use_hf,
            hub_token=self.hub_token,
            revision=self.revision,
            task_type=self.task_type)

    def _prepare_server_args(self, engine_kwargs):
        if engine_kwargs is None:
            engine_kwargs = {}
        if self.context_length is not None:
            self.max_model_len = self.context_length
            logger.info(f'Setting max_model_len: {self.context_length}')
        if self.max_model_len is not None:
            self.max_model_len -= 1
        parameters = inspect.signature(ServerArgs).parameters
        if 'pp_size' in parameters:
            engine_kwargs['pp_size'] = self.pp_size
        if 'enable_ep_moe' in parameters:
            engine_kwargs['enable_ep_moe'] = self.enable_ep_moe
        self.server_args = ServerArgs(
            model_path=self.model_dir,
            dtype=self.model_info.torch_dtype,
            tp_size=self.tp_size,
            dp_size=self.dp_size,
            ep_size=self.ep_size,
            mem_fraction_static=self.mem_fraction_static,
            context_length=self.context_length,
            disable_cuda_graph=self.disable_cuda_graph,
            quantization=self.quantization,
            kv_cache_dtype=self.kv_cache_dtype,
            enable_dp_attention=self.enable_dp_attention,
            disable_custom_all_reduce=self.disable_custom_all_reduce,
            speculative_algorithm=self.speculative_algorithm,
            speculative_num_steps=self.speculative_num_steps,
            speculative_eagle_topk=self.speculative_eagle_topk,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
            log_level=self.log_level,
            skip_tokenizer_init=True,
            trust_remote_code=True,
            **engine_kwargs,
        )
        if self.task_type == 'embedding':
            self.server_args.is_embedding = True

    def _load_generation_config(self) -> None:
        generation_config_path = os.path.join(self.model_dir, 'generation_config.json')
        if os.path.isfile(generation_config_path):
            generation_config = GenerationConfig.from_pretrained(self.model_dir)
        else:
            generation_config = GenerationConfig()
        kwargs = generation_config.to_dict()
        top_k = kwargs.get('top_k')
        if top_k == 0:
            kwargs['top_k'] = -1

        parameters = inspect.signature(SamplingParams).parameters
        self.generation_config = {k: v for k, v in kwargs.items() if k in parameters and v is not None}

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

    def _add_stop_words(self, generation_config: Dict[str, Any], request_config: RequestConfig) -> None:
        template_meta = self.template.template_meta
        stop_words = (request_config.stop or []) + (self.generation_config.get('stop') or []) + template_meta.stop_words
        generation_config['stop_token_ids'] = self._get_stop_token_ids(stop_words)

    def _create_chat_completion_response(self, output, inputs, return_details: bool = False):
        assert output is not None
        meta_info = output['meta_info']
        usage_info = self._get_usage_info(meta_info['prompt_tokens'], meta_info['completion_tokens'])
        response = self.template.decode(output['output_ids'])
        toolcall = self._get_toolcall(response)
        token_ids = output['output_ids'] if return_details else None
        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),
            finish_reason=meta_info['finish_reason']['type'],
            logprobs=None,
            token_ids=token_ids)
        prompt_token_ids = None
        images_size = None
        if return_details:
            prompt_token_ids = output.get('prompt_token_ids')
            images = inputs['template_inputs'].images
            if all(isinstance(image, Image.Image) for image in images):
                images_size = [image.size for image in images]
        return ChatCompletionResponse(
            model=self.model_name,
            choices=[choice],
            usage=usage_info,
            id=random_uuid(),
            prompt_token_ids=prompt_token_ids,
            images_size=images_size)

    def infer(
        self,
        infer_requests: List[InferRequest],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        use_tqdm: Optional[bool] = None,
    ) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
        return super().infer(infer_requests, request_config, metrics, use_tqdm=use_tqdm)

    async def infer_async(self,
                          infer_request: InferRequest,
                          request_config: Optional[RequestConfig] = None,
                          *,
                          pre_infer_hook=None,
                          **kwargs) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        request_config = deepcopy(request_config or RequestConfig())
        self.template.set_mode('sglang')
        loop = asyncio.get_running_loop()
        with torch.inference_mode():
            inputs = await loop.run_in_executor(None, self.template.encode, infer_request, True)
        if self.task_type == 'embedding':
            inputs.pop('length', None)
        self.set_default_max_tokens(request_config, inputs)
        generation_config = self._prepare_generation_config(request_config)
        self._add_stop_words(generation_config, request_config)
        kwargs.update({'inputs': inputs, 'generation_config': generation_config, 'request_config': request_config})
        if pre_infer_hook:
            kwargs = pre_infer_hook(kwargs)
        if request_config.stream:
            return self._infer_stream_async(**kwargs)
        elif self.task_type == 'embedding':
            kwargs.pop('generation_config', None)
            return await self._infer_embedding_async(**kwargs)
        else:
            return await self._infer_full_async(**kwargs)

    async def _infer_embedding_async(self, inputs: Dict[str, Any], **kwargs) -> EmbeddingResponse:
        from sglang.srt.managers.io_struct import EmbeddingReqInput
        obj = EmbeddingReqInput(
            input_ids=inputs['input_ids'], image_data=inputs.get('images'), audio_data=inputs.get('audios'))
        generator = self.engine.tokenizer_manager.generate_request(obj, None)
        output = await generator.__anext__()
        usage_info = self._get_usage_info(output['meta_info']['prompt_tokens'], 0)
        return EmbeddingResponse(
            model=self.model_name,
            data=[EmbeddingResponseData(embedding=output['embedding'])],
            usage=usage_info,
            id=random_uuid())

    async def _infer_full_async(self, inputs: Dict[str, Any], generation_config: Dict[str, Any],
                                request_config: RequestConfig) -> ChatCompletionResponse:
        engine_inputs = {k: v for k, v in inputs.items() if k != 'template_inputs'}
        output = await self.engine.async_generate(**engine_inputs, sampling_params=generation_config)
        output['prompt_token_ids'] = inputs['input_ids']
        return self._create_chat_completion_response(output, inputs, request_config.return_details)

    async def _infer_stream_async(self, inputs: Dict[str, Any], generation_config: Dict[str, Any],
                                  **kwargs) -> AsyncIterator[ChatCompletionStreamResponse]:
        engine_inputs = {k: v for k, v in inputs.items() if k != 'template_inputs'}
        result_generator = await self.engine.async_generate(
            **engine_inputs, sampling_params=generation_config, stream=True)
        infer_streamer = InferStreamer(self.template)
        async for output in result_generator:
            res = self._create_chat_completion_stream_response(output, infer_streamer)
            if res is None:
                continue
            yield res

    def _create_chat_completion_stream_response(self, output, infer_streamer) -> Optional[ChatCompletionStreamResponse]:
        assert output is not None
        meta_info = output['meta_info']
        finish_reason = meta_info['finish_reason']
        is_finished = finish_reason is not None
        delta_text = infer_streamer.get_printable_text(output['output_ids'], is_finished)
        if not delta_text and not is_finished:
            return
        toolcall = None
        if is_finished:
            finish_reason = finish_reason['type']
            toolcall = self._get_toolcall(self.template.decode(output['output_ids']))
        meta_info = output['meta_info']
        usage_info = self._get_usage_info(meta_info['prompt_tokens'], meta_info['completion_tokens'])
        # TODO: logprobs
        choice = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(role='assistant', content=delta_text, tool_calls=toolcall),
            finish_reason=finish_reason,
            logprobs=None)
        return ChatCompletionStreamResponse(model=self.model_name, choices=[choice], usage=usage_info)
