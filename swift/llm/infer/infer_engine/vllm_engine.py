# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import inspect
import os
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import torch
from packaging import version
from PIL import Image
from tqdm import tqdm
from transformers import GenerationConfig
from transformers.utils import is_torch_npu_available

from swift.llm import InferRequest, Template, TemplateMeta, get_model_tokenizer
from swift.plugin import Metric
from swift.utils import get_device, get_dist_setting, get_logger, is_dist
from ..protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
                        ChatCompletionStreamResponse, ChatMessage, DeltaMessage, EmbeddingResponse,
                        EmbeddingResponseData, RequestConfig, random_uuid)
from .infer_engine import InferEngine
from .patch import patch_auto_config, patch_auto_tokenizer
from .utils import AdapterRequest, InferStreamer, patch_npu_vllm, patch_vllm_memory_leak

logger = get_logger()
try:
    # After setting the environment variables, import vllm. This way of writing allows lint to pass.
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '86400'
    import vllm
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams, EngineArgs, LLMEngine
    from vllm.pooling_params import PoolingParams
except Exception:
    raise

try:
    from vllm.reasoning import ReasoningParserManager
except ImportError:
    ReasoningParserManager = None

dtype_mapping = {torch.float16: 'float16', torch.bfloat16: 'bfloat16', torch.float32: 'float32'}


class VllmEngine(InferEngine):

    def __init__(
        self,
        model_id_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        *,
        adapters: List[str] = None,
        use_async_engine: bool = False,
        model_type: Optional[str] = None,
        use_hf: Optional[bool] = None,
        hub_token: Optional[str] = None,
        revision: Optional[str] = None,
        # engine_kwargs
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        enable_expert_parallel: bool = False,
        max_model_len: Optional[int] = None,
        max_num_seqs: int = 256,
        disable_custom_all_reduce: bool = True,
        enforce_eager: bool = False,
        limit_mm_per_prompt: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        task_type: Optional[str] = None,  # embedding
        disable_cascade_attn: bool = False,
        load_format: str = 'auto',
        # lora
        enable_lora: bool = False,
        max_loras: int = 1,
        max_lora_rank: int = 16,
        enable_prefix_caching: bool = False,
        enable_sleep_mode: bool = False,
        distributed_executor_backend: Optional[str] = None,
        quantization: Optional[str] = None,
        # reasoning parser
        reasoning_parser: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        template: Optional[Template] = None,
        num_labels: Optional[int] = None,
        reranker_use_activation: bool = True,
    ) -> None:
        if engine_kwargs is None:
            engine_kwargs = {}
        if isinstance(adapters, str):
            adapters = [adapters]
        self.default_adapter_request = None
        if isinstance(adapters, list) and adapters:
            assert len(adapters) == 1, 'Only one adapter is supported for now.'
            enable_lora = True
            self.default_adapter_request = AdapterRequest('default', adapters[0])
        patch_vllm_memory_leak()
        self.use_async_engine = use_async_engine
        self.reranker_use_activation = reranker_use_activation
        self.processor = get_model_tokenizer(
            model_id_or_path,
            torch_dtype,
            load_model=False,
            download_model=True,
            model_type=model_type,
            use_hf=use_hf,
            hub_token=hub_token,
            revision=revision,
            num_labels=num_labels,
            task_type=task_type)[1]
        self._post_init(template)

        self._prepare_engine_kwargs(
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            enable_expert_parallel=enable_expert_parallel,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            disable_custom_all_reduce=disable_custom_all_reduce,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt=limit_mm_per_prompt,
            enable_lora=enable_lora,
            max_loras=max_loras,
            max_lora_rank=max_lora_rank,
            enable_prefix_caching=enable_prefix_caching,
            seed=seed,
            distributed_executor_backend=distributed_executor_backend,
            enable_sleep_mode=enable_sleep_mode,
            quantization=quantization,
            task=task_type,
            disable_cascade_attn=disable_cascade_attn,
            **engine_kwargs,
        )
        context = nullcontext()
        if is_torch_npu_available() and (tensor_parallel_size == 1 or pipeline_parallel_size == 1):
            context = patch_npu_vllm(get_device())
        with context:
            self._prepare_engine()
        self._load_generation_config()
        self._fix_vllm_bug()
        self.patch_remove_log()
        self._request_count = 0
        self._prepare_reasoning_parser(reasoning_parser)

    def _prepare_engine(self) -> None:
        with patch_auto_tokenizer(self.tokenizer), patch_auto_config(self.config):
            llm_engine_cls = AsyncLLMEngine if self.use_async_engine else LLMEngine
            engine = llm_engine_cls.from_engine_args(self.engine_args)
        self.engine = engine

    def _prepare_engine_kwargs(
        self,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        enable_expert_parallel: bool = False,
        max_model_len: Optional[int] = None,
        max_num_seqs: int = 256,
        disable_custom_all_reduce: bool = True,
        enforce_eager: bool = False,
        limit_mm_per_prompt: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        enable_lora: bool = False,
        max_loras: int = 1,
        max_lora_rank: int = 16,
        enable_prefix_caching: bool = False,
        distributed_executor_backend: Optional[str] = None,
        enable_sleep_mode: bool = False,
        task: Optional[str] = None,
        disable_cascade_attn: bool = False,
        load_format: str = 'auto',
        **engine_kwargs,
    ) -> None:
        if task == 'embedding':
            task = 'embed'
        elif task == 'seq_cls':
            task = 'classify'
        elif task in ('reranker', 'generative_reranker'):
            task = 'score'
        disable_log_stats = engine_kwargs.pop('disable_log_stats', True)
        if self.use_async_engine:
            engine_cls = AsyncEngineArgs
        else:
            engine_cls = EngineArgs
        parameters = inspect.signature(engine_cls).parameters
        if self.use_async_engine and 'disable_log_requests' in parameters:
            engine_kwargs['disable_log_requests'] = True
        if 'enable_lora' in parameters and enable_lora:
            engine_kwargs['enable_lora'] = enable_lora
            engine_kwargs['max_loras'] = max_loras
            engine_kwargs['max_lora_rank'] = max_lora_rank
        else:
            assert not enable_lora, 'The current version of vLLM does not support `enable_lora`. Please upgrade vLLM.'

        if 'limit_mm_per_prompt' in parameters and limit_mm_per_prompt:
            engine_kwargs['limit_mm_per_prompt'] = limit_mm_per_prompt
        else:
            assert not limit_mm_per_prompt, (
                'The current version of vLLM does not support `limit_mm_per_prompt`. Please upgrade vLLM.')
        for key in ['enable_expert_parallel', 'enable_sleep_mode', 'disable_cascade_attn', 'load_format']:
            if key in parameters:
                engine_kwargs[key] = locals()[key]
            else:
                logger.warning(f'The current version of vLLM does not support `{key}`. Ignored.')
        for key in ['task', 'seed']:
            val = locals()[key]
            if val is not None:
                engine_kwargs[key] = val

        model_info = self.model_info
        arch_mapping = {'deepseek_vl2': ['DeepseekVLV2ForCausalLM'], 'glm4v': ['GLM4VForCausalLM']}
        if self.model_meta.model_type in arch_mapping:
            architectures = arch_mapping[self.model_meta.model_type]
            engine_kwargs['hf_overrides'] = {'architectures': architectures}
        self.default_template.set_mode('vllm')
        engine_kwargs.update(self.default_template.prepare_engine_kwargs())
        engine_args = engine_cls(
            model=self.model_dir,
            dtype=dtype_mapping[model_info.torch_dtype],
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            disable_log_stats=disable_log_stats,
            disable_custom_all_reduce=disable_custom_all_reduce,
            enforce_eager=enforce_eager,
            trust_remote_code=True,
            enable_prefix_caching=enable_prefix_caching,
            distributed_executor_backend=distributed_executor_backend,
            **engine_kwargs,
        )
        self.engine_args = engine_args
        self.enable_lora = enable_lora
        if max_model_len is not None:
            self.max_model_len = max_model_len
            logger.info(f'Setting max_model_len: {max_model_len}')

    def _prepare_reasoning_parser(self, reasoning_parser: Optional[str]) -> None:
        self.reasoning_parser = None
        if not reasoning_parser:
            return

        # Validate reasoning_parser if provided
        if ReasoningParserManager is None:
            raise ImportError('the version of vLLM is too old, please upgrade vLLM')

        valid_reasoning_parsers = list(ReasoningParserManager.reasoning_parsers.keys())
        if reasoning_parser not in valid_reasoning_parsers:
            raise ValueError(f'Invalid reasoning_parser: {reasoning_parser}. '
                             f'Available parsers: {valid_reasoning_parsers}')
        logger.info(f'Using reasoning_parser: {reasoning_parser}')

        reasoning_parser_cls = ReasoningParserManager.get_reasoning_parser(reasoning_parser)
        self.reasoning_parser = reasoning_parser_cls(self.tokenizer)

    def _fix_vllm_bug(self) -> None:
        # fix vllm==0.4 bug (very slow)
        tokenizer = self.tokenizer
        if self._version_ge(
                '0.4') and not self._version_ge('0.6') and not tokenizer.__class__.__name__.startswith('Cached'):
            _tokenizer_len = len(tokenizer)
            __old_len__ = tokenizer.__class__.__len__

            def __len__(self) -> int:
                if self is tokenizer:
                    return _tokenizer_len
                else:
                    return __old_len__(self)

            tokenizer.__class__.__len__ = __len__

    def _load_generation_config(self) -> None:
        generation_config_path = os.path.join(self.model_dir, 'generation_config.json')
        if os.path.isfile(generation_config_path):
            generation_config = GenerationConfig.from_pretrained(self.model_dir)
            kwargs = generation_config.to_dict()
            max_new_tokens = kwargs.get('max_new_tokens')
            if max_new_tokens is not None:
                kwargs['max_tokens'] = max_new_tokens
            top_k = kwargs.get('top_k')
            if top_k == 0:
                kwargs['top_k'] = -1
            parameters = inspect.signature(SamplingParams).parameters
            for k, v in kwargs.copy().items():
                if k not in parameters or v is None:
                    kwargs.pop(k)
            self.generation_config = SamplingParams(**kwargs)
        else:
            self.generation_config = SamplingParams()

    def _add_stop_words(self, generation_config: SamplingParams, request_config: RequestConfig,
                        template_meta: TemplateMeta) -> None:
        stop_words = (request_config.stop or []) + (self.generation_config.stop or []) + template_meta.stop_words
        generation_config.stop = self._get_stop_words(stop_words)
        # stop parameter is not effective in v1 engine (test version: vllm 0.8.5.post)
        generation_config.stop_token_ids = self._get_stop_token_ids(stop_words)

    @staticmethod
    def _version_ge(base_version: str):
        vllm_version = vllm.__version__
        if vllm_version is None or 'dev' in vllm_version:
            return True
        return version.parse(vllm_version) >= version.parse(base_version)

    def _add_adapter(self, adapter_request: Optional[AdapterRequest] = None):
        assert self.enable_lora, f'adapter_request: {adapter_request}, self.enable_lora: {self.enable_lora}'
        from vllm.lora.request import LoRARequest
        adapter_name = adapter_request.name
        adapter_path = adapter_request.path
        if adapter_name in self._adapters_pool:
            lora_request = self._adapters_pool[adapter_name]
        else:
            lora_request = LoRARequest(
                lora_name=adapter_name, lora_path=adapter_path, lora_int_id=len(self._adapters_pool) + 1)
            self._adapters_pool[adapter_name] = lora_request
        return lora_request

    def _add_request(self,
                     inputs: Dict[str, Any],
                     generation_config: SamplingParams,
                     request_id: str,
                     adapter_request: Optional[AdapterRequest] = None):
        kwargs = {}
        adapter_request = adapter_request or self.default_adapter_request
        if adapter_request:
            kwargs['lora_request'] = self._add_adapter(adapter_request)

        input_ids = inputs['input_ids']
        if self._version_ge('0.4.3'):
            llm_inputs = {'prompt_token_ids': input_ids}
            mm_data = {}
            for key in ['images', 'audios', 'videos']:
                media_data = inputs.get(key) or []
                if media_data:
                    if self._version_ge('0.6'):
                        mm_data[key.rstrip('s')] = media_data[0] if len(media_data) == 1 else media_data
                    else:
                        assert len(media_data) == 1, (
                            f'The current version of vllm only supports single {key}. Please upgrade to vllm >= 0.6.0')
                        mm_data[key.rstrip('s')] = media_data[0]
            if mm_data:
                llm_inputs['multi_modal_data'] = mm_data
            mm_processor_kwargs = inputs.get('mm_processor_kwargs')
            if mm_processor_kwargs:
                llm_inputs['mm_processor_kwargs'] = mm_processor_kwargs

            has_task_arg = 'task' in inspect.signature(PoolingParams).parameters
            has_activation_arg = 'activation' in inspect.signature(PoolingParams).parameters
            task_mapping = {
                'embedding': 'embed',
                'seq_cls': 'classify',
                'reranker': 'score',
                'generative_reranker': 'score',
            }
            if self.task_type in task_mapping:
                pooling_kwargs = {}
                if has_task_arg:
                    pooling_kwargs['task'] = task_mapping[self.task_type]
                if self.task_type in ('reranker', 'generative_reranker') and \
                        has_activation_arg and self.reranker_use_activation:
                    pooling_kwargs['activation'] = True
                pooling_params = PoolingParams(**pooling_kwargs)
                return self.engine.encode(llm_inputs, pooling_params, request_id)
            elif self.use_async_engine:
                return self.engine.generate(llm_inputs, generation_config, request_id, **kwargs)
            else:
                return self.engine.add_request(request_id, llm_inputs, generation_config, **kwargs)
        else:
            if self.use_async_engine:
                return self.engine.generate(None, generation_config, request_id, input_ids, **kwargs)
            else:
                return self.engine.add_request(request_id, None, generation_config, input_ids, **kwargs)

    def _get_logprobs(self,
                      logprobs_list: Optional[List[Dict[int, float]]],
                      token_ids: List[int],
                      top_logprobs: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if logprobs_list is None or len(token_ids) == 0:
            return None
        if len(token_ids) > 0:
            logprobs_list = logprobs_list[-len(token_ids):]
        for logprobs in logprobs_list:
            for token_id, logprob in logprobs.items():
                logprobs[token_id] = logprob.logprob
        return super()._get_logprobs(logprobs_list, token_ids, top_logprobs)

    def _prepare_generation_config(self, request_config: RequestConfig) -> SamplingParams:
        kwargs = {'max_tokens': request_config.max_tokens}
        for key in ['temperature', 'top_k', 'top_p', 'repetition_penalty']:
            new_value = getattr(request_config, key)
            if new_value is None:
                kwargs[key] = getattr(self.generation_config, key)
            else:
                kwargs[key] = new_value

        if request_config.logprobs:
            kwargs['logprobs'] = 1
            if request_config.top_logprobs is not None:
                kwargs['logprobs'] = max(1, request_config.top_logprobs)

        # TODO: beam search
        for key in ['n', 'best_of', 'frequency_penalty', 'presence_penalty', 'seed']:
            kwargs[key] = getattr(request_config, key)

        res = SamplingParams(**kwargs)

        if hasattr(res, 'output_kind') and res.n > 1:
            # fix n > 1 in V1 Engine
            from vllm.sampling_params import RequestOutputKind
            res.output_kind = RequestOutputKind.FINAL_ONLY
        return res

    @property
    def inner_model(self):
        return self.engine.model_executor.driver_worker.worker.model_runner.model

    @property
    def inner_model_executor(self):
        return self.engine.model_executor

    async def _infer_stream_async(
        self,
        template: Template,
        inputs: Dict[str, Any],
        generation_config: SamplingParams,
        adapter_request: Optional[AdapterRequest],
        request_config: RequestConfig,
    ) -> AsyncIterator[ChatCompletionStreamResponse]:
        request_id = random_uuid()
        result_generator = self._add_request(inputs, generation_config, request_id, adapter_request=adapter_request)
        infer_streamers = [InferStreamer(template) for _ in range(generation_config.n)]
        token_idxs = [0 for _ in range(generation_config.n)]
        async for result in result_generator:
            res = self._create_chat_completion_stream_response(result, template, request_config, request_id,
                                                               infer_streamers, token_idxs)
            if res is None:
                continue
            yield res

    def _create_chat_completion_stream_response(self, result, template, request_config, request_id, infer_streamers,
                                                token_idxs) -> Optional[ChatCompletionStreamResponse]:
        is_diff = False
        is_finished = False
        for output in result.outputs:
            output.token_ids = list(output.token_ids)
            output.delta_text = infer_streamers[output.index].get_printable_text(output.token_ids, output.finished())
            output.is_finished = output.finish_reason is not None
            is_diff |= bool(output.delta_text)
            is_finished |= output.is_finished
        if not is_diff and not is_finished:
            return

        num_generated_tokens = sum(len(output.token_ids) for output in result.outputs)
        usage_info = self._get_usage_info(len(result.prompt_token_ids), num_generated_tokens)
        choices = []
        previous_texts = [''] * len(result.outputs)
        for output in result.outputs:
            i = output.index
            logprobs = self._get_logprobs(output.logprobs, output.token_ids[token_idxs[i]:],
                                          request_config.top_logprobs)

            # Handle reasoning content in streaming
            delta_content = output.delta_text
            delta_reasoning_content = None

            if self.reasoning_parser and output.delta_text:
                try:
                    # Get token IDs for the delta (new tokens in this step)
                    delta_token_ids = output.token_ids[token_idxs[i]:]
                    previous_token_ids = output.token_ids[:token_idxs[i]]

                    # Get current accumulated text for this output
                    previous_text = previous_texts[i]
                    current_text = previous_text + output.delta_text
                    previous_texts[i] = current_text
                    # Extract reasoning content from the delta
                    delta_message = self.reasoning_parser.extract_reasoning_content_streaming(
                        previous_text, current_text, output.delta_text, previous_token_ids, output.token_ids,
                        delta_token_ids)

                    if delta_message:
                        delta_reasoning_content = delta_message.reasoning_content
                        if delta_message.content:
                            delta_content = delta_message.content
                        else:
                            delta_content = None

                except Exception as e:
                    logger.warning(f'Failed to extract reasoning content in streaming: {e}')
                    # Fallback to original delta_text
                    delta_content = output.delta_text
            token_idxs[i] = len(output.token_ids)

            toolcall = None
            if output.is_finished:
                toolcall = self._get_toolcall(template.decode(output.token_ids), template)

            choice = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(
                    role='assistant',
                    content=delta_content,
                    reasoning_content=delta_reasoning_content,
                    tool_calls=toolcall),
                finish_reason=output.finish_reason,
                logprobs=logprobs)
            choices.append(choice)
        return ChatCompletionStreamResponse(model=self.model_name, choices=choices, usage=usage_info, id=request_id)

    def _create_embedding_response(self, result, template, generation_config, request_id) -> EmbeddingResponse:
        assert result is not None
        embedding = result.outputs.data.cpu().numpy().tolist()
        usage_info = self._get_usage_info(len(result.prompt_token_ids), 0)
        return EmbeddingResponse(
            model=self.model_name, data=[EmbeddingResponseData(embedding=embedding)], usage=usage_info, id=request_id)

    def _create_chat_completion_response(
        self,
        result,
        inputs,
        template,
        request_config,
        request_id,
    ) -> ChatCompletionResponse:
        assert result is not None
        num_generated_tokens = sum(len(output.token_ids) for output in result.outputs)
        usage_info = self._get_usage_info(len(result.prompt_token_ids), num_generated_tokens)
        choices = []
        for output in result.outputs:
            output.token_ids = list(output.token_ids)
            response = template.decode(output.token_ids)

            # Extract reasoning content if reasoning_parser is enabled
            reasoning_content = None
            content = response
            if self.reasoning_parser:
                try:
                    reasoning_content, content = self.reasoning_parser.extract_reasoning_content(
                        response,
                        request=None  # We don't have the original request here
                    )
                except Exception as e:
                    logger.warning(f'Failed to extract reasoning content: {e}')
                    # Fallback to original response
                    content = response

            logprobs = self._get_logprobs(output.logprobs, output.token_ids, request_config.top_logprobs)
            toolcall = self._get_toolcall(content, template)  # Use content instead of response for tool calls
            token_ids = template.skip_stop_tokens(output.token_ids) if request_config.return_details else None
            choice = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(
                    role='assistant', content=content, reasoning_content=reasoning_content, tool_calls=toolcall),
                finish_reason=output.finish_reason,
                logprobs=logprobs,
                token_ids=token_ids)
            choices.append(choice)
        prompt_token_ids = None
        images_size = None
        if request_config.return_details:
            prompt_token_ids = result.prompt_token_ids
            images = inputs['template_inputs'].images
            if all(isinstance(image, Image.Image) for image in images):
                images_size = [image.size for image in images]
        return ChatCompletionResponse(
            model=self.model_name,
            choices=choices,
            usage=usage_info,
            id=request_id,
            prompt_token_ids=prompt_token_ids,
            images_size=images_size)

    def _create_seq_cls_response(
        self,
        result,
        template,
        request_config,
        request_id,
    ) -> ChatCompletionResponse:
        assert result is not None
        choices = []
        preds = result.outputs.data
        if preds.dim() == 1:
            preds = preds.unsqueeze(0)
        if self.task_type == 'seq_cls':
            top_logprobs = request_config.top_logprobs or 20
            preds, logprobs = template.decode_seq_cls(preds, top_logprobs)
        else:
            logprobs = [None] * len(preds)
        num_prompt_token_ids = 0
        num_generated_tokens = 0
        for i, pred in enumerate(preds):
            num_prompt_token_ids += len(result.prompt_token_ids)
            num_generated_tokens += 1
            if isinstance(pred, torch.Tensor):
                pred = pred.tolist()
            choices.append(
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role='assistant', content=pred, tool_calls=None),
                    finish_reason='stop',
                    logprobs=logprobs[i]))
        usage_info = self._get_usage_info(num_prompt_token_ids, num_generated_tokens)
        return ChatCompletionResponse(
            model=self.model_name,
            choices=choices,
            usage=usage_info,
            id=request_id,
            prompt_token_ids=result.prompt_token_ids)

    async def _infer_full_async(
        self,
        template: Template,
        inputs: Dict[str, Any],
        generation_config: SamplingParams,
        adapter_request: Optional[AdapterRequest],
        request_config: RequestConfig,
        request_id: Optional[str] = None,
    ) -> Union[ChatCompletionResponse, EmbeddingResponse]:
        if request_id is None:
            request_id = random_uuid()
        result_generator = self._add_request(inputs, generation_config, request_id, adapter_request=adapter_request)
        result = None
        async for result in result_generator:
            pass
        if self.task_type == 'embedding':
            return self._create_embedding_response(result, template, generation_config, request_id)
        elif self.task_type in ('seq_cls', 'reranker', 'generative_reranker'):
            return self._create_seq_cls_response(result, template, request_config, request_id)
        else:
            return self._create_chat_completion_response(result, inputs, template, request_config, request_id)

    def _batch_infer_stream(self, *args, **kwargs):
        if hasattr(self.engine, 'engine'):
            self.engine.engine.model_executor.parallel_worker_tasks = None
        return super()._batch_infer_stream(*args, **kwargs)

    def infer(
        self,
        infer_requests: List[InferRequest],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        template: Optional[Template] = None,
        use_tqdm: Optional[bool] = None,
        adapter_request: Optional[AdapterRequest] = None,
    ) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
        if self.use_async_engine:
            return super().infer(
                infer_requests,
                request_config,
                metrics,
                template=template,
                use_tqdm=use_tqdm,
                adapter_request=adapter_request,
            )
        else:
            request_config = deepcopy(request_config or RequestConfig())
            if request_config.stream and len(infer_requests) > 1:
                raise ValueError('If you want to use stream batch inference, you need to set use_async_engine to True.')
            if use_tqdm is None:
                use_tqdm = len(infer_requests) > 1
            rank = get_dist_setting()[0]
            if is_dist() and rank % self.engine_args.tensor_parallel_size != 0:
                use_tqdm = False
            if template is None:
                template = self.default_template
            template.set_mode('vllm')
            batched_inputs, error_list = self._batch_encode(
                infer_requests, template=template, strict=getattr(self, 'strict', True))
            request_id_list = []
            for i, inputs in enumerate(batched_inputs):
                request_id = str(self._request_count)
                request_id_list.append(request_id)
                self._request_count += 1
                _request_config = deepcopy(request_config)
                self.set_default_max_tokens(_request_config, inputs)
                generation_config = self._prepare_generation_config(_request_config)
                if generation_config.seed is not None:
                    generation_config.seed += i
                self._add_stop_words(generation_config, _request_config, template.template_meta)
                self._add_request(inputs, generation_config, request_id, adapter_request=adapter_request)
            prog_bar = tqdm(total=len(batched_inputs), dynamic_ncols=True, disable=not use_tqdm)
            outputs = {}
            if request_config.stream:

                def _gen_wrapper():
                    infer_streamers = [InferStreamer(template) for _ in range(generation_config.n)]
                    token_idxs = [0 for _ in range(generation_config.n)]
                    while self.engine.has_unfinished_requests():
                        result = self.engine.step()
                        if not result:
                            continue
                        result = result[0]
                        res = self._create_chat_completion_stream_response(result, template, request_config, request_id,
                                                                           infer_streamers, token_idxs)
                        if res is None:
                            continue
                        yield res
                        if result.finished:
                            break

                    self._update_metrics(res, metrics)

                return [_gen_wrapper()]
            else:
                while self.engine.has_unfinished_requests():
                    step_outputs = self.engine.step()
                    for output in step_outputs:
                        if output.finished:
                            outputs[output.request_id] = output
                            prog_bar.update()
                prog_bar.close()
                outputs = [outputs[request_id] for request_id in request_id_list]
                res = [
                    self._create_chat_completion_response(result, inputs, template, request_config, request_id)
                    for request_id, inputs, result in zip(request_id_list, batched_inputs, outputs)
                ]
                self._update_metrics(res, metrics)
                return self._add_error_list(res, error_list)

    async def infer_async(
        self,
        infer_request: InferRequest,
        request_config: Optional[RequestConfig] = None,
        *,
        template: Optional[Template] = None,
        adapter_request: Optional[AdapterRequest] = None,
        pre_infer_hook=None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        if not self.use_async_engine:
            raise ValueError('If you want to use `infer_async`, you need to pass `use_async_engine` as True.')
        request_config = deepcopy(request_config or RequestConfig())
        if template is None:
            template = self.default_template

        template.set_mode('vllm')
        loop = asyncio.get_running_loop()
        with torch.inference_mode():
            inputs = await loop.run_in_executor(None, template.encode, infer_request, True)
        self.set_default_max_tokens(request_config, inputs)
        generation_config = self._prepare_generation_config(request_config)
        self._add_stop_words(generation_config, request_config, template.template_meta)
        kwargs = {
            'template': template,
            'inputs': inputs,
            'generation_config': generation_config,
            'adapter_request': adapter_request,
            'request_config': request_config,
        }
        if hasattr(infer_request, 'uuid') and infer_request.uuid:
            # RolloutInferRequest
            kwargs.update({'request_id': infer_request.uuid})
        if pre_infer_hook:
            kwargs = pre_infer_hook(kwargs)
        if request_config.stream:
            return self._infer_stream_async(**kwargs)
        else:
            return await self._infer_full_async(**kwargs)

    @staticmethod
    def patch_remove_log():
        from vllm.engine import async_llm_engine
        if not hasattr(async_llm_engine, '_log_task_completion'):
            return

        async_llm_engine._origin_log_task_completion = async_llm_engine._log_task_completion

        def new_log_task_completion(task, error_callback) -> None:
            try:
                return_value = task.result()
                raise AssertionError(f'The engine background task should never finish without an '
                                     f'exception. {return_value}')
            except asyncio.exceptions.CancelledError:
                pass

        async_llm_engine._log_task_completion = new_log_task_completion
