# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import inspect
import os
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import torch
from packaging import version
from tqdm import tqdm
from transformers import GenerationConfig
from transformers.utils import is_torch_npu_available

from swift.llm import InferRequest, Template, TemplateMeta, get_model_tokenizer
from swift.plugin import Metric
from swift.utils import get_dist_setting, is_dist
from ..protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
                        ChatCompletionStreamResponse, ChatMessage, DeltaMessage, RequestConfig, random_uuid)
from .infer_engine import InferEngine
from .patch import patch_auto_config, patch_auto_tokenizer
from .utils import AdapterRequest, InferStreamer, patch_npu_vllm, patch_vllm_memory_leak

try:
    # After setting the environment variables, import vllm. This way of writing allows lint to pass.
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '86400'
    import vllm
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams, EngineArgs, LLMEngine
except Exception:
    raise

dtype_mapping = {torch.float16: 'float16', torch.bfloat16: 'bfloat16', torch.float32: 'float32'}


class VllmEngine(InferEngine):

    def __init__(
        self,
        model_id_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        *,
        use_async_engine: bool = False,
        model_type: Optional[str] = None,
        use_hf: Optional[bool] = None,
        hub_token: Optional[str] = None,
        revision: Optional[str] = None,
        # engine_kwargs
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        max_num_seqs: int = 256,
        disable_custom_all_reduce: bool = True,
        enforce_eager: bool = False,
        limit_mm_per_prompt: Optional[Dict[str, Any]] = None,
        device: str = 'auto',
        seed: Optional[int] = None,
        # lora
        enable_lora: bool = False,
        max_loras: int = 1,
        max_lora_rank: int = 16,
        enable_prefix_caching: bool = False,
        enable_sleep_mode: bool = False,
        distributed_executor_backend: Optional[str] = None,
        quantization: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        template: Optional[Template] = None,
    ) -> None:
        if engine_kwargs is None:
            engine_kwargs = {}
        patch_vllm_memory_leak()
        self.use_async_engine = use_async_engine
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

        self._prepare_engine_kwargs(
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            disable_custom_all_reduce=disable_custom_all_reduce,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt=limit_mm_per_prompt,
            enable_lora=enable_lora,
            max_loras=max_loras,
            max_lora_rank=max_lora_rank,
            enable_prefix_caching=enable_prefix_caching,
            device=device,
            seed=seed,
            distributed_executor_backend=distributed_executor_backend,
            enable_sleep_mode=enable_sleep_mode,
            quantization=quantization,
            **engine_kwargs,
        )
        context = nullcontext()
        if is_torch_npu_available() and (tensor_parallel_size == 1 or pipeline_parallel_size == 1):
            context = patch_npu_vllm(self.engine_args.device)
        with context:
            self._prepare_engine()
        self._load_generation_config()
        self._fix_vllm_bug()
        self.patch_remove_log()
        self._request_count = 0

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
        max_model_len: Optional[int] = None,
        max_num_seqs: int = 256,
        disable_custom_all_reduce: bool = True,
        enforce_eager: bool = False,
        limit_mm_per_prompt: Optional[Dict[str, Any]] = None,
        device: str = 'auto',
        enable_lora: bool = False,
        max_loras: int = 1,
        max_lora_rank: int = 16,
        enable_prefix_caching: bool = False,
        distributed_executor_backend: Optional[str] = None,
        enable_sleep_mode: bool = False,
        **engine_kwargs,
    ) -> None:
        disable_log_stats = engine_kwargs.pop('disable_log_stats', True)
        if self.use_async_engine:
            engine_cls = AsyncEngineArgs
            engine_kwargs['disable_log_requests'] = True
        else:
            engine_cls = EngineArgs
        parameters = inspect.signature(engine_cls).parameters
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
                'The current version of VLLM does not support `limit_mm_per_prompt`. Please upgrade VLLM.')
        if 'enable_sleep_mode' in parameters:
            engine_kwargs['enable_sleep_mode'] = enable_sleep_mode

        model_info = self.model_info
        if self.config.architectures is None:
            architectures = {'deepseek_vl2': ['DeepseekVLV2ForCausalLM']}[self.model_meta.model_type]
            engine_kwargs['hf_overrides'] = {'architectures': architectures}
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
            device=device,
            **engine_kwargs,
        )
        self.engine_args = engine_args
        self.enable_lora = enable_lora
        if max_model_len is not None:
            model_info.max_model_len = max_model_len

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

    @staticmethod
    def _version_ge(base_version: str):
        vllm_version = vllm.__version__
        if vllm_version is None or 'dev' in vllm_version:
            return True
        return version.parse(vllm_version) >= version.parse(base_version)

    def _add_request(self,
                     inputs: Dict[str, Any],
                     generation_config: SamplingParams,
                     request_id: str,
                     adapter_request: Optional[AdapterRequest] = None):
        kwargs = {}
        if self.enable_lora and adapter_request:
            from vllm.lora.request import LoRARequest
            adapter_name = adapter_request.name
            adapter_path = adapter_request.path
            if adapter_name in self._adapters_pool:
                kwargs['lora_request'] = self._adapters_pool[adapter_name]
            else:
                kwargs['lora_request'] = LoRARequest(
                    lora_name=adapter_name, lora_path=adapter_path, lora_int_id=len(self._adapters_pool) + 1)
                self._adapters_pool[adapter_name] = kwargs['lora_request']
        input_ids = inputs['input_ids']
        if self._version_ge('0.4.3'):
            llm_inputs = {'prompt_token_ids': input_ids}
            mm_data = {}
            for key in ['images', 'audios', 'videos']:
                media_data = inputs.get(key) or []
                if media_data:
                    if self._version_ge('0.6'):
                        mm_data = {key.rstrip('s'): media_data[0] if len(media_data) == 1 else media_data}
                    else:
                        assert len(media_data) == 1, (
                            f'The current version of vllm only supports single {key}. Please upgrade to vllm >= 0.6.0')
                        mm_data = {key.rstrip('s'): media_data[0]}
            if mm_data:
                llm_inputs['multi_modal_data'] = mm_data
            if self.use_async_engine:
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

        res.top_logprobs = request_config.top_logprobs
        return res

    @property
    def inner_model(self):
        return self.engine.model_executor.driver_worker.worker.model_runner.model

    @property
    def inner_model_executor(self):
        return self.engine.model_executor

    async def _infer_stream_async(self, template: Template, inputs: Dict[str, Any], generation_config: SamplingParams,
                                  **kwargs) -> AsyncIterator[ChatCompletionStreamResponse]:
        request_id = random_uuid()
        result_generator = self._add_request(inputs, generation_config, request_id, **kwargs)
        infer_streamers = [InferStreamer(template) for _ in range(generation_config.n)]
        token_idxs = [0 for _ in range(generation_config.n)]
        async for result in result_generator:
            res = self._create_chat_completion_stream_response(result, template, generation_config, request_id,
                                                               infer_streamers, token_idxs)
            if res is None:
                continue
            yield res

    def _create_chat_completion_stream_response(self, result, template, generation_config, request_id, infer_streamers,
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
        for output in result.outputs:
            logprobs = self._get_logprobs(output.logprobs, output.token_ids[token_idxs[output.index]:],
                                          generation_config.top_logprobs)
            token_idxs[output.index] = len(output.token_ids)
            toolcall = None
            if output.is_finished:
                toolcall = self._get_toolcall(template.decode(output.token_ids), template)
            choice = ChatCompletionResponseStreamChoice(
                index=output.index,
                delta=DeltaMessage(role='assistant', content=output.delta_text, tool_calls=toolcall),
                finish_reason=output.finish_reason,
                logprobs=logprobs)
            choices.append(choice)
        return ChatCompletionStreamResponse(model=self.model_name, choices=choices, usage=usage_info, id=request_id)

    def _create_chat_completion_response(self, result, template, generation_config,
                                         request_id) -> ChatCompletionResponse:
        assert result is not None
        num_generated_tokens = sum(len(output.token_ids) for output in result.outputs)
        usage_info = self._get_usage_info(len(result.prompt_token_ids), num_generated_tokens)
        choices = []
        for output in result.outputs:
            output.token_ids = list(output.token_ids)
            response = template.decode(output.token_ids)
            logprobs = self._get_logprobs(output.logprobs, output.token_ids, generation_config.top_logprobs)
            toolcall = self._get_toolcall(response, template)
            choice = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),
                finish_reason=output.finish_reason,
                logprobs=logprobs)
            choices.append(choice)
        return ChatCompletionResponse(model=self.model_name, choices=choices, usage=usage_info, id=request_id)

    async def _infer_full_async(
        self,
        template: Template,
        inputs: Dict[str, Any],
        generation_config: SamplingParams,
        adapter_request: Optional[AdapterRequest] = None,
    ) -> ChatCompletionResponse:
        request_id = random_uuid()
        result_generator = self._add_request(inputs, generation_config, request_id, adapter_request=adapter_request)
        result = None
        async for result in result_generator:
            pass
        return self._create_chat_completion_response(result, template, generation_config, request_id)

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
            self.set_default_max_tokens(request_config, batched_inputs)
            request_id_list = []
            for i, inputs in enumerate(batched_inputs):
                request_id = str(self._request_count)
                request_id_list.append(request_id)
                self._request_count += 1
                generation_config = self._prepare_generation_config(request_config)
                if generation_config.seed is not None:
                    generation_config.seed += i
                self._add_stop_words(generation_config, request_config, template.template_meta)
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
                        res = self._create_chat_completion_stream_response(result, template, generation_config,
                                                                           request_id, infer_streamers, token_idxs)
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
                    self._create_chat_completion_response(result, template, generation_config, request_id)
                    for request_id, result in zip(request_id_list, outputs)
                ]
                self._update_metrics(res, metrics)
                return res

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
            inputs = await loop.run_in_executor(None, template.encode, infer_request)
        self.set_default_max_tokens(request_config, inputs)
        generation_config = self._prepare_generation_config(request_config)
        self._add_stop_words(generation_config, request_config, template.template_meta)
        kwargs = {
            'template': template,
            'inputs': inputs,
            'generation_config': generation_config,
            'adapter_request': adapter_request,
        }
        if pre_infer_hook:
            kwargs = pre_infer_hook(kwargs)
        if request_config.stream:
            return self._infer_stream_async(**kwargs)
        else:
            return await self._infer_full_async(**kwargs)

    @staticmethod
    def patch_remove_log():
        from vllm.engine import async_llm_engine

        async_llm_engine._origin_log_task_completion = async_llm_engine._log_task_completion

        def new_log_task_completion(task, error_callback) -> None:
            try:
                return_value = task.result()
                raise AssertionError(f'The engine background task should never finish without an '
                                     f'exception. {return_value}')
            except asyncio.exceptions.CancelledError:
                pass

        async_llm_engine._log_task_completion = new_log_task_completion
