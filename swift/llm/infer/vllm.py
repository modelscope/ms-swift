import concurrent.futures
import inspect
import os
import time
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import vllm
from modelscope import GenerationConfig
from packaging import version
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from vllm import AsyncEngineArgs, AsyncLLMEngine, EngineArgs, LLMEngine, SamplingParams

from swift.utils import get_logger
from ..template import Template, get_template
from .base import InferEngine
from .patch import patch_auto_config, patch_auto_tokenizer
from .protocol import InferRequest

try:
    from vllm.lora.request import LoRARequest
except ImportError:
    pass

logger = get_logger()
dtype_mapping = {torch.float16: 'float16', torch.bfloat16: 'bfloat16', torch.float32: 'float32'}


class VllmEngine(InferEngine):

    def __init__(
            self,
            model_id_or_path: str,
            torch_dtype: Optional[torch.dtype] = None,
            *,
            model_type: Optional[str] = None,
            # engine_kwargs
            gpu_memory_utilization: float = 0.9,
            tensor_parallel_size: int = 1,
            max_num_seqs: int = 256,
            max_model_len: Optional[int] = None,
            disable_custom_all_reduce: bool = True,  # Default values different from vllm
            enforce_eager: bool = False,
            limit_mm_per_prompt: Optional[Dict[str, Any]] = None,
            # lora
            enable_lora: bool = False,
            max_loras: int = 1,
            max_lora_rank: int = 16,
            engine_kwargs: Optional[Dict[str, Any]] = None,  # extra
            **kwargs) -> None:
        self._init_env()
        self._prepare_model_tokenizer(model_id_or_path, torch_dtype, False, model_type=model_type, **kwargs)
        self._prepare_engine_kwargs(
            self.model_dir,
            self.torch_dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            disable_custom_all_reduce=disable_custom_all_reduce,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt=limit_mm_per_prompt,
            enable_lora=enable_lora,
            max_loras=max_loras,
            max_lora_rank=max_lora_rank,
            engine_kwargs=engine_kwargs)

        self._prepare_engine()
        self._prepare_generation_config()
        self._fix_vllm_bug()

    def _prepare_engine(self) -> None:
        with patch_auto_tokenizer(self.tokenizer), patch_auto_config(self.config):
            engine = AsyncLLMEngine.from_engine_args()
        self.engine = engine

    def _prepare_template(self):
        self.template = get_template(
            self.chat_template,
            tokenizer,
            args.system,
            args.max_length,
            args.truncation_strategy,
            model=llm_engine,
            tools_prompt=args.tools_prompt)

    def _prepare_engine_kwargs(
            self,
            model_dir: str,
            torch_dtype: torch.dtype,
            *,
            gpu_memory_utilization: float = 0.9,
            tensor_parallel_size: int = 1,
            max_num_seqs: int = 256,
            max_model_len: Optional[int] = None,
            disable_custom_all_reduce: bool = True,  # Default values different from vllm
            enforce_eager: bool = False,
            limit_mm_per_prompt: Optional[Dict[str, Any]] = None,
            enable_lora: bool = False,
            max_loras: int = 1,
            max_lora_rank: int = 16,
            engine_kwargs: Optional[Dict[str, Any]] = None) -> AsyncEngineArgs:
        if engine_kwargs is None:
            engine_kwargs = {}
        disable_log_stats = engine_kwargs.pop('disable_log_stats', True)
        engine_kwargs['disable_log_requests'] = True

        parameters = inspect.signature(AsyncEngineArgs.__init__).parameters
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

        torch_dtype = dtype_mapping[torch_dtype]
        engine_args = AsyncEngineArgs(
            model=model_dir,
            dtype=torch_dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            disable_log_stats=disable_log_stats,
            disable_custom_all_reduce=disable_custom_all_reduce,
            enforce_eager=enforce_eager,
            trust_remote_code=True,
            **engine_kwargs)
        self.engine_args = engine_args
        self.max_model_len = max_model_len

    @staticmethod
    def _init_env() -> None:
        try:
            from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except ImportError:
            pass
        # fix HTTPError bug (use model_dir)
        os.environ.pop('VLLM_USE_MODELSCOPE', None)
        if version.parse(vllm.__version__) >= version.parse('0.5.1'):
            os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    def _fix_vllm_bug(self) -> None:
        _engine = self.engine.engine
        # compatible with vllm==0.3.*
        if version.parse(vllm.__version__) >= version.parse('0.3'):
            assert isinstance(_engine.tokenizer.tokenizer, PreTrainedTokenizerBase)
            _engine.tokenizer.tokenizer = self.tokenizer

            # fix vllm==0.4 bug (very slow)
            if version.parse(vllm.__version__) >= version.parse('0.4'):
                _tokenizer_len = len(self.tokenizer)
                __old_len__ = self.tokenizer.__class__.__len__

                def __len__(self) -> int:
                    if self is self.tokenizer:
                        return _tokenizer_len
                    else:
                        return __old_len__(self)

                self.tokenizer.__class__.__len__ = __len__

        else:
            assert isinstance(_engine.tokenizer, PreTrainedTokenizerBase)
            _engine.tokenizer = self.tokenizer

    def _prepare_generation_config(self) -> None:
        generation_config_path = os.path.join(self.model_dir, 'generation_config.json')
        if os.path.isfile(generation_config_path):
            generation_config = GenerationConfig.from_pretrained(self.model_dir)
            kwargs = generation_config.to_dict()
            max_new_tokens = kwargs.get('max_new_tokens')
            if max_new_tokens is not None:
                kwargs['max_tokens'] = max_new_tokens
            parameters = inspect.signature(SamplingParams.__init__).parameters
            for k, v in kwargs.copy().items():
                if k not in parameters or v is None:
                    kwargs.pop(k)
            self.generation_config = SamplingParams(**kwargs)
        else:
            self.generation_config = SamplingParams()

    @torch.inference_mode()
    async def infer_async(
        self,
        template: Template,
        request_list: List[InferRequest],
        *,
        generation_config: Optional[SamplingParams] = None,
        lora_request: Optional['LoRARequest'] = None,
    ):
        pass

    @torch.inference_mode()
    def infer(self,
              template: Template,
              request_list: List[InferRequest],
              *,
              generation_config: Optional[Any] = None,
              generation_info: Optional[Dict[str, Any]] = None,
              max_batch_size: Optional[int] = None,
              lora_request: Optional[Any] = None,
              use_tqdm: bool = False,
              verbose: bool = False,
              prompt_prefix: str = '[PROMPT]',
              output_prefix: str = '[OUTPUT]',
              **kwargs) -> List[Dict[str, Any]]:
        if len(request_list) == 0:
            return []
        runtime = time.perf_counter()

        is_multimodal = getattr(self.llm_engine, 'is_multimodal', False)
        if is_multimodal and max_batch_size is None:
            max_batch_size = 512

        _inner_call = kwargs.get('_inner_call', False)
        if generation_info is None:
            generation_info = {}
        elif not _inner_call:
            generation_info.clear()
        if max_batch_size is not None and len(request_list) > max_batch_size:
            i = 0
            resp_list = []
            kwargs['_inner_call'] = True
            while i < len(request_list):
                resp_list += self.inference(
                    request_list[i:i + max_batch_size],
                    generation_config=generation_config,
                    generation_info=generation_info,
                    max_batch_size=max_batch_size,
                    lora_request=lora_request,
                    use_tqdm=use_tqdm,
                    verbose=verbose,
                    prompt_prefix=prompt_prefix,
                    output_prefix=output_prefix,
                    **kwargs)
                i += max_batch_size
            runtime = time.perf_counter() - runtime
            generation_info['runtime'] = runtime
            generation_info['samples/s'] = generation_info['num_samples'] / runtime
            generation_info['tokens/s'] = generation_info['num_generated_tokens'] / runtime
            return resp_list

        if generation_config is None:
            generation_config = getattr(self.llm_engine, 'generation_config', None) or VllmGenerationConfig()
        assert isinstance(generation_config, VllmGenerationConfig)
        request_list = deepcopy(request_list)
        generation_config = deepcopy(generation_config)

        old_num_samples = generation_info.get('num_samples', 0)
        resp_list, agent_state = self._prepare_vllm_request(
            request_list,
            generation_config=generation_config,
            generation_info=generation_info,
            lora_request=lora_request,
            use_tqdm=use_tqdm,
            **kwargs)

        tokenizer = self.template.tokenizer
        if use_tqdm:
            assert verbose is False
        prog_bar = tqdm(
            total=generation_info['num_samples'] - old_num_samples, dynamic_ncols=True, disable=not use_tqdm)
        outputs = []
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    prog_bar.update()
        prog_bar.close()

        for output in outputs:
            i = int(output.request_id)
            request = request_list[i]
            generate_ids = output.outputs[0].token_ids
            logprobs = output.outputs[0].logprobs
            response = self.template.generate_ids_to_response(generate_ids)
            query = request['query']
            messages = request['messages']
            if not agent_state[i][0]:
                messages.extend([
                    {
                        'role': 'user',
                        'content': query
                    },
                    {
                        'role': 'assistant',
                        'content': response
                    },
                ])
            else:
                messages[-1]['content'] = messages[-1]['content'] + response

            generation_info['num_generated_tokens'] += sum(len(_output.token_ids) for _output in output.outputs)
            resp_list[i] = {'response': response, 'messages': messages}
            if logprobs is not None:
                resp_list[i]['logprobs'] = logprobs
            if verbose:
                print(f'{prompt_prefix}{tokenizer.decode(output.prompt_token_ids, False)}{output_prefix}', end='')
                print(tokenizer.decode(output.outputs[0].token_ids, False))
        runtime = time.perf_counter() - runtime
        generation_info['runtime'] = runtime
        generation_info['samples/s'] = generation_info['num_samples'] / runtime
        generation_info['tokens/s'] = generation_info['num_generated_tokens'] / runtime
        return resp_list

    @torch.inference_mode()
    def inference_stream(
            self,
            request_list: List[Dict[str, Any]],
            *,
            generation_config: Optional[Any] = None,
            generation_info: Optional[Dict[str, Any]] = None,
            lora_request: Optional['LoRARequest'] = None,
            use_tqdm: bool = False,
            flush_steps: Optional[int] = None,  # Ensuring efficiency
            **kwargs) -> Iterator[List[Dict[str, Any]]]:
        """
        request_list: e.g. [{'query': 'hello!'}].
            The keys that can be included are: 'query', 'messages', 'system', 'images'.
        generation_config: Priority: generation_config > model.generation_config.
        return: e.g. [{'response': 'hi!', 'messages': [{'role': 'user', 'content': 'question'},
                                                       {'role': 'assistant', 'content': 'answer'}]}].
            The keys to be included will be: 'response', 'messages'.
        """
        if len(request_list) == 0:
            return
        start_runtime = time.perf_counter()
        if generation_config is None:
            generation_config = getattr(self.llm_engine, 'generation_config', None) or VllmGenerationConfig()
        assert isinstance(generation_config, VllmGenerationConfig)
        request_list = deepcopy(request_list)
        generation_config = deepcopy(generation_config)
        if generation_info is None:
            generation_info = {}
        else:
            generation_info.clear()

        resp_list, agent_state = self._prepare_vllm_request(
            request_list,
            generation_config=generation_config,
            generation_info=generation_info,
            lora_request=lora_request,
            use_tqdm=use_tqdm,
            **kwargs)

        if generation_config.use_beam_search:
            error_msg = 'Streaming generation does not support beam search.'
            raise ValueError(error_msg)

        n_finished = 0
        n_steps = 0
        if flush_steps is None:
            flush_steps = min(10, generation_info['num_samples'])
        print_idx_list = [[0] for _ in range(len(request_list))]
        num_generated_tokens = [0] * len(request_list)
        prog_bar = tqdm(total=generation_info['num_samples'], dynamic_ncols=True, disable=not use_tqdm)
        while self.llm_engine.has_unfinished_requests():
            is_flush = False
            n_steps += 1
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if not output.finished and n_steps % flush_steps != 0:
                    continue
                is_flush = True
                i = int(output.request_id)
                request = request_list[i]
                generate_ids = output.outputs[0].token_ids
                logprobs = output.outputs[0].logprobs
                safe_response = self.template.generate_ids_to_response(
                    generate_ids, output.finished, print_idx=print_idx_list[i])
                query = request['query']
                messages = request['messages']
                if resp_list[i] is None and not agent_state[i][0]:
                    messages.extend([
                        {
                            'role': 'user',
                            'content': None
                        },
                        {
                            'role': 'assistant',
                            'content': None
                        },
                    ])
                if not agent_state[i][0]:
                    messages[-2]['content'] = query
                    messages[-1]['content'] = safe_response
                else:
                    messages[-1]['content'] = messages[-1]['content'][:agent_state[i][1]] + safe_response

                n_gen_tokens = sum(len(_output.token_ids) for _output in output.outputs)
                generation_info['num_generated_tokens'] += n_gen_tokens - num_generated_tokens[i]
                num_generated_tokens[i] = n_gen_tokens

                resp_list[i] = {'response': safe_response, 'messages': messages}
                if logprobs is not None:
                    resp_list[i]['logprobs'] = logprobs
                if output.finished:
                    n_finished += 1
                    prog_bar.update()
            if not is_flush:
                continue
            runtime = time.perf_counter() - start_runtime
            generation_info['runtime'] = runtime
            generation_info['samples/s'] = n_finished / runtime
            generation_info['tokens/s'] = generation_info['num_generated_tokens'] / runtime
            yield resp_list
        prog_bar.close()

    @staticmethod
    def get_vllm_engine(
            model_type: str,
            torch_dtype: Optional[torch.dtype] = None,
            *,
            model_id_or_path: Optional[str] = None,
            revision: Optional[str] = None,
            gpu_memory_utilization: float = 0.9,
            tensor_parallel_size: int = 1,
            max_num_seqs: int = 256,
            max_model_len: Optional[int] = None,
            disable_custom_all_reduce: bool = True,  # Default values different from vllm
            enforce_eager: bool = False,
            engine_kwargs: Optional[Dict[str, Any]] = None,
            use_async: bool = False,
            # lora
            enable_lora: bool = False,
            max_loras: int = 1,
            max_lora_rank: int = 16,
            **kwargs) -> LLMEngine:
        model_dir = kwargs.pop('model_dir', None)  # compat with swift<1.7
        tokenizer = get_model_tokenizer(
            model_type,
            load_model=False,
            model_id_or_path=model_id_or_path,
            model_dir=model_dir,
            revision=revision,
            download_model=True)[1]
        model_dir = tokenizer.model_dir

        if engine_kwargs is None:
            engine_kwargs = {}
        dtype_mapping = {torch.float16: 'float16', torch.bfloat16: 'bfloat16', torch.float32: 'float32', None: 'auto'}
        dtype = dtype_mapping[torch_dtype]
        disable_log_stats = engine_kwargs.pop('disable_log_stats', True)

        if use_async:
            engine_args_cls = AsyncEngineArgs
            llm_engine_cls = AsyncLLMEngine
            engine_kwargs['disable_log_requests'] = True
        else:
            engine_args_cls = EngineArgs
            llm_engine_cls = LLMEngine

        parameters = inspect.signature(engine_args_cls.__init__).parameters
        if 'enable_lora' in parameters and enable_lora:
            engine_kwargs['enable_lora'] = enable_lora
            engine_kwargs['max_loras'] = max_loras
            engine_kwargs['max_lora_rank'] = max_lora_rank
        else:
            assert not enable_lora, 'The current version of VLLM does not support `enable_lora`. Please upgrade VLLM.'

        engine_args = engine_args_cls(
            model=model_dir,
            trust_remote_code=True,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            disable_log_stats=disable_log_stats,
            disable_custom_all_reduce=disable_custom_all_reduce,
            enforce_eager=enforce_eager,
            **engine_kwargs)
        try:
            from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except ImportError:
            pass
        # fix HTTPError bug (use model_dir)
        os.environ.pop('VLLM_USE_MODELSCOPE', None)
        if version.parse(vllm.__version__) >= version.parse('0.5.1'):
            os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

        llm_engine = llm_engine_cls.from_engine_args(engine_args)
        llm_engine.engine_args = engine_args
        llm_engine.model_dir = model_dir
        llm_engine.model_type = model_type

        if use_async:
            _engine = llm_engine.engine
        else:
            _engine = llm_engine
        model_config = _engine.model_config
        llm_engine.model_config = model_config
        llm_engine.dtype = model_config.dtype  # compat with pt
        llm_engine.max_model_len = model_config.max_model_len
        llm_engine.is_multimodal = tokenizer.is_multimodal
        # compatible with vllm==0.3.*
        if version.parse(vllm.__version__) >= version.parse('0.3'):
            assert isinstance(_engine.tokenizer.tokenizer, PreTrainedTokenizerBase)
            _engine.tokenizer.tokenizer = tokenizer

            # fix vllm==0.4 bug (very slow)
            if version.parse(vllm.__version__) >= version.parse('0.4'):
                _tokenizer_len = len(tokenizer)
                __old_len__ = tokenizer.__class__.__len__

                def __len__(self) -> int:
                    if self is tokenizer:
                        return _tokenizer_len
                    else:
                        return __old_len__(self)

                tokenizer.__class__.__len__ = __len__

        else:
            assert isinstance(_engine.tokenizer, PreTrainedTokenizerBase)
            _engine.tokenizer = tokenizer

        llm_engine.hf_tokenizer = tokenizer
        generation_config_path = os.path.join(model_dir, 'generation_config.json')
        if os.path.isfile(generation_config_path):
            generation_config = GenerationConfig.from_pretrained(model_dir)
            kwargs = generation_config.to_dict()
            max_new_tokens = kwargs.get('max_new_tokens')
            if max_new_tokens is not None:
                kwargs['max_tokens'] = max_new_tokens
            if version.parse(vllm.__version__) < version.parse('0.5.5'):
                parameters = inspect.signature(VllmGenerationConfig.__init__).parameters
            else:
                parameters = VllmGenerationConfig.__annotations__
            for k, v in kwargs.copy().items():
                if k not in parameters or v is None:
                    kwargs.pop(k)
            llm_engine.generation_config = VllmGenerationConfig(**kwargs)
        else:
            llm_engine.generation_config = VllmGenerationConfig()
        return llm_engine

    @staticmethod
    def _add_vllm_request(llm_engine: LLMEngine, inputs: Dict[str, Any], *, request_id: str,
                          generation_config: VllmGenerationConfig, **kwargs) -> None:
        input_ids = inputs['input_ids']
        if version.parse(vllm.__version__) >= version.parse('0.4.3'):
            llm_inputs = {'prompt_token_ids': input_ids}
            images = inputs.get('images') or []
            if images:
                assert len(images) == 1, 'Currently, only one image is supported.'
                llm_inputs['multi_modal_data'] = {'image': images[0]}
            llm_engine.add_request(request_id, llm_inputs, generation_config, **kwargs)
        else:
            llm_engine.add_request(request_id, None, generation_config, input_ids, **kwargs)

    @staticmethod
    def _prepare_vllm_request(llm_engine: LLMEngine,
                              template: Template,
                              request_list: List[Dict[str, Any]],
                              *,
                              generation_config: VllmGenerationConfig,
                              generation_info: Dict[str, Any],
                              lora_request: Optional['LoRARequest'] = None,
                              use_tqdm: bool = False,
                              **kwargs) -> Tuple[List[Optional[Dict[str, Any]]], List[Tuple[bool, int]]]:
        for key in ['num_prompt_tokens', 'num_generated_tokens', 'num_samples']:
            if key not in generation_info:
                generation_info[key] = 0

        tokenizer = template.tokenizer
        if tokenizer.eos_token is not None and tokenizer.eos_token not in generation_config.stop:
            generation_config.stop.append(tokenizer.eos_token)
        if isinstance(template.suffix[-1], str) and template.suffix[-1] not in generation_config.stop:
            generation_config.stop.append(template.suffix[-1])
        if isinstance(template.suffix[-1], list):
            token_str = tokenizer.decode(template.suffix[-1])
            if token_str not in generation_config.stop:
                generation_config.stop.append(token_str)

        parameters = inspect.signature(llm_engine.add_request).parameters
        add_request_kwargs = {}
        if 'lora_request' in parameters:
            add_request_kwargs['lora_request'] = lora_request
        else:
            assert lora_request is None, (
                'The current version of VLLM does not support `lora_request`. Please upgrade VLLM.')

        resp_list: List[Optional[Dict[str, Any]]] = [None] * len(request_list)
        agent_state = []
        is_multimodal = getattr(llm_engine, 'is_multimodal', False)
        max_workers = os.cpu_count()
        if not is_multimodal:
            use_tqdm = False
            max_workers = 1

        prog_bar = tqdm(request_list, dynamic_ncols=True, disable=not use_tqdm)

        def _prepare_inputs(request: Dict[str, Any]) -> Dict[str, Any]:
            messages = request.get('messages') or []
            query = request_list[0]['query']
            system = request_list[0]['system']
            system = [{'role': 'system', 'content': system}] if system else []
            messages = system + messages + [{'role': 'user', 'content': query}]
            # agent support
            is_observation = messages[-1]['content'].endswith(
                'Observation:') if messages and messages[-1]['content'] else False
            act_length = None
            if is_observation:
                messages[-1]['content'] = messages[-1]['content'] + request['query']
                act_length = len(messages[-1]['content'])
                request['query'] = None
            agent_state.append((is_observation, act_length))
            request['messages'] = messages

            inputs = template.encode(request)[0]
            prog_bar.update()
            return inputs

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(request_list))) as executor:
            futures = [executor.submit(_prepare_inputs, request) for request in request_list]
            concurrent.futures.wait(futures)
            inputs_list = [future.result() for future in futures]
        prog_bar.close()

        for i, (inputs, request) in enumerate(zip(inputs_list, request_list)):
            truncation_strategy = kwargs.pop('truncation_strategy', 'delete')
            if len(inputs) == 0 and truncation_strategy == 'delete':
                # input_ids exceeds `max_length`. Please increase the value of `max_length`.
                resp_list[i] = {'response': '', 'messages': request['messages']}
                continue
            generation_info['num_prompt_tokens'] += len(inputs['input_ids'])
            generation_info['num_samples'] += 1
            VLLMFramework._add_vllm_request(
                llm_engine, inputs, request_id=str(i), generation_config=generation_config, **add_request_kwargs)
        return resp_list, agent_state
