import json
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import time
from copy import deepcopy
from functools import partial
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple
from typing import Iterator

import torch
from modelscope import BitsAndBytesConfig, GenerationConfig
from transformers import PreTrainedModel, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
from transformers.utils import is_torch_npu_available
from queue import Empty, Queue
from swift.llm import InferArguments, Template, get_model_tokenizer, DeployArguments, get_template, StopWords
from swift.llm.infer.base import InferFramework
from swift.llm.model import ConfigReader
from swift.llm.model.utils import to_device
from swift.llm.template.base import StopWordsCriteria
from swift.llm.utils import set_generation_config, Messages
from swift.plugin.tuner import Tuner, extra_tuners
from swift.tuners import Swift
from swift.utils import (get_model_info, show_layers)
from swift import get_logger

logger = get_logger()


class TokenListIteratorStreamer(BaseStreamer):

    def __init__(self, timeout: Optional[float] = None):
        self.token_queue = Queue()  # Queue[int]
        self.stop_signal = None
        self.timeout = timeout

    def put(self, value: torch.Tensor) -> None:
        if value.ndim > 1:
            value = value[0]
        value = value.tolist()
        self.token_queue.put(value)

    def end(self) -> None:
        self.token_queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self) -> List[int]:
        value = self.token_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


class TransformersFramework(InferFramework):

    def __init__(self, args: InferArguments, use_async: bool = False, **kwargs):
        super().__init__(*self.prepare_model_template_hf(args, use_async, **kwargs))

    @staticmethod
    def prepare_model_template_hf(args: InferArguments, use_async: bool = False, **kwargs):
        if args.quant_method == 'hqq':
            from transformers import HqqConfig
            if args.hqq_dynamic_config_path is not None:
                cwd = os.getcwd()
                config_path = args.hqq_dynamic_config_path if os.path.isabs(
                    args.hqq_dynamic_config_path) else os.path.join(cwd, args.hqq_dynamic_config_path)
                with open(config_path, 'r') as json_file:
                    args.quant_config = HqqConfig(dynamic_config=json.load(json_file))
            else:
                if args.quantization_bit == 0:
                    logger.info("You haven't set the quantization_bit parameter; set it to 8.")
                    args.quantization_bit = 8
                args.quant_config = HqqConfig(nbits=args.quantization_bit, axis=args.hqq_axis)
        elif args.quant_method == 'eetq':
            from transformers import EetqConfig
            args.quant_config = EetqConfig('int8')
        device_map = args.device_map_config
        verbose = kwargs.get('verbose', True)
        automodel_class = kwargs.get('automodel_class')
        model_kwargs = {}
        if is_torch_npu_available():
            logger.info(f'device_count: {torch.npu.device_count()}')
            if device_map is None:
                device_map = 'npu:0'
        else:
            logger.info(f'device_count: {torch.cuda.device_count()}')
            if device_map is None:
                device_map = 'auto' if torch.cuda.device_count() > 1 else 'cuda:0'
        if device_map == 'auto':
            model_kwargs['low_cpu_mem_usage'] = True
        model_kwargs['device_map'] = device_map
        if args.device_max_memory:
            assert len(args.device_max_memory) == torch.cuda.device_count()
            model_kwargs['max_memory'] = {i: mem for i, mem in enumerate(args.device_max_memory)}

        # Loading Model and Tokenizer
        if hasattr(args, 'quant_config'):
            model_kwargs['quantization_config'] = args.quant_config
        elif args.load_in_8bit or args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                args.load_in_8bit,
                args.load_in_4bit,
                bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant)
            if args.bnb_4bit_compute_dtype is None:
                quantization_config.bnb_4bit_compute_dtype = None
            logger.info(f'quantization_config: {quantization_config.__dict__}')
            model_kwargs['quantization_config'] = quantization_config
        kwargs = {}
        if args.use_flash_attn is not None:
            kwargs['use_flash_attn'] = args.use_flash_attn
        model_id_or_path = None
        if args.sft_type == 'full' and args.ckpt_dir is not None:
            model_id_or_path = args.ckpt_dir
        elif args.model_id_or_path is not None:
            model_id_or_path = args.model_id_or_path
        if automodel_class is not None:
            kwargs['automodel_class'] = automodel_class
        if args.local_repo_path:
            kwargs['local_repo_path'] = args.local_repo_path
        if args.rope_scaling:
            kwargs['rope_scaling'] = args.rope_scaling
            kwargs['max_length'] = args.max_length
        model, tokenizer = get_model_tokenizer(
            args.model_type,
            args.torch_dtype,
            model_kwargs,
            model_id_or_path=model_id_or_path,
            revision=args.model_revision,
            quant_method=args.quant_method,
            **kwargs)
        if verbose:
            logger.info(f'model_config: {model.config}')

        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
            num_beams=args.num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
        set_generation_config(model, generation_config)
        logger.info(f'model.generation_config: {model.generation_config}')

        if model.generation_config.num_beams != 1:
            args.stream = False
            logger.info('Setting args.stream: False')
        if model.max_model_len is None:
            model.max_model_len = args.max_model_len
        elif args.max_model_len is not None:
            if args.max_model_len <= model.max_model_len:
                model.max_model_len = args.max_model_len
            else:
                raise ValueError('args.max_model_len exceeds the maximum max_model_len supported by the model.'
                                 f'args.max_model_len: {args.max_model_len}, model.max_model_len: {model.max_model_len}')

        if getattr(model, 'is_tuner_plugin', False):
            with open(os.path.join(args.ckpt_dir, 'sft_args.json'), 'r') as f:
                content = json.load(f)
            tuner: Tuner = extra_tuners[content['sft_type']]
            model = tuner.from_pretrained(model, args.ckpt_dir)
        elif args.is_adapter() and args.ckpt_dir is not None:
            if isinstance(args, DeployArguments) and args.lora_request_list is not None:
                logger.info(f'args.lora_request_list: {args.lora_request_list}')
                for lora_request in args.lora_request_list:
                    model = Swift.from_pretrained(
                        model, lora_request.lora_local_path, lora_request.lora_name, inference_mode=True)
            else:
                model = Swift.from_pretrained(model, args.ckpt_dir, inference_mode=True)
            model = model.to(model.dtype)
        model.requires_grad_(False)

        if verbose:
            show_layers(model)
            logger.info(model)
        logger.info(get_model_info(model))
        template: Template = get_template(
            args.template_type,
            tokenizer,
            args.system,
            args.max_length,
            args.truncation_strategy,
            model=model,
            tools_prompt=args.tools_prompt)
        args.system = template.default_system
        template.encode = partial(template.encode, streaming=args.streaming, dtype=model.dtype, device=model.device)
        logger.info(f'system: {args.system}')
        if args.overwrite_generation_config:
            assert args.ckpt_dir is not None, 'args.ckpt_dir is not specified.'
            model.generation_config.save_pretrained(args.ckpt_dir)

        return model, template

    @torch.inference_mode()
    def inference(self,
                  request_list: List[Dict[str, Any]],
                  *,
                  generation_config: Optional[Any] = None,
                  generation_info: Optional[Dict[str, Any]] = None,
                  max_batch_size: Optional[int] = None,
                  use_tqdm: bool = False,
                  verbose: bool = False,
                  prompt_prefix: str = '[PROMPT]',
                  output_prefix: str = '[OUTPUT]',
                  **kwargs) -> List[Dict[str, Any]]:
        """
        generation_config: Priority: generation_config > model.generation_config.
        """
        if args.stop_words:
            infer_kwargs['stop_words'] = args.stop_words
        model = engine
        runtime = time.perf_counter()
        if history is None:
            history = []
        else:
            history = deepcopy(history)
        if generation_config is None:
            generation_config = getattr(model, 'generation_config')
        generation_config = deepcopy(generation_config)
        if generation_info is None:
            generation_info = {}
        else:
            generation_info.clear()
        inputs, tokenizer_kwargs, token_len, example = self._prepare_inputs(
            self.llm_engine,
            self.template,
            query,
            history,
            system,
            images,
            generation_config=generation_config,
            generation_info=generation_info,
            stop_words=stop_words,
            adapter_names=adapter_names,
            **kwargs)
        if len(inputs) == 0:
            return '', history

        # agent support
        is_observation = history[-1][-1].endswith('Observation:') if history and history[-1][-1] else False
        if is_observation:
            history[-1][-1] = history[-1][-1] + query
            query = None

        if stream and not verbose:
            logger.warning('Please set verbose to True to support TextStreamer, or use `inference_stream.`')
            stream = False
        streamer = None
        tokenizer = template.tokenizer
        if stream:
            streamer = TextStreamer(tokenizer, skip_prompt=True)
        if verbose:
            if 'input_ids' in inputs:
                input_ids = inputs['input_ids']
                print(
                    f'{prompt_prefix}{safe_tokenizer_decode(tokenizer, input_ids[0], **tokenizer_kwargs)}{output_prefix}',
                    end='')
            else:
                print(f'[QUERY]{query}\n{output_prefix}', end='')

        return_dict = generation_config.return_dict_in_generate
        generate_ids = model.generate(streamer=streamer, generation_config=generation_config, **inputs)
        if return_dict:
            res = dict(generate_ids)
            generate_ids = generate_ids['sequences']
        generate_ids = self.template.get_generate_ids(generate_ids, token_len)
        generation_info['num_generated_tokens'] = len(generate_ids)
        if verbose and stream is False:
            response = tokenizer.decode(generate_ids, **tokenizer_kwargs)
            print(response)
        response = self.template.generate_ids_to_response(generate_ids, tokenizer_kwargs=tokenizer_kwargs)
        response = self.template.post_process_generate_response(response=response, example=example)
        if not is_observation:
            history.append([query, response])
        else:
            history[-1][-1] = history[-1][-1] + response
        runtime = time.perf_counter() - runtime
        generation_info['runtime'] = runtime
        generation_info['samples/s'] = 1 / runtime
        generation_info['tokens/s'] = generation_info['num_generated_tokens'] / runtime
        if return_dict:
            res['sequences'] = generate_ids
            res.update({'response': response, 'history': history})
            return res
        else:
            return response, history

    @torch.inference_mode()
    def inference_stream(self,
                         request_list: List[Dict[str, Any]],
                         *,
                         generation_config: Optional[Any] = None,
                         generation_info: Optional[Dict[str, Any]] = None,
                         lora_request: Optional['LoRARequest'] = None,
                         use_tqdm: bool = False,
                         flush_steps: Optional[int] = None,  # Ensuring efficiency
                         **kwargs) -> Iterator[List[Dict[str, Any]]]:
        """
        generation_config: Priority: generation_config > model.generation_config.
        """
        if args.stop_words:
            infer_kwargs['stop_words'] = args.stop_words
        start_runtime = time.perf_counter()
        if history is None:
            history = []
        else:
            history = deepcopy(history)
        if generation_config is None:
            generation_config = getattr(model, 'generation_config')
        generation_config = deepcopy(generation_config)
        if generation_info is None:
            generation_info = {}
        else:
            generation_info.clear()
        inputs, tokenizer_kwargs, token_len, example = self._prepare_inputs(
            self.model,
            self.template,
            query,
            history,
            system,
            images,
            generation_config=generation_config,
            generation_info=generation_info,
            stop_words=stop_words,
            adapter_names=adapter_names,
            **kwargs)
        if len(inputs) == 0:
            return '', history

        # agent support
        is_observation = history[-1][-1].endswith('Observation:') if history and history[-1][-1] else False
        if is_observation:
            history[-1][-1] = history[-1][-1] + query
            act_length = len(history[-1][-1])
            query = None

        if generation_config.num_beams != 1:
            error_msg = 'Streaming generation does not support beam search.'
            raise ValueError(error_msg)

        streamer = TokenListIteratorStreamer()
        return_dict = generation_config.return_dict_in_generate
        generation_kwargs = {'streamer': streamer, 'generation_config': generation_config, **inputs}
        result_queue = Queue()

        def _model_generate(*args, **kwargs):
            if is_torch_npu_available():
                torch.npu.set_device(self.llm_engine.device)
            res = self.llm_engine.generate(*args, **kwargs)
            result_queue.put(res)
            return res

        thread = Thread(target=_model_generate, kwargs=generation_kwargs)
        thread.start()
        raw_generate_ids, generate_ids = [], []

        if not is_observation:
            history.append(None)  # dummy

        print_idx = [0]
        first_num_space = [-1]

        is_finished = False
        while not is_finished:
            try:
                token_list = next(streamer)
                raw_generate_ids += token_list
            except StopIteration:
                is_finished = True
            res = {}
            generate_ids = self.template.get_generate_ids(torch.tensor(raw_generate_ids)[None], token_len)
            if return_dict and is_finished:
                thread.join()
                res = dict(result_queue.get())
                res['sequences'] = generate_ids
            generation_info['num_generated_tokens'] = len(generate_ids)
            response = self.template.generate_ids_to_response(
                generate_ids,
                is_finished,
                tokenizer_kwargs=tokenizer_kwargs,
                print_idx=print_idx,
                first_num_space=first_num_space)
            if not is_observation:
                history[-1] = [query, response]
            else:
                history[-1][-1] = history[-1][-1][:act_length] + response

            runtime = time.perf_counter() - start_runtime
            generation_info['runtime'] = runtime
            generation_info['samples/s'] = 1 / runtime
            generation_info['tokens/s'] = generation_info['num_generated_tokens'] / runtime
            if return_dict:
                res.update({'response': response, 'history': history})
                yield res
            else:
                yield [{
                    'response': response,
                    'history': history,
                }]

    @staticmethod
    def _prepare_inputs(
            model: PreTrainedModel,
            template: Template,
            query: str,
            history: History,
            system: Optional[str] = None,
            images: Optional[List[str]] = None,
            *,
            generation_config: GenerationConfig,
            generation_info: Dict[str, Any],
            stop_words: Optional[StopWords] = None,
            adapter_names: Optional[List[str]] = None,
            **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], int, Dict[str, Any]]:
        if stop_words is None:
            stop_words = []

        example = {
            'query': query,
            'history': history,
            'system': system,
            'images': images or [],  # for vl. str.
            'audios': kwargs.pop('audios', None) or [],
            'videos': kwargs.pop('videos', None) or [],
            'tools': kwargs.pop('tools', None),
            'objects': kwargs.pop('objects', None),
        }
        template.model = model
        inputs, tokenizer_kwargs = template.encode(example)

        truncation_strategy = kwargs.pop('truncation_strategy', 'delete')
        if len(inputs) == 0 and truncation_strategy == 'delete':
            # input_ids exceeds `max_length`. Please increase the value of `max_length`.
            return {}, tokenizer_kwargs, 0, example

        inputs.pop('labels', None)
        tokenizer = template.tokenizer
        device = next(model.parameters()).device
        if 'input_ids' in inputs:  # 1d
            input_ids = torch.tensor(inputs['input_ids'])[None]
            inputs['input_ids'] = input_ids
            token_len = input_ids.shape[1]
        if 'inputs_embeds' in inputs:  # 2d
            inputs_embeds = inputs['inputs_embeds'][None]
            inputs['inputs_embeds'] = inputs_embeds
            token_len = inputs_embeds.shape[1]

        inputs['attention_mask'] = torch.ones(token_len, dtype=torch.int64)[None]
        if 'token_type_ids' in inputs:
            inputs['token_type_ids'] = torch.tensor(inputs['token_type_ids'])[None]
        model.eval()
        if not generation_config.do_sample:
            generation_config.temperature = 1.
            generation_config.top_p = 1.
            generation_config.top_k = 50
        if tokenizer.eos_token_id is not None:
            generation_config.eos_token_id = tokenizer.eos_token_id
        if tokenizer.pad_token_id is not None:
            generation_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.bos_token_id is not None:
            generation_config.bos_token_id = tokenizer.bos_token_id
        if generation_config.max_new_tokens is not None:
            generation_config.max_length = 20  # fix max_length, max_new_tokens warning
            max_length = ConfigReader.get_max_model_len(model.config)
            if max_length and token_len + generation_config.max_new_tokens > max_length:
                generation_config.max_new_tokens = max_length - token_len
                if generation_config.max_new_tokens <= 0:
                    raise AssertionError(f'Current sentence length exceeds the model max_length: {max_length}')
        if template.suffix[-1] not in stop_words:
            stop_words.append(template.suffix[-1])
        inputs = to_device(inputs, device)
        if 'inputs_embeds' in inputs:
            inputs.pop('input_ids', None)
        if adapter_names is not None:
            inputs['adapter_names'] = adapter_names

        stopping_criteria = StoppingCriteriaList([StopWordsCriteria(tokenizer, stop_words, **tokenizer_kwargs)])
        inputs['stopping_criteria'] = stopping_criteria
        generation_info['num_prompt_tokens'] = token_len
        return inputs, tokenizer_kwargs, token_len, example

    @staticmethod
    def messages_join_observation(messages: Messages):
        """
            Joins observations from 'tool' message into the 'assistant' response.

            Example:
            ---------
            Original messages:
            messages = [
                {'role': 'user', 'content': "What's the weather today in Hangzhou?"},
                {'role': 'assistant', 'content': 'Action: get_weather\nAction Input:\
                      [{"location": "Hangzhou"}]\nObservations:'},
                {'role': 'tool', 'content': 'It is 26 degrees Celsius and sunny in Hangzhou today.'}
            ]

            Transformed messages:
            messages = [
                {'role': 'user', 'content': "What's the weather today in Hangzhou?"},
                {'role': 'assistant', 'content': 'Action: get_weather\nAction Input:\
                      [{"location": "Hangzhou"}]\nObservations: It is 26 degrees Celsius and sunny in Hangzhou today.'}
            ]
            """

        if len(messages) >= 2 and messages[-2]['role'] == 'assistant' and messages[-2]['content'] and messages[-2][
            'content'].endswith('Observation:'):
            assert messages[-1]['role'] == 'tool'
            observations = messages[-1]['content']
            messages.pop(-1)
            messages[-1]['content'] += observations
        return
