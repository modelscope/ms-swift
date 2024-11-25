# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch.distributed as dist
from datasets import Dataset as HfDataset

from swift.llm import (InferArguments, InferRequest, Messages, Processor, SwiftPipeline, Template, get_template,
                       load_dataset, sample_dataset)
from swift.tuners import Swift
from swift.utils import append_to_jsonl, get_logger, is_master
from .protocol import RequestConfig

logger = get_logger()


@dataclass
class InferCliState:
    # None: use default-system. '': not use system.
    system: Optional[str] = None
    messages: Messages = field(default_factory=list)  # not including system

    images: List[str] = field(default_factory=list)
    audios: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)

    multiline_mode: bool = False
    input_system: bool = False

    def clear(self):
        self.messages = []
        self.images = []
        self.audios = []
        self.videos = []

    def add_query(self, query: str) -> None:
        self.messages.append({'role': 'user', 'content': query})

    def add_response(self, response: str) -> None:
        self.messages.append({'role': 'assistant', 'content': response})

    def to_dict(self):
        infer_state = deepcopy(self)
        if infer_state.system is not None:
            infer_state.messages.insert(0, {'role': 'system', 'content': infer_state.system})
        return {
            'messages': infer_state.messages,
            'images': infer_state.images,
            'audios': infer_state.audios,
            'videos': infer_state.videos
        }


class SwiftInfer(SwiftPipeline):
    args_class = InferArguments
    args: args_class

    def __init__(self, args: Union[List[str], InferArguments, None] = None) -> None:
        from swift.llm import merge_lora
        super().__init__(args)
        args = self.args
        if args.merge_lora:
            merge_lora(args, device_map='cpu')
        self.infer_engine = self.get_infer_engine(args)
        if args.ckpt_dir and args.weight_type == 'lora':
            self.infer_engine.model = Swift.from_pretrained(self.infer_engine.model, args.ckpt_dir, inference_mode=True)
        self.template = self.get_template(args, self.processor)
        self.random_state = np.random.RandomState(args.data_seed)

    def __getattr__(self, key: str):
        try:
            return super().__getattr__(key)
        except AttributeError:
            if 'infer_engine' in self.__dict__:
                return getattr(self.infer_engine, key)
            raise

    @staticmethod
    def get_infer_engine(args: InferArguments, **kwargs):
        kwargs.update({
            'model_id_or_path': args.model,
            'model_type': args.model_type,
            'revision': args.model_revision,
            'torch_dtype': args.torch_dtype,
        })
        if args.infer_backend == 'pt':
            from .infer_engine import PtEngine
            infer_engine_cls = PtEngine
            kwargs.update(args.get_model_kwargs())
            kwargs.update({'max_batch_size': args.max_batch_size})
        elif args.infer_backend == 'vllm':
            from .infer_engine import VllmEngine
            infer_engine_cls = VllmEngine
            kwargs.update(args.get_vllm_engine_kwargs())
        else:
            from .infer_engine import LmdeployEngine
            infer_engine_cls = LmdeployEngine
            kwargs.update(args.get_lmdeploy_engine_kwargs())

        return infer_engine_cls(**kwargs)

    @staticmethod
    def get_template(args, processor: Processor) -> Template:
        template_kwargs = args.get_template_kwargs()
        template = get_template(args.template, processor, use_chat_template=args.use_chat_template, **template_kwargs)
        logger.info(f'default_system: {template.default_system}')
        return template

    def run(self) -> List[Dict[str, Any]]:
        args = self.args
        if args.eval_human:
            result = self.infer_cli()
        else:
            result = self.infer_dataset()
        if args.result_path is not None:
            logger.info(f'The inference results have been saved to result_path: `{args.result_path}`.')
        return result

    @staticmethod
    def _input_mm_data(infer_state: InferCliState) -> None:

        def _input_mm_file(mm_type: Literal['image', 'video', 'audio']) -> str:
            a_an = 'an' if mm_type[0] in {'i', 'a'} else 'a'
            return input(f'Input {a_an} {mm_type} path or URL <<< ')

        mm_types = ['image', 'video', 'audio']
        query = infer_state.messages[-1]['content']
        mm_tags = re.findall('|'.join(f'<({mm_type})>' for mm_type in mm_types), query)
        # mm_tag -> mm_type/mm_key
        mm_mapping = {f'<{mm_type}>': (mm_type, f'{mm_type}s') for mm_type in mm_types}
        for mm_tag in mm_tags:
            mm_type, mm_key = mm_mapping[mm_tag]
            mm_val = getattr(infer_state, mm_key)
            mm_val.append(_input_mm_file(mm_type))

    @staticmethod
    def _input_multiline(prompt: str) -> str:
        query = ''
        stop_words = '#\n'
        while True:
            text = f'{input(prompt)}\n'
            prompt = ''
            if text.endswith(stop_words):
                query += text[:len(stop_words)]
                break
            query += text
        return query

    @staticmethod
    def _input_text(multiline_mode: bool, input_system: bool) -> str:
        if multiline_mode:
            addi_prompt = '[MS]' if input_system else '[M]'
            text = SwiftInfer._input_multiline(f'<<<[{addi_prompt}] ')
        else:
            addi_prompt = '[S]' if input_system else ''
            text = input(f'<<<{addi_prompt} ')
        return text

    @staticmethod
    def _check_query(infer_state: InferCliState, query: str) -> Optional[str]:
        query_std = query.strip().lower()
        if infer_state.input_system:
            if query == 'default-system':
                infer_state.system = None
            else:
                infer_state.system = query
            infer_state.input_system = False
            query_std = 'clear'
        if query_std == 'clear':
            infer_state.clear()
            return
        if query_std == '':
            return
        if query_std == 'reset-system':
            infer_state.input_system = True
            return
        if query_std == 'multi-line':
            infer_state.multiline_mode = True
            logger.info('End multi-line input with `#`.')
            logger.info('Input `single-line` to switch to single-line input mode.')
            return
        if query_std == 'single-line':
            infer_state.multiline_mode = False
            return
        return query

    def infer_single(self, infer_request: InferRequest, request_config: RequestConfig) -> str:
        res_or_gen = self.infer([infer_request], request_config, template=self.template, use_tqdm=False)
        if request_config.stream:
            response = ''
            for res in res_or_gen:
                delta = res[0].choices[0].delta.content
                print(delta, end='', flush=True)
                response += delta
            print()
        else:
            response = res_or_gen[0].choices[0].message.content
            print(response)
        return response

    def infer_cli(self) -> List[Dict[str, Any]]:
        args = self.args
        template = self.template
        request_config = args.get_request_config()

        logger.info('Input `exit` or `quit` to exit the conversation.')
        logger.info('Input `multi-line` to switch to multi-line input mode.')
        logger.info('Input `reset-system` to reset the system and clear the history.')
        support_multi_round = template.template_meta.support_multi_round
        if support_multi_round:
            logger.info('Input `clear` to clear the history.')
        else:
            logger.info('The current template only supports single-round dialogues.')

        infer_state = InferCliState()
        result_list = []
        while True:
            if not support_multi_round:
                infer_state.clear()
            query = self._input_text(infer_state.multiline_mode, infer_state.input_system)
            if query.strip().lower() in {'exit', 'quit'}:
                break
            query = self._check_query(infer_state, query)
            if query is None:
                continue
            infer_state.add_query(query)
            self._input_mm_data(infer_state)
            data = infer_state.to_dict()
            response = self.infer_single(InferRequest(**data), request_config)
            infer_state.add_response(response)
            data['response'] = response
            result_list.append(data)
            if args.result_path is not None:
                append_to_jsonl(args.result_path, data, strict=False)

        return result_list

    def _prepare_val_dataset(self) -> HfDataset:
        args = self.args
        dataset_kwargs = args.get_dataset_kwargs()
        if len(args.val_dataset) > 0:
            _, val_dataset = load_dataset(args.val_dataset, split_dataset_ratio=1.0, **dataset_kwargs)
        else:
            _, val_dataset = load_dataset(args.dataset, split_dataset_ratio=args.split_dataset_ratio, **dataset_kwargs)
        assert val_dataset is not None
        val_dataset = sample_dataset(val_dataset, args.val_dataset_sample, self.random_state)
        return val_dataset

    def infer_dataset(self) -> List[Dict[str, Any]]:
        args = self.args
        request_config = args.get_request_config(args.stream)
        logger.info(f'request_config: {request_config}')

        val_dataset = self._prepare_val_dataset()
        logger.info(f'val_dataset: {val_dataset}')
        result_list = []
        if request_config.stream:
            for data in val_dataset:
                data['response'] = self.infer_single(InferRequest(**data), request_config)
                result_list.append(data)
                if args.result_path is not None:
                    append_to_jsonl(args.result_path, data)
        else:
            if dist.is_initialized():
                val_dataset = val_dataset.shard(args.global_world_size, args.rank, contiguous=True)
            infer_requests = [InferRequest(**data) for i, data in enumerate(val_dataset)]

            resp_list = self.infer(infer_requests, request_config, template=self.template, use_tqdm=True)
            for data, resp in zip(val_dataset, resp_list):
                data['response'] = resp.choices[0].message.content
                result_list.append(data)
            if dist.is_initialized():
                total_result_list = [None for _ in range(args.global_world_size)] if args.rank == 0 else None
                dist.gather_object(result_list, total_result_list)
                result_list = total_result_list and list(chain.from_iterable(total_result_list))

            if is_master() and args.result_path and result_list:
                append_to_jsonl(args.result_path, result_list)
        return result_list


def infer_main(args: Union[List[str], InferArguments, None] = None):
    return SwiftInfer(args).main()
