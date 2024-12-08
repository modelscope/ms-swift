# Copyright (c) Alibaba, Inc. and its affiliates.

import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

from swift.llm import InferArguments, Template
from swift.plugin import extra_tuners
from swift.tuners import Swift
from swift.utils import get_logger
from ..model.register import load_by_unsloth
from ..template import Messages
from .infer_engine import PtEngine

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
        role = 'user'
        if query.startswith('tool:'):
            role = 'tool'
            query = query[len('tool:'):]
        self.messages.append({'role': role, 'content': query})

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

    def input_mm_data(self) -> None:

        def _input_mm_file(mm_type: Literal['image', 'video', 'audio']) -> str:
            a_an = 'an' if mm_type[0] in {'i', 'a'} else 'a'
            return input(f'Input {a_an} {mm_type} path or URL <<< ')

        mm_types = ['image', 'video', 'audio']
        query = self.messages[-1]['content']
        mm_tags = re.findall('|'.join(f'<{mm_type}>' for mm_type in mm_types), query)
        # mm_tag -> mm_type/mm_key
        mm_mapping = {f'<{mm_type}>': (mm_type, f'{mm_type}s') for mm_type in mm_types}
        for mm_tag in mm_tags:
            mm_type, mm_key = mm_mapping[mm_tag]
            mm_val = getattr(self, mm_key)
            mm_val.append(_input_mm_file(mm_type))

    @staticmethod
    def _input_multiline(prompt: str) -> str:
        query = ''
        stop_words = '#\n'
        while True:
            text = f'{input(prompt)}\n'
            prompt = ''
            if text.endswith(stop_words):
                query += text[:-len(stop_words)]
                break
            query += text
        return query

    def input_text(self) -> str:
        if self.multiline_mode:
            addi_prompt = '[MS]' if self.input_system else '[M]'
            text = InferCliState._input_multiline(f'<<<{addi_prompt} ')
        else:
            addi_prompt = '[S]' if self.input_system else ''
            text = input(f'<<<{addi_prompt} ')
        return text

    def check_query(self, query: str) -> Optional[str]:
        query_std = query.strip().lower()
        if self.input_system:
            if query == 'default-system':
                self.system = None
            else:
                self.system = query
            self.input_system = False
            query_std = 'clear'
        if query_std == 'clear':
            self.clear()
            return
        if query_std == '':
            return
        if query_std == 'reset-system':
            self.input_system = True
            return
        if query_std == 'multi-line':
            self.multiline_mode = True
            logger.info('End multi-line input with `#`.')
            logger.info('Input `single-line` to switch to single-line input mode.')
            return
        if query_std == 'single-line':
            self.multiline_mode = False
            return
        return query


def _prepare_pt_engine(args: InferArguments, pt_engine):
    if args.train_type in extra_tuners:
        extra_tuners[args.train_type].from_pretrained(pt_engine.model, args.ckpt_dir, inference_mode=True)
    else:
        if args.tuner_backend == 'unsloth':
            model, processor = load_by_unsloth(args.ckpt_dir, args.torch_dtype, args.max_length, args.quant_bits == 4,
                                               args.model_meta.is_multimodal)
            model_info = pt_engine.processor.model_info
            model_meta = pt_engine.processor.model_meta
            processor.model_info = model_info
            processor.model_meta = model_meta
            model.model_info = model_info
            model.model_meta = model_meta

            if args.model_meta.is_multimodal:
                from unsloth import FastVisionModel as UnslothModel
            else:
                from unsloth import FastLanguageModel as UnslothModel
            UnslothModel.for_inference(model)

            pt_engine.model = model
            pt_engine.generation_config = model.generation_config
            pt_engine.processor = processor
        else:
            pt_engine.model = Swift.from_pretrained(pt_engine.model, args.ckpt_dir, inference_mode=True)
            if args.train_type == 'bone':
                # Bone has a problem of float32 matmul with bloat16 in `peft==0.14.0`
                pt_engine.model.to(pt_engine.model.dtype)


def prepare_pt_engine_template(args: InferArguments, load_model: bool = True, **kwargs) -> Tuple[PtEngine, Template]:
    from .infer import SwiftInfer
    if args.tuner_backend == 'unsloth' and args.weight_type == 'adapter':
        kwargs['load_model'] = False

    pt_engine: PtEngine = SwiftInfer.get_infer_engine(args, infer_backend='pt', load_model=load_model, **kwargs)
    if args.ckpt_dir and args.weight_type == 'adapter':
        _prepare_pt_engine(args, pt_engine)

    template = SwiftInfer.get_template(args, pt_engine.processor)
    return pt_engine, template
