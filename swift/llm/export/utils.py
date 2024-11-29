# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import json
from transformers import PreTrainedModel

from swift.llm import ExportArguments, Processor, PtEngine, SwiftInfer, Template
from swift.plugin import extra_tuners
from swift.tuners import Swift
from ..model.register import load_by_unsloth


def prepare_pt_engine(args: ExportArguments, pt_engine):
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


def prepare_pt_engine_template(args: ExportArguments, load_model: bool = True, **kwargs) -> Tuple[PtEngine, Template]:
    kwargs = {}
    if args.tuner_backend == 'unsloth' and args.weight_type == 'adapter':
        kwargs = {'load_model': False}

    pt_engine: PtEngine = SwiftInfer.get_infer_engine(args, infer_backend='pt', load_model=load_model, **kwargs)
    if args.ckpt_dir and args.weight_type == 'adapter':
        prepare_pt_engine(args, pt_engine)

    template = SwiftInfer.get_template(args, pt_engine.processor)
    return pt_engine, template


def save_checkpoint(model: Optional[PreTrainedModel],
                    processor: Processor,
                    output_dir: str,
                    *,
                    safe_serialization: bool = True,
                    max_shard_size: Union[int, str] = '5GB',
                    ckpt_dir: str = None,
                    additional_saved_files: Optional[List[str]] = None) -> None:
    if model is not None:
        model.save_pretrained(output_dir, safe_serialization=safe_serialization, max_shard_size=max_shard_size)
    processor.save_pretrained(output_dir)

    for src_file in additional_saved_files or [] + ['preprocessor_config.json', 'args.json']:
        for model_dir in [model and model.model_dir, ckpt_dir]:
            if model_dir is None:
                continue
            src_path: str = os.path.join(model_dir, src_file)
            tgt_path = os.path.join(output_dir, src_file)
            if os.path.isfile(src_path):
                shutil.copy(src_path, tgt_path)
                break
            elif os.path.isdir(src_path):
                shutil.copytree(src_path, tgt_path)
                break
