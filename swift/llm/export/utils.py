# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import json
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from swift.llm import ExportArguments, PtEngine, SwiftInfer, Template


def prepare_pt_engine_template(args: ExportArguments, load_model: bool = True) -> Tuple[PtEngine, Template]:
    args.infer_backend = 'pt'
    pt_engine: PtEngine = SwiftInfer.get_infer_engine(args, load_model=load_model)
    delattr(args, 'infer_backend')
    template = SwiftInfer.get_template(args, pt_engine.tokenizer)
    return pt_engine, template


def save_checkpoint(model: Optional[PreTrainedModel],
                    tokenizer: PreTrainedTokenizerBase,
                    output_dir: str,
                    *,
                    safe_serialization: bool = True,
                    max_shard_size: Union[int, str] = '5GB',
                    additional_saved_files: Optional[List[str]] = None) -> None:
    if model is not None:
        model.save_pretrained(output_dir, safe_serialization=safe_serialization, max_shard_size=max_shard_size)
    if hasattr(tokenizer, 'processor'):
        tokenizer.processor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    for src_file in additional_saved_files:
        src_path: str = os.path.join(model.model_dir, src_file)
        tgt_path = os.path.join(output_dir, src_file)
        if os.path.isfile(src_path):
            shutil.copy(src_path, tgt_path)
            break
        elif os.path.isdir(src_path):
            shutil.copytree(src_path, tgt_path)
            break
