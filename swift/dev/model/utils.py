# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import glob
import os
import shutil
from typing import Any, List, Optional

__all__ = ['save_checkpoint']


def _copy_files_by_pattern(source_dir: str, dest_dir: str, pattern: str) -> None:
    for src_path in glob.glob(os.path.join(source_dir, pattern)):
        shutil.copy(src_path, os.path.join(dest_dir, os.path.basename(src_path)))


def save_checkpoint(model: Optional[Any],
                    processor: Any,
                    output_dir: str,
                    *,
                    safe_serialization: bool = True,
                    max_shard_size: Any = '5GB',
                    model_dirs: List[str] = None,
                    additional_saved_files: Optional[List[str]] = None) -> None:
    """Write a model directory out: weights via ``save_pretrained``, then processor + sidecar files.

    ``model=None`` writes only the processor and copies sidecars -- the streaming transforms already
    wrote their weights and reuse this for the non-weight half. Each sidecar file (args.json,
    preprocessor_config.json, extras) is searched first-come across ``model_dirs``; an existing
    args.json in the destination is never overwritten.
    """
    if model is not None:
        if model.__class__.__name__ != 'SentenceTransformer':
            model.save_pretrained(output_dir, safe_serialization=safe_serialization, max_shard_size=max_shard_size)
        else:
            model.save_pretrained(output_dir, safe_serialization=safe_serialization)
            # SentenceTransformer keeps modules/configs as loose .py/.json files; carry them over.
            _copy_files_by_pattern(model.model_dir, output_dir, '*.py')
            _copy_files_by_pattern(model.model_dir, output_dir, '*.json')
    processor.save_pretrained(output_dir)

    if model_dirs is None:
        model_dirs = []
    else:
        model_dirs = model_dirs.copy()
    if model and model.model_dir and model.model_dir not in model_dirs:
        model_dirs.append(model.model_dir)
    for src_file in (additional_saved_files or []) + ['preprocessor_config.json', 'args.json']:
        tgt_path = os.path.join(output_dir, src_file)
        if os.path.exists(tgt_path) and src_file == 'args.json':
            continue
        for model_dir in model_dirs:
            src_path: str = os.path.join(model_dir, src_file)
            if os.path.isfile(src_path):
                shutil.copy(src_path, tgt_path)
                break
            elif os.path.isdir(src_path):
                shutil.copytree(src_path, tgt_path)
                break
