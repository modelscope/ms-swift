# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Optional, Set, Union

import json
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset

from swift.utils import get_logger

logger = get_logger()
DATASET_TYPE = Union[HfDataset, HfIterableDataset]


def _check_path(value: Union[str, List[str]],
                k: Optional[str] = None,
                check_exist_path_set: Optional[Set[str]] = None) -> Union[str, List[str]]:
    if check_exist_path_set is None:
        check_exist_path_set = set()
    if isinstance(value, str):
        value = os.path.expanduser(value)
        value = os.path.abspath(value)
        if k in check_exist_path_set and not os.path.exists(value):
            if k is not None:
                raise FileNotFoundError(f"`{k}`: '{value}'")
            else:
                raise FileNotFoundError(f"path: '{value}'")
    elif isinstance(value, list):
        res = []
        for v in value:
            res.append(cls._check_path(v, k, check_exist_path_set))
        value = res
    return value


def handle_path(args: Union['SftArguments', 'InferArguments']) -> None:
    """Check all paths in the args correct and exist"""
    check_exist_path = ['ckpt_dir', 'resume_from_checkpoint', 'custom_register_path']
    maybe_check_exist_path = ['model_id_or_path', 'custom_dataset_info']
    from swift.llm.argument import SftArguments
    if isinstance(args, SftArguments):
        check_exist_path.append('deepspeed_config_path')
        maybe_check_exist_path.append('deepspeed')

    for k in maybe_check_exist_path:
        v = getattr(args, k)
        if isinstance(v, str) and v is not None and (v.startswith('~') or v.startswith('/') or os.path.exists(v)):
            check_exist_path.append(k)
    check_exist_path_set = set(check_exist_path)
    other_path = ['output_dir', 'logging_dir']
    for k in check_exist_path + other_path:
        value = getattr(args, k, None)
        if value is None:
            continue
        value = _check_path(value, k, check_exist_path_set)
        setattr(args, k, value)


def load_from_ckpt_dir(args: Union['SftArguments', 'InferArguments']) -> None:
    """Load specific attributes from sft_args.json"""
    from swift.llm.argument import SftArguments, ExportArguments
    if isinstance(args, SftArguments):
        ckpt_dir = args.resume_from_checkpoint
    else:
        ckpt_dir = args.ckpt_dir
    sft_args_path = os.path.join(ckpt_dir, 'sft_args.json')
    export_args_path = os.path.join(ckpt_dir, 'export_args.json')
    from_sft_args = os.path.exists(sft_args_path)
    if not os.path.exists(sft_args_path) and not os.path.exists(export_args_path):
        logger.warning(f'{sft_args_path} not found')
        return
    args_path = sft_args_path if from_sft_args else export_args_path
    with open(args_path, 'r', encoding='utf-8') as f:
        old_args = json.load(f)

    imported_keys = [
        'model_type', 'model_revision', 'template_type', 'dtype', 'quant_method', 'quantization_bit',
        'bnb_4bit_comp_dtype', 'bnb_4bit_quant_type', 'bnb_4bit_use_double_quant', 'model_id_or_path',
        'custom_register_path', 'custom_dataset_info'
    ]
    if (isinstance(args, SftArguments) and args.train_backend == 'megatron'
            or isinstance(args, ExportArguments) and args.to_hf is True):
        imported_keys += ['tp', 'pp']
    if not isinstance(args, SftArguments):
        imported_keys += ['sft_type', 'rope_scaling', 'system']
        if getattr(args, 'load_dataset_config', False) and from_sft_args:
            imported_keys += [
                'dataset', 'val_dataset', 'dataset_seed', 'dataset_test_ratio', 'check_dataset_strategy',
                'self_cognition_sample', 'model_name', 'model_author', 'train_dataset_sample', 'val_dataset_sample'
            ]
    for key in imported_keys:
        if not hasattr(args, key):
            continue
        value = getattr(args, key)
        old_value = old_args.get(key)
        if old_value is None:
            continue
        if key in {'dataset', 'val_dataset'} and len(value) > 0:
            continue
        if key in {
                'system', 'quant_method', 'model_id_or_path', 'custom_register_path', 'custom_dataset_info',
                'dataset_seed'
        } and value is not None:
            continue
        if key in {'template_type', 'dtype'} and value != 'AUTO':
            continue
        setattr(args, key, old_value)
