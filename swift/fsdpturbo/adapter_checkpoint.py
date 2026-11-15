# Copyright (c) ModelScope Contributors. All rights reserved.
"""Portable optimizer and RNG state for replicated native PEFT adapters."""
import json
import numpy as np
import os
import random
import torch
from pathlib import Path


def _cpu_tensors(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tensors(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_tensors(item) for item in value)
    return value


def write_tensor_state(path, state) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(_cpu_tensors(state), temporary)
    os.replace(temporary, path)


def capture_rng(device) -> dict:
    numpy_state = np.random.get_state()
    result = {
        'python': random.getstate(),
        'numpy': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        'torch': torch.get_rng_state(),
    }
    if torch.device(device).type != 'cpu':
        result['accelerator'] = torch.get_device_module(device).get_rng_state(device)
    return result


def restore_rng(state, device) -> None:
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state['torch'])
    if 'accelerator' in state:
        torch.get_device_module(device).set_rng_state(state['accelerator'], device)


def adapter_run_signature(args) -> dict:
    keys = ('max_length', 'gradient_accumulation_steps', 'data_seed', 'per_device_train_batch_size',
            'train_dataloader_shuffle', 'dataloader_drop_last')
    return {key: getattr(args, key) for key in keys}


def read_adapter_metadata(path, *, topology, dataset_fingerprint=None, run_signature=None) -> dict:
    filename = Path(path) / 'adapter_training_state.json'
    if not filename.is_file():
        raise ValueError('Adapter checkpoint has no completed optimizer state; use --adapters for a warm start.')
    with filename.open(encoding='utf-8') as source:
        metadata = json.load(source)
    if metadata.get('format_version') != 1 or metadata.get('topology') != topology:
        raise ValueError('Adapter resume requires the same world/FSDP/CP/EP topology and checkpoint format.')
    if dataset_fingerprint is not None and metadata.get('dataset_fingerprint') != dataset_fingerprint:
        raise ValueError('Adapter resume requires the same training dataset and sequence configuration.')
    if run_signature is not None:
        saved_signature = metadata.get('run_signature')
        if saved_signature is None:
            # Early format-one exports already saved these exact Swift args.
            # Read them without modifying the checkpoint or inferring defaults.
            with (Path(path) / 'args.json').open(encoding='utf-8') as source:
                saved_args = json.load(source)
            saved_signature = {key: saved_args.get(key) for key in run_signature}
        if saved_signature != run_signature:
            raise ValueError(
                'Adapter resume requires the same sequence length, accumulation, batch and sampler settings.')
    return metadata


def write_adapter_metadata(path, metadata) -> None:
    filename = Path(path) / 'adapter_training_state.json'
    temporary = filename.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8') as target:
        json.dump(dict(format_version=1, **metadata), target, indent=2)
    os.replace(temporary, filename)
