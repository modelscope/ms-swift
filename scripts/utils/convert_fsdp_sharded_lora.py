# Copyright (c) ModelScope Contributors. All rights reserved.
"""Convert an FSDP `SHARDED_STATE_DICT` LoRA checkpoint into a standard PEFT adapter directory.

When training LoRA with `--fsdp` and `state_dict_type: SHARDED_STATE_DICT`, `Trainer.save_model` is a
no-op, so `checkpoint-xxx/` only holds a DCP-sharded `pytorch_model_fsdp_0/` directory (written by
accelerate's `save_fsdp_model`) instead of `adapter_config.json` + `adapter_model.safetensors`.
This script gathers the shards on CPU (no GPU, no `torchrun`) and rebuilds the adapter directory,
so that `swift infer --adapters <output_dir>` / `swift export --merge_lora true` work as usual.

Usage:
    python scripts/utils/convert_fsdp_sharded_lora.py output/vx-xxx/checkpoint-100
    python scripts/utils/convert_fsdp_sharded_lora.py output/vx-xxx/checkpoint-100 -o /path/to/adapter
"""
import argparse
import json
import os
import re
import shutil
import tempfile
import torch
from safetensors.torch import save_file

FSDP_MODEL_NAME = 'pytorch_model_fsdp'
PEFT_PREFIX = 'base_model.model.'
# `lora_magnitude_vector` only shows up with `use_dora true`.
LORA_MODULE_PATTERN = re.compile(r'^(?P<module>.+?)\.(lora_A|lora_B|lora_embedding_A|lora_embedding_B'
                                 r'|lora_magnitude_vector)(\.|$)')
MODULES_TO_SAVE_PATTERN = re.compile(r'\.(weight|bias)$')


def find_dcp_dir(checkpoint: str, model_index: int = 0) -> str:
    """Locate the DCP directory holding the sharded weights."""
    if os.path.isfile(os.path.join(checkpoint, '.metadata')):
        return checkpoint  # already pointing at `pytorch_model_fsdp_x`
    dcp_dir = os.path.join(checkpoint, f'{FSDP_MODEL_NAME}_{model_index}')
    if os.path.isfile(os.path.join(dcp_dir, '.metadata')):
        return dcp_dir
    if os.path.exists(os.path.join(checkpoint, f'{FSDP_MODEL_NAME}.bin')):
        raise FileNotFoundError(f'{checkpoint} was saved with FULL_STATE_DICT, not SHARDED_STATE_DICT. '
                                'The adapter should already be there; no conversion is needed.')
    raise FileNotFoundError(f'Cannot find `{FSDP_MODEL_NAME}_{model_index}/.metadata` under {checkpoint}. '
                            f'Existing entries: {sorted(os.listdir(checkpoint))}')


def load_sharded_state_dict(dcp_dir: str):
    """Gather a DCP checkpoint into a plain state dict in a single CPU process."""
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, 'gathered.pt')
        dcp_to_torch_save(dcp_dir, tmp_path)
        state_dict = torch.load(tmp_path, map_location='cpu', weights_only=False)
    # `save_fsdp_model` wraps the state dict as `{'model': state_dict}`.
    return state_dict.get('model', state_dict)


def read_train_args(checkpoint: str) -> dict:
    """`args.json` is written to the run dir, and copied into each checkpoint by `SwiftMixin._save`."""
    for args_path in [os.path.join(checkpoint, 'args.json'), os.path.join(os.path.dirname(checkpoint), 'args.json')]:
        if os.path.isfile(args_path):
            with open(args_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    raise FileNotFoundError(f'Cannot find `args.json` in {checkpoint} or its parent directory. '
                            'It is required to recover lora_alpha/lora_dropout/etc.')


def parse_lora_layout(state_dict):
    """Recover `target_modules` / `rank_pattern` / `modules_to_save` from the weight keys.

    Deriving them from the checkpoint is exact, unlike replaying `get_target_modules`, which needs the
    instantiated model to expand `all-linear`.
    """
    ranks = {}
    plain_weights = []
    for key, value in state_dict.items():
        name = key[len(PEFT_PREFIX):] if key.startswith(PEFT_PREFIX) else key
        match = LORA_MODULE_PATTERN.match(name)
        if match is None:
            plain_weights.append(name)
            continue
        module = match.group('module')
        if key.endswith('lora_A.weight') or key.endswith('lora_embedding_A'):
            ranks[module] = value.shape[0]
        else:
            ranks.setdefault(module, None)
    if not ranks:
        raise ValueError('No LoRA weights found in the checkpoint. Was this a full-parameter run?')

    unresolved = sorted(m for m, r in ranks.items() if r is None)
    if unresolved:
        raise ValueError(f'Cannot infer the rank of: {unresolved}')
    counts = {}
    for rank in ranks.values():
        counts[rank] = counts.get(rank, 0) + 1
    # The majority rank goes to `r`; the rest are listed in `rank_pattern` (e.g. set via `--lora_rank_pattern`).
    main_rank = max(counts, key=counts.get)
    rank_pattern = {module: rank for module, rank in ranks.items() if rank != main_rank}

    # `get_peft_model_state_dict` drops the `modules_to_save.{adapter_name}.` infix, so those weights look like
    # ordinary parameters; `bias='all'` additionally stores the base layer bias of LoRA targets.
    modules_to_save = set()
    for name in plain_weights:
        module = MODULES_TO_SAVE_PATTERN.sub('', name)
        if module not in ranks:
            modules_to_save.add(module)
    return sorted(ranks), main_rank, rank_pattern, sorted(modules_to_save)


def get_init_weights(train_args: dict):
    """`TunerArguments.__post_init__` already casts 'true'/'false' to bool; be tolerant of older args.json."""
    init_weights = train_args.get('init_weights', True)
    if isinstance(init_weights, str) and init_weights.lower() in {'true', 'false'}:
        return init_weights.lower() == 'true'
    return init_weights


def get_task_type(train_args: dict):
    """Mirror the task_type mapping in `swift/pipelines/train/tuner.py::prepare_adapter`."""
    task_type = (train_args.get('task_type') or 'causal_lm').upper()
    return {'EMBEDDING': None, 'RERANKER': 'SEQ_CLS', 'GENERATIVE_RERANKER': 'CAUSAL_LM'}.get(task_type, task_type)


def build_adapter_config(train_args: dict, target_modules, rank, rank_pattern, modules_to_save, base_model: str):
    from peft import LoraConfig
    kwargs = {}
    if train_args.get('target_parameters') is not None:
        kwargs['target_parameters'] = train_args['target_parameters']
    return LoraConfig(
        task_type=get_task_type(train_args),
        r=rank,
        rank_pattern=rank_pattern,
        target_modules=target_modules,
        lora_alpha=train_args.get('lora_alpha', 32.0),
        lora_dropout=train_args.get('lora_dropout', 0.05),
        bias=train_args.get('lora_bias', 'none'),
        modules_to_save=modules_to_save or None,
        use_rslora=train_args.get('use_rslora', False),
        use_dora=train_args.get('use_dora', False),
        init_lora_weights=get_init_weights(train_args),
        base_model_name_or_path=base_model,
        **kwargs,
    )


def convert(checkpoint: str,
            output_dir=None,
            model_index: int = 0,
            base_model=None,
            safe_serialization: bool = True) -> str:
    checkpoint = os.path.abspath(os.path.expanduser(checkpoint))
    output_dir = output_dir or f'{checkpoint}-adapter'
    dcp_dir = find_dcp_dir(checkpoint, model_index)
    train_args = read_train_args(checkpoint)

    tuner_type = train_args.get('tuner_type')
    if tuner_type not in {'lora', 'longlora'}:
        raise ValueError(f'tuner_type="{tuner_type}" is not supported; this script only handles LoRA adapters.')
    if train_args.get('use_swift_lora') or train_args.get('tuner_backend', 'peft') != 'peft':
        raise ValueError('Only `--tuner_backend peft` without `--use_swift_lora` produces a PEFT-format adapter.')

    print(f'Gathering shards from {dcp_dir} ...')
    state_dict = load_sharded_state_dict(dcp_dir)
    target_modules, rank, rank_pattern, modules_to_save = parse_lora_layout(state_dict)
    print(f'Found {len(target_modules)} target modules, r={rank}, '
          f'rank_pattern={rank_pattern or "{}"}, modules_to_save={modules_to_save or "[]"}')

    config = build_adapter_config(train_args, target_modules, rank, rank_pattern, modules_to_save, base_model
                                  or train_args.get('model'))
    os.makedirs(output_dir, exist_ok=True)
    config.inference_mode = True  # `PeftModel.save_pretrained` also flips this before dumping the config
    config.save_pretrained(output_dir)
    # `swift.tuners.peft.LoraConfig.save_pretrained` also writes these swift-only fields.
    additional_config = {
        'lora_dtype': train_args.get('lora_dtype'),
        'lorap_lr_ratio': train_args.get('lorap_lr_ratio'),
        'lorap_emb_lr': train_args.get('lorap_emb_lr', 1e-6),
    }
    with open(os.path.join(output_dir, 'additional_config.json'), 'w', encoding='utf-8') as f:
        json.dump(additional_config, f)

    # The gathered keys are exactly what `PeftModel.save_pretrained` writes, so dump them verbatim.
    state_dict = {key: value.contiguous() for key, value in state_dict.items()}
    if safe_serialization:
        save_file(state_dict, os.path.join(output_dir, 'adapter_model.safetensors'), metadata={'format': 'pt'})
    else:
        torch.save(state_dict, os.path.join(output_dir, 'adapter_model.bin'))

    args_path = os.path.join(checkpoint, 'args.json')
    if not os.path.exists(args_path):
        args_path = os.path.join(os.path.dirname(checkpoint), 'args.json')
    shutil.copy(args_path, os.path.join(output_dir, 'args.json'))
    print(f'Adapter saved to {output_dir}')
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', help='e.g. output/vx-xxx/checkpoint-100')
    parser.add_argument('-o', '--output_dir', default=None, help='default: `<checkpoint>-adapter`')
    parser.add_argument('--model_index', type=int, default=0, help='index of `pytorch_model_fsdp_{index}`, default 0')
    parser.add_argument('--base_model', default=None, help='override `base_model_name_or_path`')
    parser.add_argument('--safe_serialization', type=lambda x: x.lower() != 'false', default=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert(args.checkpoint, args.output_dir, args.model_index, args.base_model, args.safe_serialization)
