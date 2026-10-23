# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import json
import os
import shutil
from typing import TYPE_CHECKING, Any, Dict, Optional

from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.config import CheckpointConfig, ModelConfig, TemplateConfig, TunerConfig

logger = get_logger()

#: One adapter directory's on-disk names (peft's layout; a swift-format checkpoint uses the same
#: file names inside its per-adapter subdirectories).
_ADAPTER_CONFIG_NAME = 'adapter_config.json'
_ADAPTER_WEIGHTS_NAME = 'adapter_model.safetensors'
#: Legacy swift checkpoints park per-adapter side state (activations of extra_state_keys) here;
#: peft's layout has no counterpart, so a checkpoint carrying it cannot be converted.
_EXTRA_STATE_DIR = 'extra_states'
#: The adapter-name segment peeled off each LoRA parameter key when rewriting into peft's form.
_KEY_REWRITES = ('lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B')


def run_merge_lora(
    model_config: ModelConfig,
    tuner_config: TunerConfig,
    *,
    template_config: Optional[TemplateConfig] = None,
    checkpoint_config: Optional[CheckpointConfig] = None,
    output_dir: Optional[str] = None,
    device_map: Optional[Any] = None,
    replace_if_exists: bool = False,
    parallel_size: int = 1,
) -> str:
    """Merge ``tuner_config.adapters`` into the base model and save it; returns the output directory.

    Defaults to ``'{first adapter}-merged'`` so a merge next to the checkpoint needs no argument;
    an existing directory is left untouched unless ``replace_if_exists`` (merging is idempotent,
    and accidentally re-writing multi-GB weights is expensive).
    """
    from swift.dev.components import MergeComponent

    component = MergeComponent(
        model_config,
        tuner_config,
        template_config=template_config,
        checkpoint_config=checkpoint_config,
        output_dir=output_dir,
        device_map=device_map,
        replace_if_exists=replace_if_exists,
        parallel_size=parallel_size)
    return component().path


def run_to_peft_format(adapter: str, output_dir: str) -> str:
    """Convert a Swift-format adapter to PEFT's directory layout; returns the converted directory.

    A directory without a ``default/`` subdirectory is already in peft layout and is returned
    unchanged; otherwise every per-adapter subdirectory is rewritten in place into the copy at
    ``output_dir`` and the ``default`` adapter is hoisted to the top level.
    """
    os.makedirs(output_dir, exist_ok=True)
    if 'default' not in os.listdir(adapter):
        logger.info('The format of the checkpoint is already in peft format.')
        return adapter
    _save_to_peft_format(adapter, output_dir)
    logger.info(f'Converting the swift format checkpoint to peft format, and saving it to: `{output_dir}`')
    return output_dir


def _reject_unconvertible(adapter_config: Dict[str, Any]) -> None:
    """Refuse adapter configs the peft layout cannot represent, naming the reason."""
    tuner = adapter_config.get('peft_type')
    if tuner != 'LORA':
        raise ValueError(f'Only LoRA can be converted to peft format, got {tuner!r}.')
    if adapter_config.get('use_qa_lora') or adapter_config.get('use_merged_linear'):
        raise ValueError('Cannot convert to peft format: use_qa_lora / use_merged_linear produce parameter '
                         'shapes the peft layout has no counterpart for.')


def _peft_state_dict(state_dict: Dict[str, Any], adapter: str) -> Dict[str, Any]:
    """Rewrite one adapter's swift-format keys into peft's naming.

    Two mechanical rules: keys gain the ``base_model.model.`` prefix peft expects, and the
    adapter-name segment (``lora_A.<adapter>.``) is peeled off every tuner parameter. DoRA's
    ``lora_magnitude_vector`` carries the adapter name without a trailing dot.
    """
    rewritten: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if not key.startswith('base_model.model.'):
            key = 'base_model.model.' + key
        for part in _KEY_REWRITES:
            key = key.replace(f'{part}.{adapter}.', f'{part}.')
        key = key.replace(f'lora_magnitude_vector.{adapter}', 'lora_magnitude_vector')
        rewritten[key] = value
    return rewritten


def _write_peft_adapter_config(adapter_config: Dict[str, Any], adapter_dir: str) -> None:
    """Write ``adapter_config.json`` in plain peft form, dropping swift's own keys.

    Swift-only knobs (LoRA+ lr ratios, lora dtype, ...) have no peft field; the ones legacy's own
    converter preserved for its reload path ride the sibling ``additional_config.json``, exactly
    as legacy's extended ``LoraConfig.save_pretrained`` wrote them.
    """
    import dataclasses

    from peft import LoraConfig

    field_names = {field.name for field in dataclasses.fields(LoraConfig)}
    filtered = {key: value for key, value in adapter_config.items()
                if key in field_names and key != 'peft_type'}
    filtered['peft_type'] = 'LORA'
    LoraConfig(**filtered).save_pretrained(adapter_dir)
    additional = {
        'lora_dtype': adapter_config.get('lora_dtype'),
        'lorap_lr_ratio': adapter_config.get('lorap_lr_ratio', 16.0),
        'lorap_emb_lr': adapter_config.get('lorap_emb_lr', 1e-6),
    }
    with open(os.path.join(adapter_dir, 'additional_config.json'), 'w', encoding='utf-8') as stream:
        json.dump(additional, stream)


def _load_adapter_state(adapter_dir: str) -> Dict[str, Any]:
    """Load one adapter directory's weights (safetensors only, mirroring legacy's refusal of .bin)."""
    weights_path = os.path.join(adapter_dir, _ADAPTER_WEIGHTS_NAME)
    if not os.path.isfile(weights_path):
        raise ValueError(f'{adapter_dir} carries no {_ADAPTER_WEIGHTS_NAME}: the safetensors file is '
                         'required to convert the adapter (the .bin form is refused to prevent '
                         'malicious code).')
    from safetensors.torch import load_file

    return load_file(weights_path)


def _save_adapter_state(state_dict: Dict[str, Any], adapter_dir: str) -> None:
    from safetensors.torch import save_file

    save_file(state_dict, os.path.join(adapter_dir, _ADAPTER_WEIGHTS_NAME), metadata={'format': 'pt'})


def _save_to_peft_format(ckpt_dir: str, output_dir: str) -> None:
    """The conversion kernel: validate, copy, rewrite each adapter, hoist ``default`` to the top."""
    if not os.path.exists(ckpt_dir):
        raise ValueError(f'ckpt_dir: {ckpt_dir} must exist on local disk.')
    if os.path.exists(os.path.join(ckpt_dir, _EXTRA_STATE_DIR)):
        raise ValueError('Cannot convert to peft format: the checkpoint carries extra state dicts '
                         f'({_EXTRA_STATE_DIR}/) that the peft layout has no place for.')
    adapter_names = [sub_dir for sub_dir in os.listdir(ckpt_dir)
                     if os.path.isfile(os.path.join(ckpt_dir, sub_dir, _ADAPTER_CONFIG_NAME))]
    configs = {}
    for adapter in adapter_names:
        with open(os.path.join(ckpt_dir, adapter, _ADAPTER_CONFIG_NAME), encoding='utf-8') as stream:
            configs[adapter] = json.load(stream)
        _reject_unconvertible(configs[adapter])

    os.makedirs(output_dir, exist_ok=True)
    if ckpt_dir != output_dir:
        shutil.copytree(ckpt_dir, output_dir, dirs_exist_ok=True)

    for adapter in adapter_names:
        adapter_dir = os.path.join(output_dir, adapter)
        _save_adapter_state(_peft_state_dict(_load_adapter_state(adapter_dir), adapter), adapter_dir)
        _write_peft_adapter_config(configs[adapter], adapter_dir)

    if 'default' in adapter_names:
        # Hoist the default adapter to the top level: peft's load path expects the config and the
        # weights side by side, with no per-adapter subdirectory. Only adapter_config.json moves
        # explicitly; the rest of the subdirectory (including additional_config.json, which has no
        # meaning at the top level) leaves with the rmtree, matching legacy's exact output.
        default_dir = os.path.join(output_dir, 'default')
        shutil.move(os.path.join(default_dir, _ADAPTER_CONFIG_NAME),
                    os.path.join(output_dir, _ADAPTER_CONFIG_NAME))
        _save_adapter_state(_load_adapter_state(default_dir), output_dir)
        shutil.rmtree(default_dir)
