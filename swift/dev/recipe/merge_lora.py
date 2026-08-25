"""run_merge_lora: fold LoRA adapters back into the base weights, as a dev recipe.

dev counterpart of legacy ``swift export --merge_lora``
(swift/pipelines/export/merge_lora.py::merge_lora), for the transformers path. The mcore/Megatron
equivalent is a different code path and lives in ``swift.dev.recipe.convert`` (ConvertConfig.
mcore_adapter), because an mcore LoRA has to be merged while the model is still in Megatron format.

All dev Configs needed here already exist -- TunerConfig.adapters names the adapters,
CheckpointConfig owns output_dir/safe_serialization/max_shard_size -- so unlike the convert recipe
this one adds no new Config.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from swift.dev.config import CheckpointConfig, ModelConfig, TemplateConfig, TunerConfig

logger = logging.getLogger(__name__)


def run_merge_lora(
    model_config: ModelConfig,
    tuner_config: TunerConfig,
    *,
    template_config: Optional[TemplateConfig] = None,
    checkpoint_config: Optional[CheckpointConfig] = None,
    output_dir: Optional[str] = None,
    device_map: Optional[Any] = None,
    replace_if_exists: bool = False,
) -> str:
    """Merge ``tuner_config.adapters`` into the base model and save it; returns the output directory.

    Defaults to ``'{first adapter}-merged'`` (legacy's convention) so a merge next to the checkpoint
    needs no extra argument. An existing directory is left untouched unless ``replace_if_exists``,
    because merging is idempotent and re-writing multi-GB weights by accident is expensive.
    """
    if not tuner_config.adapters:
        raise ValueError('run_merge_lora needs TunerConfig.adapters: there is nothing to merge without '
                         'at least one adapter checkpoint.')
    if not model_config.model:
        raise ValueError('ModelConfig.model is required: the adapters are merged INTO this base model.')

    resolved = (
        output_dir or (checkpoint_config.output_dir if checkpoint_config else None)
        or f'{tuner_config.adapters[0]}-merged')
    if os.path.exists(resolved) and not replace_if_exists:
        logger.info(f'The weight directory for the merged LoRA already exists in {resolved}, '
                    'skipping the saving process.')
        return resolved

    from swift.model import save_checkpoint
    from swift.tuners import Swift

    model, processor = _load_base_model(model_config, device_map=device_map)
    # Built (and attached) before merging because some multimodal templates patch the model on attach;
    # the template itself is not used afterwards, only its side effect on the model.
    _build_template(template_config, processor, model)
    for adapter in tuner_config.adapters:
        model = Swift.from_pretrained(model, adapter)

    logger.info('Merge LoRA...')
    _check_tie_word_embeddings(model)
    Swift.merge_and_unload(model)
    # Unwrap the SwiftModel/PeftModel shell so what gets saved is a plain transformers model.
    model = model.model

    logger.info('Saving merged weights...')
    save_checkpoint(
        model,
        processor,
        resolved,
        safe_serialization=(checkpoint_config.safe_serialization if checkpoint_config else True),
        max_shard_size=(checkpoint_config.max_shard_size if checkpoint_config else '5GB'),
        # Copies the adapter dir's own extra files (args.json, chat template, ...) alongside the
        # weights, so the merged output is self-contained.
        model_dirs=list(tuner_config.adapters),
        additional_saved_files=model.model_meta.additional_saved_files)
    logger.info(f'Successfully merged LoRA and saved in `{resolved}`.')
    return resolved


def _load_base_model(model_config: ModelConfig, *, device_map: Optional[Any]):
    """Load the base model in full precision, ignoring any quantization on ModelConfig.

    A quantized base cannot absorb LoRA deltas correctly -- peft raises / silently degrades
    (huggingface/peft#2321) -- so the merge always runs on the unquantized weights. legacy does the
    same by clearing args.quant_method before loading.
    """
    from swift.model import get_model_processor

    kwargs: Dict[str, Any] = {}
    if model_config.torch_dtype:
        import torch
        kwargs['torch_dtype'] = getattr(torch, model_config.torch_dtype)
    if model_config.model_type:
        kwargs['model_type'] = model_config.model_type
    resolved_device_map = device_map or model_config.device_map
    if resolved_device_map:
        kwargs['device_map'] = resolved_device_map
    logger.info(f'merge_device_map: {resolved_device_map}')
    return get_model_processor(model_config.model, **kwargs)


def _build_template(template_config: Optional[TemplateConfig], processor, model):
    """Build the template and attach the model when the template needs it.

    Multimodal templates read submodules off the model during encoding; legacy wires this up inside
    prepare_model_template. It matters here only because some templates patch the model on attach.
    """
    from swift.dev.builders import build_template
    from swift.dev.config import TemplateConfig

    template = build_template(template_config or TemplateConfig(), processor)
    if getattr(template, 'use_model', False):
        template.model = model
    return template


def _check_tie_word_embeddings(model) -> None:
    """Untie word embeddings when only ONE side of the tie was actually trained.

    If a tuner wrapped the input/output embeddings via modules_to_save, the two are no longer the
    same tensor, but the config still claims tie_word_embeddings=True -- reloading would then drop the
    trained output embedding and silently restore the input one. Flipping the flag keeps both.
    Guarded broadly (like legacy) because it is a best-effort fix-up: peft internals differ across
    versions and a failure here must not abort an otherwise valid merge.
    """
    from swift.utils import HfConfigFactory

    config = model.config
    try:
        from peft.utils import ModulesToSaveWrapper
        if not HfConfigFactory.get_config_attr(config, 'tie_word_embeddings'):
            return
        for module in [model.get_input_embeddings(), model.get_output_embeddings()]:
            if not isinstance(module, ModulesToSaveWrapper):
                return
        HfConfigFactory.set_config_attr(config, 'tie_word_embeddings', False)
    except Exception:
        pass
