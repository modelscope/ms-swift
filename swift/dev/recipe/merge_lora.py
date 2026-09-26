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
from typing import TYPE_CHECKING, Any, Optional

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

    from peft import PeftModel
    from swift.model import save_checkpoint
    model, processor = _load_base_model(model_config, device_map=device_map)
    # Built (and attached) before merging because some multimodal templates patch the model on attach;
    # the template itself is not used afterwards, only its side effect on the model.
    _build_template(template_config, processor, model)

    logger.info('Merge LoRA...')
    # dev only ships peft adapters, so merge with peft directly instead of swift.tuners.Swift. Each
    # adapter is wrapped, merged into the base weights, and unwound in turn: merge_and_unload returns the
    # plain transformers model (no SwiftModel/PeftModel shell to strip afterwards), and merging
    # sequentially is additive -- equivalent to the old loop that stacked adapters onto `model`.
    for adapter in tuner_config.adapters:
        peft_model = PeftModel.from_pretrained(model, adapter)
        _check_tie_word_embeddings(peft_model)
        model = peft_model.merge_and_unload()

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
    from copy import copy

    from swift.dev.builders import load_model_processor

    resolved_device_map = device_map or model_config.device_map
    logger.info(f'merge_device_map: {resolved_device_map}')
    load_config = copy(model_config)
    load_config.device_map = resolved_device_map
    return load_model_processor(load_config, load_model=True)


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
    from swift.dev.utils import HfConfigFactory

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


def _render_ollama_parts(template, parts, placeholder: str, replacement: str) -> str:
    text = ''
    for part in parts:
        if isinstance(part, str):
            text += part.replace(placeholder, replacement)
        elif isinstance(part, (tuple, list)):
            if part and isinstance(part[0], int):
                text += template.tokenizer.decode(part)
            else:
                for name in part:
                    if name == 'bos_token_id':
                        text += template.tokenizer.bos_token or ''
                    elif name == 'eos_token_id':
                        text += template.tokenizer.eos_token or ''
                    else:
                        raise ValueError(f'Unknown template token: {name}')
    return text


def run_export_ollama(model_config, template_config, generation_config, output_dir: str) -> str:
    """Write an Ollama Modelfile for a resolved local model directory."""
    from swift.dev.builders import build_template, load_model_processor

    if not model_config.model or not os.path.isdir(model_config.model):
        raise ValueError('Ollama export requires a local model directory.')
    os.makedirs(output_dir, exist_ok=True)
    _, processor = load_model_processor(model_config)
    template = build_template(template_config, processor)
    meta = template.template_meta
    suffix = _render_ollama_parts(template, meta.suffix, '', '')
    with open(os.path.join(output_dir, 'Modelfile'), 'w', encoding='utf-8') as file:
        file.write(f'FROM {model_config.model}\n')
        file.write('TEMPLATE """{{ if .System }}')
        file.write(_render_ollama_parts(template, meta.system_prefix, '{{SYSTEM}}', '{{ .System }}'))
        file.write('{{ else }}')
        file.write(_render_ollama_parts(template, meta.prefix, '', ''))
        file.write('{{ end }}{{ if .Prompt }}')
        file.write(_render_ollama_parts(template, meta.prompt, '{{QUERY}}', '{{ .Prompt }}'))
        file.write('{{ end }}{{ .Response }}')
        file.write(suffix + '"""\n')
        file.write(f'PARAMETER stop "{suffix}"\n')
        for stop_word in generation_config.stop_words:
            file.write(f'PARAMETER stop "{stop_word}"\n')
        temperature = generation_config.temperature if generation_config.temperature is not None else 1.0
        top_k = generation_config.top_k if generation_config.top_k is not None else -1
        top_p = generation_config.top_p if generation_config.top_p is not None else 1.0
        penalty = generation_config.repetition_penalty if generation_config.repetition_penalty is not None else 1.0
        file.write(f'PARAMETER temperature {temperature}\n')
        file.write(f'PARAMETER top_k {top_k}\n')
        file.write(f'PARAMETER top_p {top_p}\n')
        file.write(f'PARAMETER repeat_penalty {penalty}\n')
    return output_dir


def run_to_peft_format(adapter: str, output_dir: str) -> str:
    """Convert a Swift-format adapter to PEFT's directory layout."""
    from swift.tuners import swift_to_peft_format
    os.makedirs(output_dir, exist_ok=True)
    return swift_to_peft_format(adapter, output_dir)


def run_push_to_hub(folder_path: str, checkpoint_config, dataset_config, commit_message: str) -> str:
    """Upload one produced model directory using the configured Hub implementation."""
    if not checkpoint_config.hub_model_id:
        raise ValueError('--hub_model_id is required with --push_to_hub.')
    if not os.path.isdir(folder_path):
        raise ValueError(f'Hub upload source is not a directory: {folder_path}')
    from swift.dev.utils.hub import get_hub
    get_hub(dataset_config.use_hf).push_to_hub(
        checkpoint_config.hub_model_id,
        folder_path,
        token=dataset_config.hub_token,
        private=checkpoint_config.hub_private_repo,
        revision=checkpoint_config.hub_revision,
        commit_message=commit_message)
    return folder_path
