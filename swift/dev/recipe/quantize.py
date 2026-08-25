"""run_quantize: post-training quantization as a dev recipe.

dev counterpart of legacy ``swift export --quant_method ...``
(swift/pipelines/export/quant.py::QuantEngine), split along the same seam the twinkle quantizers
draw:

  - the BACKEND work (AWQ/GPTQ packing, the autoawq/optimum monkey-patches, the transformers
    quantization_config objects) lives in ``twinkle.quantizer`` -- one class per scheme, each owning
    its own knobs (group_size, zero_point, version, ...). This recipe does not reimplement any of it.
  - the DATA work (loading a calibration set and encoding it into the exact shape each backend
    expects) lives here, because ``CalibrationQuantizer`` deliberately takes calibration samples from
    the caller: twinkle has no opinion on where they come from, and swift's template/dataset stack is
    what produces them.

So the flow is: build template -> build calibration samples -> get_quantizer(...) -> quantize ->
save. Load-time schemes (bnb/fp8/hqq/quanto/eetq) skip calibration entirely and do not even load the
model: they only emit a quantization_config for ``from_pretrained``, which is what
``ConfigQuantizer.quantize`` refuses to fake.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from swift.dev.config import DatasetConfig, ModelConfig, QuantizeConfig, TemplateConfig

logger = logging.getLogger(__name__)

# Schemes whose scales are fitted by running the model over real samples. The rest are applied by
# transformers at load time (see twinkle.quantizer.ConfigQuantizer) and need no data and no model.
CALIBRATION_METHODS = ('awq', 'gptq', 'gptq_v2')


def run_quantize(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    quantize_config: QuantizeConfig,
    dataset_config: Optional[DatasetConfig] = None,
    *,
    output_dir: str = 'output',
    quant_n_samples: int = 256,
    **quantizer_kwargs,
) -> str:
    """Quantize ``model_config`` and write the result to ``output_dir``; returns that path.

    ``quantizer_kwargs`` are forwarded verbatim to the twinkle quantizer, which is where every
    backend knob already lives (group_size / zero_point / version / v2 / bnb_4bit_* ...). They are
    NOT mirrored onto QuantizeConfig: that would duplicate the same fields in two places and let
    them drift.

    ``dataset_config`` is required for AWQ/GPTQ and unused otherwise. ``quant_n_samples`` caps how
    many rows are encoded for calibration (legacy's ``--quant_n_samples``).
    """
    quant_method = quantize_config.quant_method
    if not quant_method:
        raise ValueError('QuantizeConfig.quant_method is required (e.g. awq / gptq / bnb / fp8).')
    # fp8 has no bit-width knob; everything else does. Mirrors legacy QuantEngine.quantize's guard.
    if quantize_config.quant_bits is None and quant_method != 'fp8':
        raise ValueError(f'QuantizeConfig.quant_bits is required for quant_method={quant_method!r}.')

    kwargs = _quantizer_kwargs(quantize_config, quantizer_kwargs)
    if quant_method not in CALIBRATION_METHODS:
        return _run_load_time(quant_method, kwargs, output_dir=output_dir)
    return _run_calibration(
        quant_method,
        kwargs,
        model_config=model_config,
        template_config=template_config,
        dataset_config=dataset_config,
        output_dir=output_dir,
        quant_n_samples=quant_n_samples)


def _quantizer_kwargs(quantize_config: QuantizeConfig, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """QuantizeConfig -> twinkle quantizer kwargs, with explicit call-site overrides winning.

    Only the fields QuantizeConfig actually owns are translated; the bnb_4bit_* / hqq_axis ones are
    passed through under the names the corresponding quantizer declares. Unset (None) values are
    dropped so each quantizer's own default applies rather than being overwritten with None.
    """
    kwargs: Dict[str, Any] = {'quant_bits': quantize_config.quant_bits}
    method = quantize_config.quant_method
    if method == 'bnb':
        kwargs.update(
            bnb_4bit_compute_dtype=quantize_config.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=quantize_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=quantize_config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_storage=quantize_config.bnb_4bit_quant_storage)
    elif method == 'hqq':
        kwargs['axis'] = quantize_config.hqq_axis
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    kwargs.update(overrides)
    return kwargs


def _run_load_time(quant_method: str, kwargs: Dict[str, Any], *, output_dir: str) -> str:
    """bnb / fp8 / hqq / quanto / eetq: emit the quantization_config, do not touch weights.

    These schemes quantize while transformers MATERIALIZES the weights, so there is nothing to do to
    an already-loaded model -- ``ConfigQuantizer.quantize`` raises rather than pretend otherwise. The
    useful artifact is the config itself, so it is written as quantization_config.json for the caller
    to pass to ``from_pretrained`` (or to merge into a model config).
    """
    import json

    from twinkle.quantizer import get_quantizer

    quantizer = get_quantizer(quant_method, **kwargs)
    quant_config = quantizer.get_quantization_config()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'quantization_config.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(quant_config.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info(f'{quant_method} is applied at LOAD time; no weights were rewritten. '
                f'Wrote the quantization_config to `{path}` -- pass it to from_pretrained '
                '(quantization_config=...) to load the model quantized.')
    return path


def _run_calibration(
    quant_method: str,
    kwargs: Dict[str, Any],
    *,
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: Optional[DatasetConfig],
    output_dir: str,
    quant_n_samples: int,
) -> str:
    """AWQ / GPTQ: load the model, fit scales on calibration data, pack, save."""
    from twinkle.quantizer import get_quantizer

    from swift.model import save_checkpoint
    from swift.utils import HfConfigFactory

    if dataset_config is None or not dataset_config.dataset:
        raise ValueError(f'quant_method={quant_method!r} is calibration-based and needs '
                         'DatasetConfig.dataset to fit the quantization scales.')

    # Probe the backend dependency first (requires() names the pip package); without this the AWQ
    # path would surface a bare `No module named 'awq'` from _load_model_template instead.
    _require_backend(quant_method)

    model, template, processor = _load_model_template(quant_method, model_config, template_config)
    # Calibration is inference over real batches; caches would both waste memory and (for GPTQ's
    # block-by-block walk) change what each block sees. Legacy disables it the same way.
    model.config.use_cache = False
    HfConfigFactory.set_config_attr(model.config, 'use_cache', False)

    is_multimodal = bool(model.model_meta.is_multimodal)
    kwargs.setdefault('modules_to_not_convert', _modules_to_not_convert(model))
    if quant_method in ('gptq', 'gptq_v2'):
        kwargs.setdefault('is_moe_model', bool(model.model_info.is_moe_model))
        kwargs.setdefault('model_type', model.model_info.model_type)
        kwargs.setdefault('block_name_to_quantize', _block_name_to_quantize(model))
    quantizer = get_quantizer(quant_method, **kwargs)

    calib = _build_calib_data(
        quant_method,
        template,
        dataset_config,
        max_length=template_config.max_length,
        n_samples=quant_n_samples,
        is_multimodal=is_multimodal)
    if quant_method in ('gptq', 'gptq_v2') and is_multimodal:
        # optimum's prepare_dataset is bypassed, so the per-batch collate+device move that the
        # multimodal path needs has to happen here (legacy: QuantEngine._prepare_gptq_dataset).
        calib = _collate_multimodal(calib, model, template, batch_size=kwargs.get('batch_size', 1))
    quantizer.set_calib_data(calib, tokenizer=processor)

    logger.info(f'Start {quant_method} quantization ({len(calib)} calibration batches)...')
    model = quantizer.quantize(model)
    quantizer.save(model, output_dir)

    # Copy the processor + the model's declared extra files so the output is directly loadable
    # (legacy does the same after every backend's writer runs).
    save_checkpoint(
        None,
        processor,
        output_dir,
        model_dirs=[model.model_dir],
        additional_saved_files=model.model_meta.additional_saved_files)
    logger.info(f'Successfully quantized the model and saved in `{output_dir}`.')
    return output_dir


def _require_backend(quant_method: str) -> None:
    """Fail with the installable package name before any model/data work.

    The twinkle quantizers already declare their dependencies via ``requires()``, but that check runs
    in their __init__ -- which happens AFTER the model is loaded here. Probing the same names up front
    keeps the actionable message ("Required package 'autoawq' is not installed") and avoids paying for
    a multi-GB load that is going to fail.
    """
    from twinkle.utils import requires

    if quant_method == 'awq':
        requires('autoawq')
    else:
        requires('optimum')
        requires('gptqmodel')


def _load_model_template(quant_method: str, model_config: ModelConfig, template_config: TemplateConfig):
    """Load the model + template for calibration, in the shape the backend requires.

    AWQ is the odd one: autoawq owns both the packing and the writer, so the model must be an
    ``AutoAWQForCausalLM`` wrapper rather than a plain transformers model (legacy passes the same
    auto_model_cls). The template is put in 'train' mode because calibration wants the full
    prompt+response token stream, not a generation prompt.
    """
    from swift.dev.builders import build_template
    from swift.model import get_model_processor

    kwargs = {}
    if quant_method == 'awq':
        from awq import AutoAWQForCausalLM
        kwargs['auto_model_cls'] = AutoAWQForCausalLM
    model, processor = get_model_processor(model_config.model, **kwargs)
    template = build_template(template_config, processor)
    template.set_mode('train')
    if quant_method == 'awq':
        # autoawq nests the real transformers module; the template's own forward hooks need it.
        template.model = model.model
    else:
        template.model = model
    return model, template, processor


def _modules_to_not_convert(model) -> Optional[List[str]]:
    """Layers that must stay in full precision (MoE routers, vision towers, lm_head).

    Same rule as legacy ``QuantizeArguments.get_modules_to_not_convert``: routing gates decide which
    experts fire and vision/adapter stacks are numerically fragile, so quantizing them costs far more
    quality than the memory it saves. lm_head is appended only when something else was excluded --
    a fully converted model keeps the backend's own default handling.
    """
    res: List[str] = []
    if getattr(model.model_info, 'is_moe_model', False):
        res += ['mlp.gate', 'mlp.shared_expert_gate']
    model_arch = getattr(model.model_meta, 'model_arch', None)
    if model_arch is not None:
        for key in ('vision_tower', 'aligner'):
            value = getattr(model_arch, key, None)
            if value:
                res += list(value)
    if not res:
        return None
    res.append('lm_head')
    return res


def _block_name_to_quantize(model) -> Optional[str]:
    """Scope GPTQ's block walk to the language model on a multimodal checkpoint.

    ``GptqQuantizer.get_block_name_to_quantize`` finds the decoder ModuleList itself, but it needs to
    be told to look inside the LLM sub-module first -- otherwise on a VLM it may latch onto the
    vision tower's layer list.
    """
    from twinkle.quantizer import GptqQuantizer

    model_arch = getattr(model.model_meta, 'model_arch', None)
    prefix = None
    if model_arch is not None and getattr(model_arch, 'language_model', None):
        language_model = [lm for lm in model_arch.language_model if not lm.endswith('lm_head')]
        if len(language_model) == 1:
            prefix = language_model[0]
    return GptqQuantizer.get_block_name_to_quantize(model, language_model_prefix=prefix)


def _build_calib_data(quant_method: str, template, dataset_config: DatasetConfig, *, max_length: Optional[int],
                      n_samples: int, is_multimodal: bool) -> List[Any]:
    """Encode a calibration set into the shape the backend expects.

    Migrated from legacy ``QuantEngine._get_quant_dataset``. Three output shapes, because the
    backends disagree:
      - AWQ            -> a list of ``LongTensor[1, block_size]`` token blocks.
      - GPTQ (text)    -> a list of ``{'input_ids': [...]}`` dicts of the same blocks.
      - GPTQ (multimodal) -> the per-sample encoded dicts as-is (image features cannot be
        concatenated into fixed-size token blocks), with `labels` dropped since nothing is trained.

    Text samples are concatenated and re-split into ``max_length`` blocks so every calibration batch
    is uniformly sized -- short samples would otherwise make the estimated activation ranges depend
    on padding.
    """
    import torch
    from tqdm import tqdm

    from swift.dataset import load_dataset
    from swift.template import MaxLengthError

    is_gptq = quant_method in ('gptq', 'gptq_v2')
    keep_per_sample = is_gptq and is_multimodal

    # Only the train split is used for calibration (split_dataset_ratio=0), then shuffled so the
    # samples are not all from one region of the dataset.
    dataset = load_dataset(
        dataset_config.dataset,
        split_dataset_ratio=0,
        shuffle=dataset_config.dataset_shuffle,
        seed=dataset_config.data_seed,
        num_proc=dataset_config.dataset_num_proc,
        load_from_cache_file=dataset_config.load_from_cache_file,
        columns=dataset_config.columns,
        strict=dataset_config.strict,
        use_hf=dataset_config.use_hf,
        hub_token=dataset_config.hub_token)[0]
    logger.info(f'quant_dataset: {dataset}')
    dataset = dataset.shuffle()

    samples: List[Any] = []
    n_encoded = 0
    prog_bar = tqdm(total=n_samples, dynamic_ncols=True)
    with torch.inference_mode():
        for data in dataset:
            try:
                inputs = template.encode(data)
            except MaxLengthError:
                # Over-length rows are skipped rather than truncated: a clipped sample would bias
                # the activation statistics toward prompt-only content.
                continue
            if keep_per_sample:
                inputs.pop('labels', None)
                samples.append(inputs)
            else:
                samples += inputs['input_ids']
            n_encoded += 1
            prog_bar.update()
            if n_encoded == n_samples:
                break
    prog_bar.close()
    if n_encoded == 0:
        raise ValueError('No calibration sample survived encoding: every row exceeded max_length. '
                         'Raise TemplateConfig.max_length or use a shorter dataset.')
    if keep_per_sample:
        return samples

    block_size = max_length or len(samples)
    n_split = max(len(samples) // block_size, 1)
    logger.info(f'Split into {n_split} blocks')
    res: List[Any] = []
    for i in range(n_split):
        input_ids = samples[i * block_size:(i + 1) * block_size]
        res.append({'input_ids': input_ids} if is_gptq else torch.tensor(input_ids)[None])
    return res


def _collate_multimodal(examples: List[Dict[str, Any]], model, template, *, batch_size: int = 1) -> List[Any]:
    """Batch + device-move multimodal GPTQ samples (legacy ``_prepare_gptq_dataset``).

    optimum's own ``prepare_dataset`` is patched out by the quantizer, so its collate step has to be
    reproduced: each group is collated by the template, pushed to the model's device to run the
    multimodal pre-forward hook (which turns pixel values into embeddings), then moved back to CPU so
    the whole calibration set does not sit in GPU memory at once.
    """
    import torch
    from tqdm import tqdm

    from swift.utils import to_device

    res = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(examples), batch_size)):
            batched_inputs = examples[start:start + batch_size]
            inputs = to_device(template.data_collator(batched_inputs), model.device)
            _, inputs = template.pre_forward_hook(model, None, inputs)
            res.append(to_device(inputs, 'cpu'))
    return res
