# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from swift.dev.utils.logger import get_logger

from .base import BaseWeightTransform

if TYPE_CHECKING:
    from swift.dev.config import DatasetConfig, ModelConfig, QuantizeConfig, TemplateConfig

logger = get_logger()

# Schemes whose scales are fitted by running the model over real samples. The rest are applied by
# transformers at load time (see twinkle.quantizer.ConfigQuantizer) and need no data and no model.
CALIBRATION_METHODS = ('awq', 'gptq', 'gptq_v2')


class QuantizeComponent(BaseWeightTransform):
    """Quantize ``model_config`` with ``quantize_config.quant_method``.

    Two shapes, split by whether the scheme needs calibration data: load-time (bnb / fp8 / hqq /
    quanto / eetq) hands the scheme to the loader, which quantizes while transformers materializes
    the weights -- no data, no second pass; calibration (awq / gptq / gptq_v2) loads the model in
    full precision and fits the scales over encoded samples before the backend packs and writes.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        template_config: TemplateConfig,
        quantize_config: QuantizeConfig,
        dataset_config: Optional[DatasetConfig] = None,
        *,
        output_dir: Optional[str] = None,
        quant_n_samples: Optional[int] = None,
        safe_serialization: bool = True,
        max_shard_size: str = '5GB',
        **quantizer_kwargs,
    ):
        super().__init__(model_config, output_dir=output_dir)
        self.template_config = template_config
        self.quantize_config = quantize_config
        #: Required for AWQ/GPTQ, unused otherwise.
        self.dataset_config = dataset_config
        #: Caps how many rows are encoded for calibration; falls back to the config's
        #: ``quant_n_samples`` so programmatic callers that only fill QuantizeConfig still get it.
        self.quant_n_samples = quant_n_samples if quant_n_samples is not None else quantize_config.quant_n_samples
        self.safe_serialization = safe_serialization
        self.max_shard_size = max_shard_size
        #: Backend knobs (group_size / zero_point / v2 / bnb_4bit_* ...) forwarded verbatim to the
        #: twinkle quantizer -- NOT mirrored onto QuantizeConfig, where they would drift.
        self.quantizer_kwargs = quantizer_kwargs

    def validate(self) -> None:
        if not self.quantize_config.quant_method:
            raise ValueError('QuantizeConfig.quant_method is required (e.g. awq / gptq / bnb / fp8).')
        # fp8 has no bit-width knob; everything else does. Mirrors legacy QuantEngine.quantize's guard.
        if self.quantize_config.quant_bits is None and self.quantize_config.quant_method != 'fp8':
            raise ValueError(
                f'QuantizeConfig.quant_bits is required for quant_method={self.quantize_config.quant_method!r}.')

    def run(self, output_dir: str) -> str:
        from swift.dev.builders.quantization import quantizer_kwargs

        quant_method = self.quantize_config.quant_method
        kwargs = quantizer_kwargs(
            self.quantize_config, self.quantizer_kwargs, torch_dtype=self.model_config.torch_dtype)
        if quant_method not in CALIBRATION_METHODS:
            return self._run_load_time(quant_method, kwargs, output_dir)
        return self._run_calibration(quant_method, kwargs, output_dir)

    def _run_load_time(self, quant_method: str, kwargs: Dict[str, Any], output_dir: str) -> str:
        """bnb / fp8 / hqq / quanto / eetq: load the model quantized, then save the full directory.

        These schemes quantize WHILE transformers materializes the weights (``ConfigQuantizer.
        quantize`` raises on an already-loaded model), so the scheme's ``quantization_config`` goes
        to the loader and the result is persisted with ``save_pretrained`` + ``save_checkpoint``.
        Not streaming on purpose: the op is model-bound; backends expose no per-tensor writer.
        """
        from twinkle.quantizer import get_quantizer

        from swift.dev.builders import load_model_processor
        from swift.dev.model import save_checkpoint

        quant_config = get_quantizer(quant_method, **kwargs).get_quantization_config()
        model, processor = load_model_processor(
            self.model_config, load_model=True, quantization_config=quant_config)
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(
            output_dir, safe_serialization=self.safe_serialization, max_shard_size=self.max_shard_size)
        # Copy the processor + the model's declared extra files so the output is self-contained
        # and loadable on its own (legacy runs the same save_checkpoint after the weights).
        save_checkpoint(
            None,
            processor,
            output_dir,
            model_dirs=[model.model_dir],
            additional_saved_files=model.model_meta.additional_saved_files)
        logger.info(f'Successfully quantized the model ({quant_method}, applied at load time) '
                    f'and saved in `{output_dir}`.')
        return output_dir

    def _run_calibration(self, quant_method: str, kwargs: Dict[str, Any], output_dir: str) -> str:
        """AWQ / GPTQ: load the model, fit scales on calibration data, pack, save."""
        from twinkle.quantizer import get_quantizer

        from swift.dev.utils import HfConfigFactory
        from swift.dev.model import save_checkpoint

        dataset_config = self.dataset_config
        if dataset_config is None or not dataset_config.dataset:
            raise ValueError(f'quant_method={quant_method!r} is calibration-based and needs '
                             'DatasetConfig.dataset to fit the quantization scales.')

        # Probe the backend dependency first (requires() names the pip package); without this the
        # AWQ path would surface a bare `No module named 'awq'` from _load_model_template instead.
        self._require_backend(quant_method)

        model, template, processor = self._load_model_template(quant_method)
        # Calibration is inference over real batches; caches would both waste memory and (for
        # GPTQ's block-by-block walk) change what each block sees. Legacy disables them too.
        model.config.use_cache = False
        HfConfigFactory.set_config_attr(model.config, 'use_cache', False)

        is_multimodal = bool(model.model_meta.is_multimodal)
        kwargs.setdefault('modules_to_not_convert', self._modules_to_not_convert(model))
        if quant_method in ('gptq', 'gptq_v2'):
            kwargs.setdefault('is_moe_model', bool(model.model_info.is_moe_model))
            kwargs.setdefault('model_type', model.model_info.model_type)
            kwargs.setdefault('block_name_to_quantize', self._block_name_to_quantize(model))
        quantizer = get_quantizer(quant_method, **kwargs)

        calib = self._build_calib_data(
            quant_method,
            template,
            dataset_config,
            max_length=self.template_config.max_length,
            n_samples=self.quant_n_samples,
            is_multimodal=is_multimodal)
        if quant_method in ('gptq', 'gptq_v2') and is_multimodal:
            # optimum's prepare_dataset is bypassed, so the per-batch collate + device move the
            # multimodal path needs happens here (legacy: QuantEngine._prepare_gptq_dataset).
            calib = self._collate_multimodal(calib, model, template, batch_size=kwargs.get('batch_size', 1))
        quantizer.set_calib_data(calib, tokenizer=processor)

        logger.info(f'Start {quant_method} quantization ({len(calib)} calibration batches)...')
        model = quantizer.quantize(model)
        quantizer.save(
            model, output_dir, safe_serialization=self.safe_serialization, max_shard_size=self.max_shard_size)

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

    @staticmethod
    def _require_backend(quant_method: str) -> None:
        """Fail with the installable package name before any model/data work.

        The twinkle quantizers declare their dependencies via ``requires()``, but that check runs
        in their __init__ -- AFTER the model is loaded here. Probing the same names up front keeps
        the actionable message and avoids paying for a multi-GB load that is going to fail.
        """
        from twinkle.utils import requires

        if quant_method == 'awq':
            requires('autoawq')
        else:
            requires('optimum')
            requires('gptqmodel')

    def _load_model_template(self, quant_method: str):
        """Load the model + template for calibration, in the shape the backend requires.

        AWQ is the odd one: autoawq owns both the packing and the writer, so the model must be an
        ``AutoAWQForCausalLM`` wrapper rather than a plain transformers model (legacy passes the
        same auto_model_cls). The template is put in 'train' mode because calibration wants the
        full prompt+response token stream, not a generation prompt.
        """
        from swift.dev.builders import build_template, load_model_processor

        kwargs = {}
        if quant_method == 'awq':
            from awq import AutoAWQForCausalLM
            kwargs['auto_model_cls'] = AutoAWQForCausalLM
        # Calibration runs the FULL-PRECISION model over real batches; default to accelerate's
        # 'auto' device map so only one block is resident. An explicit model_kwargs['device_map'] wins.
        if 'device_map' not in (self.model_config.model_kwargs or {}):
            kwargs['device_map'] = 'auto'
        model, processor = load_model_processor(self.model_config, load_model=True, **kwargs)
        template = build_template(self.template_config, processor)
        template.set_mode('train')
        if quant_method == 'awq':
            # autoawq nests the real transformers module; the template's own forward hooks need it.
            template.model = model.model
        else:
            template.model = model
        return model, template, processor

    @staticmethod
    def _modules_to_not_convert(model) -> Optional[List[str]]:
        """Layers that must stay in full precision (MoE routers, vision towers, lm_head).

        Same rule as legacy ``QuantizeArguments.get_modules_to_not_convert``: routing gates decide
        which experts fire and vision/adapter stacks are numerically fragile. lm_head is appended
        only when something else was excluded -- a fully converted model keeps the backend's
        default handling.
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

    @staticmethod
    def _block_name_to_quantize(model) -> Optional[str]:
        """Scope GPTQ's block walk to the language model on a multimodal checkpoint.

        ``GptqQuantizer.get_block_name_to_quantize`` finds the decoder ModuleList itself, but it
        needs to be told to look inside the LLM sub-module first -- otherwise on a VLM it may
        latch onto the vision tower's layer list.
        """
        from twinkle.quantizer import GptqQuantizer

        model_arch = getattr(model.model_meta, 'model_arch', None)
        prefix = None
        if model_arch is not None and getattr(model_arch, 'language_model', None):
            language_model = [lm for lm in model_arch.language_model if not lm.endswith('lm_head')]
            if len(language_model) == 1:
                prefix = language_model[0]
        return GptqQuantizer.get_block_name_to_quantize(model, language_model_prefix=prefix)

    @staticmethod
    def _build_calib_data(quant_method: str, template, dataset_config: DatasetConfig, *, max_length: Optional[int],
                          n_samples: int, is_multimodal: bool) -> List[Any]:
        """Encode a calibration set into the shape the backend expects (legacy ``_get_quant_dataset``).

        AWQ -> ``LongTensor[1, block_size]`` token blocks; GPTQ text -> ``{'input_ids': [...]}``
        dicts of the same blocks; GPTQ multimodal -> per-sample encoded dicts as-is (image features
        cannot join fixed-size blocks), ``labels`` dropped. Text is concatenated and re-split into
        ``max_length`` blocks so every batch is uniformly sized, not padding-dependent.
        """
        import torch
        from tqdm import tqdm

        from swift.dev.dataset import load_dataset
        from swift.template import MaxLengthError

        is_gptq = quant_method in ('gptq', 'gptq_v2')
        keep_per_sample = is_gptq and is_multimodal

        # Only the train split is used for calibration (split_dataset_ratio=0), then shuffled so
        # the samples are not all from one region of the dataset.
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
                    # Over-length rows are skipped rather than truncated: a clipped sample would
                    # bias the activation statistics toward prompt-only content.
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
            # GPTQ batches are dicts fed straight to model(**batch): twinkle patches optimum's
            # prepare_dataset to identity, so tensor-izing the ids must happen here -- a plain list
            # would reach the embedding as a non-Tensor. AWQ takes bare [1, L] tensors instead.
            res.append({'input_ids': torch.tensor(input_ids)[None]} if is_gptq else torch.tensor(input_ids)[None])
        return res

    @staticmethod
    def _collate_multimodal(examples: List[Dict[str, Any]], model, template, *, batch_size: int = 1) -> List[Any]:
        """Batch + device-move multimodal GPTQ samples (legacy ``_prepare_gptq_dataset``).

        optimum's own ``prepare_dataset`` is patched out by the quantizer, so each group is
        collated by the template, pushed to the model's device to run the multimodal pre-forward
        hook (which turns pixel values into embeddings), then moved back to CPU so the whole
        calibration set does not sit in GPU memory at once.
        """
        import torch
        from tqdm import tqdm

        from swift.dev.utils import to_device

        res = []
        with torch.inference_mode():
            for start in tqdm(range(0, len(examples), batch_size)):
                batched_inputs = examples[start:start + batch_size]
                inputs = to_device(template.data_collator(batched_inputs), model.device)
                _, inputs = template.pre_forward_hook(model, None, inputs)
                res.append(to_device(inputs, 'cpu'))
        return res
