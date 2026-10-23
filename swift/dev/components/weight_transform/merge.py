# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

from swift.dev.utils.logger import get_logger

from .base import BaseWeightTransform, StreamingWeightTransform

if TYPE_CHECKING:
    from swift.dev.config import CheckpointConfig, ModelConfig, TemplateConfig, TunerConfig

logger = get_logger()


class MergeComponent(BaseWeightTransform):
    """Fold ``tuner_config.adapters`` into the base weights of ``model_config``.

    Named after the operation rather than after one adapter format. The base is loaded in full
    precision (a quantized base cannot absorb adapter deltas correctly) and each adapter is merged
    in turn, which is additive -- equivalent to stacking them all at once.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        tuner_config: TunerConfig,
        *,
        template_config: Optional[TemplateConfig] = None,
        checkpoint_config: Optional[CheckpointConfig] = None,
        output_dir: Optional[str] = None,
        device_map: Optional[Any] = None,
        replace_if_exists: bool = False,
        parallel_size: int = 1,
    ):
        super().__init__(model_config, output_dir=output_dir, checkpoint_config=checkpoint_config)
        self.tuner_config = tuner_config
        self.template_config = template_config
        #: Placement for this merge load; infer/deploy pass ``'cpu'`` to fold without touching the GPU.
        self.device_map = device_map
        self.replace_if_exists = replace_if_exists
        #: Streaming data-parallel workers; 1 merges in-process (see ``StreamingWeightTransform``).
        self.parallel_size = parallel_size

    def validate(self) -> None:
        if not self.tuner_config.adapters:
            raise ValueError('run_merge_lora needs TunerConfig.adapters: there is nothing to merge without '
                             'at least one adapter checkpoint.')
        if not self.model_config.model:
            raise ValueError('ModelConfig.model is required: the adapters are merged INTO this base model.')

    def default_output_dir(self) -> str:
        """Legacy's convention: merge next to the first adapter, so no extra argument is needed."""
        return f'{self.tuner_config.adapters[0]}-merged'

    def skip(self, output_dir: str) -> bool:
        """An existing directory is left untouched unless ``replace_if_exists``.

        Merging is idempotent, and re-writing multi-GB weights by accident is expensive.
        """
        if os.path.exists(output_dir) and not self.replace_if_exists:
            logger.info(f'The weight directory for the merged LoRA already exists in {output_dir}, '
                        'skipping the saving process.')
            return True
        return False

    def run(self, output_dir: str) -> str:
        """Merge through the streaming path when eligible, otherwise the materialized path.

        The streaming path keeps peak memory at a single module; the materialized path -- load the
        whole base, ``merge_and_unload``, ``save_checkpoint`` -- is the fallback for everything
        streaming deliberately does not cover, so the component still produces the same result it
        did before this dispatch existed.
        """
        streaming = StreamingMerge(
            self.model_config,
            self.tuner_config.adapters,
            checkpoint_config=self.checkpoint_config,
            parallel_size=self.parallel_size)
        if streaming.is_supported():
            logger.info('Merge LoRA (streaming, per-module)...')
            streaming.run(output_dir)
            logger.info(f'Successfully merged LoRA and saved in `{output_dir}`.')
            return output_dir
        return self._run_materialized(output_dir)

    def _run_materialized(self, output_dir: str) -> str:
        from peft import PeftModel

        from swift.dev.model import save_checkpoint

        model, processor = self._load_base_model()
        # Built (and attached) before merging because some multimodal templates patch the model on
        # attach; the template is not used afterwards, only that side effect on the model.
        self._build_template(processor, model)

        logger.info('Merge LoRA...')
        # dev ships only peft adapters, so merge with peft directly: each adapter is wrapped,
        # merged into the base weights, and merge_and_unload returns the plain transformers model.
        for adapter in self.tuner_config.adapters:
            peft_model = PeftModel.from_pretrained(model, adapter)
            self._check_tie_word_embeddings(peft_model)
            model = peft_model.merge_and_unload()

        logger.info('Saving merged weights...')
        save_checkpoint(
            model,
            processor,
            output_dir,
            safe_serialization=(self.checkpoint_config.safe_serialization if self.checkpoint_config else True),
            max_shard_size=(self.checkpoint_config.max_shard_size if self.checkpoint_config else '5GB'),
            # Copy the adapter dir's own extra files (args.json, chat template, ...) alongside the
            # weights, so the merged output is self-contained.
            model_dirs=list(self.tuner_config.adapters),
            additional_saved_files=model.model_meta.additional_saved_files)
        logger.info(f'Successfully merged LoRA and saved in `{output_dir}`.')
        return output_dir

    def _load_base_model(self):
        """Load the base model in full precision for the merge.

        A quantized base cannot absorb adapter deltas correctly (huggingface/peft#2321), and the
        merge CLI parses no load-time quantization, so the base is full-precision by construction.
        ``device_map`` rides in as a load override because ``ModelConfig`` no longer carries it.
        """
        from swift.dev.builders import apply_new_special_tokens, load_model_processor, resolve_new_special_tokens

        logger.info(f'merge_device_map: {self.device_map}')
        overrides = {} if self.device_map is None else {'device_map': self.device_map}
        # Legacy's loader derived a default placement unconditionally ('auto' whenever no explicit
        # map was given); without it a raw load materializes the whole base in host RAM. The
        # streaming path never loads weights, so this only shapes the materialized fallback.
        # An explicit map (infer/deploy pin 'cpu') or model_kwargs entry wins.
        if 'device_map' not in overrides and 'device_map' not in (self.model_config.model_kwargs or {}):
            overrides['device_map'] = 'auto'
        model, processor = load_model_processor(self.model_config, load_model=True, **overrides)
        # The base must carry the adapter's vocabulary before peft injects against it: a
        # new-special-token run trained with a resized embedding, and the shapes only line up
        # after the same resize here.
        apply_new_special_tokens(model, processor, model.config, resolve_new_special_tokens(self.model_config))
        return model, processor

    def _build_template(self, processor, model):
        """Build the template and attach the model when the template needs it.

        Multimodal templates read submodules off the model during encoding; legacy wires this up
        inside prepare_model_template. It matters here only because of the attach-time patching.
        """
        from swift.dev.builders import build_template
        from swift.dev.config import TemplateConfig

        template = build_template(self.template_config or TemplateConfig(), processor)
        if getattr(template, 'use_model', False):
            template.model = model
        return template

    @staticmethod
    def _check_tie_word_embeddings(model) -> None:
        """Untie word embeddings when only ONE side of the tie was actually trained.

        If a tuner wrapped the embeddings via modules_to_save, the two are no longer the same
        tensor but the config still claims tie_word_embeddings=True -- reloading would drop the
        trained output embedding. Flipping the flag keeps both; guarded broadly (like legacy) as
        a best-effort fix-up.
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


class StreamingMerge(StreamingWeightTransform):
    """Fold LoRA adapters into the base weights one module at a time, never holding the whole model.

    A meta base supplies the architecture, peft injects the adapters; each base tensor passes
    through untouched or, on a LoRA target, is materialized alone, merged via peft's own
    ``LoraLayer.merge()`` -- bit-identical to ``merge_and_unload`` for every variant whose
    injection does not read the base weight -- emitted, and released. Scope: ``is_supported``.
    """

    #: peft nests the wrapped transformers model under ``base_model.model.``, so stripping this prefix
    #: turns a peft module name back into the base-model module path used by the checkpoint keys.
    _PEFT_BASE_PREFIX = 'base_model.model.'

    def __init__(
        self,
        model_config: ModelConfig,
        adapters,
        *,
        checkpoint_config: Optional[CheckpointConfig] = None,
        parallel_size: int = 1,
    ):
        super().__init__(model_config, checkpoint_config=checkpoint_config, parallel_size=parallel_size)
        self.adapters = list(adapters)
        self._base_model = None
        self._base_dir = None
        self._processor = None
        self._peft_model = None
        #: base-model module path -> the peft LoRA layer wrapping it.
        self._lora_modules = {}
        #: base-model module path -> its merged ``{'weight', 'bias'?}``, held only until fully emitted.
        self._merged_cache = {}
        #: peft adapter names loaded onto the base, in ``self.adapters`` order; passed to ``merge()``.
        self._adapter_names = []

    def _clone_for_worker(self):
        """Rebuild an unprepared ``StreamingMerge`` for a Ray worker (the base clone cannot pass adapters)."""
        return StreamingMerge(self.model_config, self.adapters, checkpoint_config=self.checkpoint_config)

    def is_supported(self) -> bool:
        """Streaming-compatible? LoRA-family only; no modules_to_save, DoRA, or base-modifying init."""
        from peft import PeftConfig
        from peft.utils import PeftType

        from swift.dev.builders.adapter import is_base_modifying_init

        if self.checkpoint_config is not None and self.checkpoint_config.safe_serialization is False:
            # The streaming saver only writes safetensors; pickle ``.bin`` output needs the
            # materialized path.
            return False
        if self.model_config.new_special_tokens:
            # The merge needs the base embedding resized to the adapter's vocabulary; the meta
            # base streams the checkpoint's own row count, so the shapes would not line up.
            return False
        for adapter in self.adapters:
            config = PeftConfig.from_pretrained(adapter)
            if config.peft_type != PeftType.LORA:
                return False
            # modules_to_save can untie embeddings and add output keys absent from the base
            # checkpoint, which a writer streaming the base key set cannot reproduce.
            if getattr(config, 'modules_to_save', None):
                return False
            if getattr(config, 'use_dora', False):
                # DoRA's magnitude vector is derived from the base weight and saved under a bare key
                # the generic loader cannot place onto a meta base; fall back, do not special-case.
                return False
            init = getattr(config, 'init_lora_weights', True)
            if is_base_modifying_init(init):
                # These inits read or rewrite the base weight at injection: broken on a meta base, and
                # the raw adapter only fits the residual base they produce. Materialized re-runs the init.
                logger.warning(
                    f'`{adapter}` uses base-modifying LoRA init ({init!r}): merging via the materialized '
                    'path, which re-runs that init and is only correct against the base checkpoint (same '
                    'dtype) the adapter was trained on.')
                return False
        return True

    def prepare(self) -> None:
        """Build the weightless base, inject every adapter, and index the LoRA target modules."""
        from accelerate import init_empty_weights
        from peft import PeftModel
        from peft.tuners.lora import LoraLayer

        from swift.dev.builders import load_model_processor

        # return_dummy_model builds ``cls(config)`` without reading the checkpoint; under
        # init_empty_weights every parameter lands on meta. download_model=True still fetches it.
        with init_empty_weights():
            base, self._processor = load_model_processor(
                self.model_config,
                load_model=True,
                return_dummy_model=True,
                download_model=True)
        self._base_model = base
        self._base_dir = base.model_dir

        first, *rest = self.adapters
        peft_model = PeftModel.from_pretrained(base, first, adapter_name='adapter_0', low_cpu_mem_usage=True)
        self._rematerialize_adapter(peft_model, first, 'adapter_0')
        for index, adapter in enumerate(rest, start=1):
            name = f'adapter_{index}'
            peft_model.load_adapter(adapter, adapter_name=name, low_cpu_mem_usage=True)
            self._rematerialize_adapter(peft_model, adapter, name)
        # Every adapter is merged by name per module: PeftModel.set_adapter takes one name, and
        # LoRA deltas are additive, so merging the list per module equals merge-one-then-next.
        self._adapter_names = [f'adapter_{index}' for index in range(len(self.adapters))]
        self._peft_model = peft_model

        for name, module in peft_model.named_modules():
            if isinstance(module, LoraLayer):
                self._lora_modules[name.removeprefix(self._PEFT_BASE_PREFIX)] = module

    @staticmethod
    def _rematerialize_adapter(peft_model, adapter_dir: str, adapter_name: str) -> None:
        """Pull one adapter's weights off meta and onto CPU, leaving the base weightless.

        peft couples an adapter's device to its base layer's, so injecting onto a meta base parks
        ``lora_A``/``lora_B`` on meta with no data -- the adapters are small, so reload them and
        assign directly. ``_insert_adapter_name_into_state_dict`` is peft's own key remapping (the
        on-disk keys omit the adapter-name segment), so the assigned keys line up exactly.
        """
        from peft.utils import load_peft_weights
        from peft.utils.save_and_load import _insert_adapter_name_into_state_dict

        weights = load_peft_weights(adapter_dir, device='cpu')
        weights = _insert_adapter_name_into_state_dict(weights, adapter_name, 'lora_')
        peft_model.load_state_dict(weights, strict=False, assign=True)

    def source_model_dir(self) -> str:
        return self._base_dir

    def max_shard_size(self) -> str:
        return self.checkpoint_config.max_shard_size if self.checkpoint_config else '5GB'

    def stream_tensors(self, state: dict):
        """Yield every base key, merging LoRA targets in place and passing the rest through.

        The output key set is exactly the base checkpoint's: a merge changes a weight's values,
        never its key. A module's merged tensors are freed once all its keys (weight and, when
        present, bias) have been emitted, which -- with the saver flushing per shard -- caps peak
        memory at one shard plus one module.
        """
        pending = {}
        for key in state:
            module_path, _, suffix = key.rpartition('.')
            if suffix in ('weight', 'bias') and module_path in self._lora_modules:
                pending.setdefault(module_path, set()).add(key)

        for key, lazy in state.items():
            module_path, _, suffix = key.rpartition('.')
            if suffix in ('weight', 'bias') and module_path in self._lora_modules:
                yield key, self._merge_module(module_path, state)[suffix]
                pending[module_path].discard(key)
                if not pending[module_path]:
                    self._merged_cache.pop(module_path, None)
            else:
                yield key, lazy.load()

    def _merge_module(self, module_path: str, state: dict) -> dict:
        """Materialize one module's base weight, fold in the LoRA delta via peft, and cache it.

        The merged tensors are cached until every key of the module has been emitted (weight and
        bias arrive as separate keys, in checkpoint order). Right after merging, the base layer is
        reset to a meta parameter so peft holds no materialized weight while the next module runs.
        """
        cached = self._merged_cache.get(module_path)
        if cached is not None:
            return cached
        import torch

        module = self._lora_modules[module_path]
        base_layer = module.get_base_layer()
        weight = state[f'{module_path}.weight'].load()
        base_layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        bias_key = f'{module_path}.bias'
        has_bias = bias_key in state
        bias = state[bias_key].load() if has_bias else None
        if has_bias:
            base_layer.bias = torch.nn.Parameter(bias, requires_grad=False)

        module.merge(adapter_names=self._adapter_names)
        result = {'weight': base_layer.weight.data}
        if has_bias:
            result['bias'] = base_layer.bias.data
        self._merged_cache[module_path] = result

        base_layer.weight = torch.nn.Parameter(
            torch.empty(weight.shape, device='meta', dtype=weight.dtype), requires_grad=False)
        if has_bias:
            base_layer.bias = torch.nn.Parameter(
                torch.empty(bias.shape, device='meta', dtype=bias.dtype), requires_grad=False)
        return result

    def save_metadata(self, output_dir: str) -> None:
        """Write config / generation_config / tokenizer and copy the adapters' sidecar files.

        Reproduces only the non-weight half of ``save_checkpoint``: the config from the same
        post-processed meta base the materialized path would save, and ``save_checkpoint(None,
        ...)`` reused verbatim for the processor and extra-file copy, keeping the sidecar set
        identical to the materialized output.
        """
        from swift.dev.model import save_checkpoint

        base = self._base_model
        base.config.save_pretrained(output_dir)
        if getattr(base, 'generation_config', None) is not None:
            base.generation_config.save_pretrained(output_dir)
        model_meta = getattr(base, 'model_meta', None)
        save_checkpoint(
            None,
            self._processor,
            output_dir,
            model_dirs=list(self.adapters) + [self._base_dir],
            additional_saved_files=(model_meta.additional_saved_files if model_meta is not None else None))
