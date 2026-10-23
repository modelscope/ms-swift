# Copyright (c) ModelScope Contributors. All rights reserved.
# The only transform that does NOT stream itself: the megatron bridge already does, sharded by torchrun TP/PP/EP.
from __future__ import annotations

import dataclasses
import math
import os
import shutil
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from swift.dev.utils.logger import get_logger

from .base import BaseWeightTransform
from .mcore_ckpt import load_mcore_dist_checkpoint, load_source_args, save_mcore_dist_checkpoint

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        ConvertConfig,
        DistributedConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        TemplateConfig,
    )
    from twinkle.model.megatron import MegatronStrategy

logger = get_logger()

#: Conversion-forced attention backend. 'unfused' keeps the numerics of the (optional)
#: precision test comparable to the HF side by removing flash/fused kernel variance; it is
#: resolved to mcore's AttnBackend ENUM, never passed as a bare string.
_CONVERT_ATTN_BACKEND = 'unfused'


class ConvertComponent(BaseWeightTransform):
    """Migrate weights between HF safetensors and the mcore ``torch_dist`` checkpoint.

    Direction is taken from ``convert_config``: ``to_mcore=True`` with no ``mcore_model`` ->
    HF -> mcore; ``to_hf=True`` -> mcore -> HF; ``to_mcore=True`` with ``mcore_model`` ->
    mcore -> mcore (reshard, and/or merge ``mcore_adapter``). The two directions are separate
    workers; ``run`` only picks between them, so the routing decision stays readable.

    Model construction goes through twinkle's ``MegatronStrategy`` (mpu init from the
    DeviceMesh + the selected bridge backend), and the torch_dist read/write kernels live in
    :mod:`.mcore_ckpt` -- together they replace legacy ``MegatronArguments`` /
    ``get_mcore_model`` / ``save_mcore_checkpoint`` / ``load_mcore_checkpoint``.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        convert_config: ConvertConfig,
        *,
        template_config: Optional[TemplateConfig] = None,
        distributed_config: Optional[DistributedConfig] = None,
        checkpoint_config: Optional[CheckpointConfig] = None,
        megatron_config: Optional[MegatronConfig] = None,
        moe_config: Optional[MoEConfig] = None,
        output_dir: Optional[str] = None,
    ):
        super().__init__(model_config, output_dir=output_dir, checkpoint_config=checkpoint_config)
        self.convert_config = convert_config
        self.template_config = template_config
        self.distributed_config = distributed_config
        self.megatron_config = megatron_config
        self.moe_config = moe_config

    def validate(self) -> None:
        """Refuse an ambiguous or under-specified direction before any megatron import."""
        if self.convert_config.to_mcore and self.convert_config.to_hf:
            raise ValueError('ConvertConfig.to_mcore and to_hf are mutually exclusive; pick one direction.')
        if not self.convert_config.to_mcore and not self.convert_config.to_hf:
            raise ValueError('Set either ConvertConfig.to_mcore or ConvertConfig.to_hf.')
        if not self.model_config.model:
            raise ValueError('ModelConfig.model is required: the HF model/config is what defines the '
                             'target architecture for both conversion directions.')

    def skip(self, output_dir: str) -> bool:
        """Refuse to overwrite an existing export unless ``exist_ok`` (legacy FileExistsError guard).

        A conversion writes a fresh file set; re-running into a stale directory would mix the
        previous run's shards and sidecar files with the new ones.
        """
        if os.path.exists(output_dir) and not self.convert_config.exist_ok:
            raise FileExistsError(
                f'The output directory already exists: {output_dir}. Pass --exist_ok true to overwrite it.')
        return False

    def run(self, output_dir: str) -> str:
        """Route to the mcore-reading worker when a mcore source is involved, else HF -> mcore."""
        if self.convert_config.to_hf or self.convert_config.mcore_model or self.convert_config.mcore_adapter:
            return self._convert_mcore(output_dir)
        return self._convert_hf2mcore(output_dir)

    def _convert_hf2mcore(self, output_dir: str) -> str:
        """HF -> mcore: load the HF weights, hand them to the bridge, write a dist checkpoint."""
        from swift.dev.patch.torch_dist_save import TorchDistSaveThreadCountPatch
        from twinkle.patch import apply_patch

        hf_model, template, processor = self._load_hf_model_template(
            load_model=True, patch_offload=not self.convert_config.test_convert_precision)
        self._set_thread_count(hf_model)
        apply_patch(None, TorchDistSaveThreadCountPatch(self.convert_config.thread_count))

        strategy = self._build_strategy(processor)
        # load_weights=True hands the freshly-built mcore model straight to bridge.load_weights
        # -- the same transfer legacy performed explicitly after get_mcore_model.
        models = strategy.create_megatron_model(load_weights=True)
        logger.info('Megatron model created successfully.')
        logger.info('Successfully transferred HF model weights to MG model.')

        parallel_meta = self._parallel_meta(strategy, source_args=None)
        # SWIFT_TEST_CONVERT_PRECISION is the legacy escape hatch used by the align tests: verify
        # the conversion without paying for the (large) checkpoint write.
        if not self._skip_save():
            self._save_args(parallel_meta, output_dir)
            logger.info('Saving the model...')
            save_mcore_dist_checkpoint(models, output_dir, parallel_meta)
        # Placed last: the test runs forward passes, which can perturb the numerics being measured.
        if self.convert_config.test_convert_precision:
            self._run_precision_test(strategy, hf_model, models, template, processor)
        return output_dir

    def _convert_mcore(self, output_dir: str) -> str:
        """mcore -> HF, or mcore -> mcore (reshard / merge an mcore LoRA)."""
        from swift.dev.patch.torch_dist_save import TorchDistSaveThreadCountPatch
        from twinkle.patch import apply_patch

        # No HF weights are needed to READ an mcore checkpoint -- only the processor/config for the
        # target architecture -- so the model is not loaded here.
        _, template, processor = self._load_hf_model_template(load_model=False)

        # The source args.json carries the tuner/task settings the checkpoint was written with; they
        # must win over defaults, otherwise a LoRA checkpoint is rebuilt as a full model.
        source_args = load_source_args(self.convert_config.mcore_adapter or self.convert_config.mcore_model)
        if self.convert_config.mcore_model is not None:
            source_args['mcore_model'] = self.convert_config.mcore_model

        strategy = self._build_strategy(processor, source_args)
        models = strategy.create_megatron_model(load_weights=False)
        if self.convert_config.mcore_model is None:
            raise ValueError('Please specify `ConvertConfig.mcore_model`.')
        parallel_meta = self._parallel_meta(strategy, source_args)
        load_mcore_dist_checkpoint(models, self.convert_config.mcore_model, parallel_meta)
        if self.convert_config.mcore_adapter is not None:
            models = self._merge_lora_adapter(models, source_args, parallel_meta)
        logger.info('Megatron model created successfully.')

        if self.convert_config.to_hf:
            from swift.dev.utils import is_master

            bridge = strategy.bridge
            logger.info('Converting weights and saving the model...')
            self._save_hf_weights(bridge, models, output_dir)
            self._write_hf_export_metadata(strategy, processor, output_dir)
            if is_master():
                # Prefer the source checkpoint's args.json so the HF export records how the weights
                # were actually trained, not how this conversion was invoked.
                src = self.convert_config.mcore_adapter or self.convert_config.mcore_model
                src_args_path = os.path.join(src, 'args.json') if src else None
                if src_args_path and os.path.exists(src_args_path):
                    shutil.copy(src_args_path, os.path.join(output_dir, 'args.json'))
                else:
                    self._save_args(parallel_meta, output_dir)
            if self.convert_config.test_convert_precision:
                # Pin the model_type AND template resolved from the source model: output_dir is a
                # freshly-written dir with no hub name to match, so an unpinned re-load is ambiguous.
                resolved_model_type = processor.model_info.model_type
                resolved_template = template.template_meta.template_type
                hf_model, template, _ = self._load_hf_model_template(
                    load_model=True,
                    model=output_dir,
                    model_type=resolved_model_type,
                    template_type=resolved_template)
                self._run_precision_test(strategy, hf_model, models, template, processor)
        else:
            self._set_thread_count(models[0])
            apply_patch(None, TorchDistSaveThreadCountPatch(self.convert_config.thread_count))
            self._save_args(parallel_meta, output_dir)
            logger.info('Saving the model...')
            save_mcore_dist_checkpoint(models, output_dir, parallel_meta)
        return output_dir

    # -- mcore construction --------------------------------------------------------------------

    def _build_strategy(self, processor, source_args: Optional[Dict[str, Any]] = None) -> 'MegatronStrategy':
        """dev Configs -> a twinkle ``MegatronStrategy`` that owns mpu init and model construction.

        The strategy takes the DeviceMesh (built off DistributedConfig exactly like the training
        path) and the already-loaded HF config, so parallel sizes never travel as loose kwargs.
        Legacy's ``MegatronArguments`` defaults chain shrinks to the conversion constants below:
        no optimizer/RNG flags (a conversion writes none), no training defaults, and the
        ``use_cpu_initialization``/``recompute_granularity`` entries of legacy ``_CONVERT_KWARGS``
        disappear -- the mcore-bridge backend already builds on CPU, and legacy's 'none' ->
        ``None`` -> "not forwarded" translation was a no-op against mcore's own default.
        """
        import torch

        from swift.dev.utils.naming import resolve_megatron_attn_backend
        from twinkle.model.megatron import MegatronStrategy

        self._init_conversion_parallel()
        device_mesh = self._conversion_mesh()
        if getattr(device_mesh, 'sequence_parallel', False) and (device_mesh.tp_world_size or 1) <= 1:
            # Legacy MegatronArguments guard: TP sequence-parallelism is meaningless without TP.
            device_mesh.sequence_parallel = False

        source_args = source_args or {}
        kwargs: Dict[str, Any] = {
            # ENUM, never a string: mcore compares attention_backend by identity.
            'attention_backend': resolve_megatron_attn_backend(_CONVERT_ATTN_BACKEND),
        }
        if processor.model_info.is_moe_model:
            # Grouped GEMM is the layout the mcore MoE checkpoint is written in; without it the
            # expert weights are laid out per-expert and do not load back.
            kwargs['moe_grouped_gemm'] = True
        if processor.model_info.mcore_model_type:
            # Declared by the dev loader registry; otherwise mcore-bridge infers it from the HF
            # config's own model_type (ModelConfig.__post_init__).
            kwargs['mcore_model_type'] = processor.model_info.mcore_model_type
        task_type, num_labels = self._resolved_task(source_args)
        kwargs['task_type'] = task_type
        if num_labels is not None:
            kwargs['num_labels'] = num_labels
        if task_type not in ('causal_lm', 'generative_reranker'):
            # Legacy MegatronArguments.__post_init__: a non-generative head is always untied.
            kwargs['untie_embeddings_and_output_weights'] = True

        # MegatronConfig/MoEConfig explicit fields + megatron_extra_kwargs, rejected against the
        # ModelConfig field set (mcore-bridge's ModelConfig, which the bridge backend consumes).
        from mcore_bridge import ModelConfig

        accepted = {field.name for field in dataclasses.fields(ModelConfig)}
        for config in (self.megatron_config, self.moe_config):
            if config is None:
                continue
            explicit = getattr(config, '_explicit_fields', None)
            for field in dataclasses.fields(config):
                value = getattr(config, field.name)
                changed = field.name in explicit if explicit is not None else value != field.default
                if not changed:
                    continue
                if field.name == 'megatron_extra_kwargs':
                    extra_kwargs = value or {}
                    unknown = set(extra_kwargs) - accepted
                    if unknown:
                        raise ValueError(f'Unsupported Megatron conversion options: {sorted(unknown)}.')
                    kwargs.update(extra_kwargs)
                elif field.name in accepted:
                    kwargs[field.name] = value
                else:
                    raise ValueError(f'{type(config).__name__}.{field.name} is not supported by Megatron conversion. '
                                     'Remove it or upgrade the installed Megatron implementation.')
        # An attention_backend arriving via megatron_extra_kwargs is a string; normalize it to the
        # ENUM so mcore's identity comparison cannot silently pick a different kernel.
        if isinstance(kwargs.get('attention_backend'), str):
            kwargs['attention_backend'] = resolve_megatron_attn_backend(kwargs['attention_backend'])

        # The source checkpoint's bridge backend wins: its weights were written by that backend.
        bridge_backend = source_args.get('bridge_backend') or (
            self.distributed_config.bridge_backend if self.distributed_config is not None else 'mcore-bridge')
        torch_dtype = self._dtype(self.model_config.torch_dtype) or processor.model_info.torch_dtype
        return MegatronStrategy(
            processor.model_info.model_dir,
            device_mesh,
            mixed_precision={torch.float16: 'fp16', torch.float32: 'no'}.get(torch_dtype, 'bf16'),
            config=processor.model_info.config,
            bridge_backend=bridge_backend,
            **kwargs)

    def _init_conversion_parallel(self) -> None:
        """torch.distributed init plus the process patches the conversion path needs.

        ``mpu.initialize_model_parallel`` (inside MegatronStrategy) asserts an initialized
        process group, so single-process runs get a world-1 group here -- the same thing
        legacy's ``MegatronArguments.__post_init__ -> initialize_megatron`` did. The three
        patches are the twinkle equivalents of the process patches legacy applied when the
        ``swift.megatron`` package was imported (batched P2P routing, parallel DCP reads,
        shard-validation skip); ``MegatronModel`` applies the identical set, keeping training
        and conversion on one code path.
        """
        import torch.distributed as dist
        from twinkle import Platform, torch_util
        from twinkle.patch import apply_patch
        from twinkle.patch.megatron_p2p import MegatronBatchedP2PGroupPatch
        from twinkle.patch.torch_dcp_reader import TorchDCPParallelReaderPatch
        from twinkle.patch.torch_dcp_validation import MegatronDCPValidationPatch

        os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')  # legacy MegatronArguments.__post_init__
        if not dist.is_initialized():
            torch_util.set_device()
            dist.init_process_group(backend=Platform.device_backend())
        apply_patch(None, MegatronBatchedP2PGroupPatch())
        apply_patch(None, TorchDCPParallelReaderPatch())
        apply_patch(None, MegatronDCPValidationPatch())

    def _conversion_mesh(self):
        """DistributedConfig -> DeviceMesh; the launcher's WORLD_SIZE when the config is unset."""
        from swift.dev.builders.model import build_parallel_mesh

        if self.distributed_config is not None and (self.distributed_config.parallel_spec
                                                    or self.distributed_config.nproc_per_node is not None):
            return build_parallel_mesh(self.distributed_config)
        from twinkle import DeviceMesh

        # Direct component use without a DistributedConfig: mirror the launcher's world size so
        # the mesh cannot disagree with the process group _init_conversion_parallel just built.
        return DeviceMesh.from_sizes(world_size=int(os.environ.get('WORLD_SIZE', '1') or 1))

    def _resolved_task(self, source_args: Optional[Dict[str, Any]]) -> Tuple[str, Optional[int]]:
        """Effective task shape; a source checkpoint's declaration wins over ModelConfig."""
        source_args = source_args or {}
        task_type = source_args.get('task_type') or self.model_config.task_type or 'causal_lm'
        num_labels = source_args.get('num_labels', self.model_config.num_labels)
        if task_type == 'reranker':
            # mcore-bridge has no plain 'reranker' task; the head is seq_cls with one label --
            # the same mapping dev's training path applies.
            return 'seq_cls', num_labels or 1
        if task_type == 'seq_cls' and num_labels is None:
            raise ValueError('task_type="seq_cls" requires num_labels (from the source checkpoint or '
                             'ModelConfig) to size the classification head.')
        return task_type, num_labels

    def _parallel_meta(self, strategy: 'MegatronStrategy',
                       source_args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """The args-side metadata stored inside the dist checkpoint and mirrored into args.json.

        Key set: what the load kernel compares (tp/pp) plus everything a legacy reader
        (``load_mcore_checkpoint`` / ``MegatronArguments.load_args_config``) looks up; see
        :func:`.mcore_ckpt.load_source_args` for the reading half of the contract.
        """
        mesh = strategy.device_mesh
        vpp = getattr(mesh, 'vpp_size', None)
        task_type, num_labels = self._resolved_task(source_args)
        return {
            'tensor_model_parallel_size': mesh.tp_world_size or 1,
            'pipeline_model_parallel_size': mesh.pp_world_size or 1,
            'context_parallel_size': mesh.cp_world_size or 1,
            'expert_model_parallel_size': mesh.ep_size or 1,
            'expert_tensor_parallel_size': mesh.etp_world_size or 1,
            'virtual_pipeline_model_parallel_size': (vpp if vpp not in (None, 0, 1) else None),
            'sequence_parallel': bool(getattr(mesh, 'sequence_parallel', False)),
            # Constants of a conversion, recorded so legacy readers see the same contract.
            'no_save_optim': True,
            'no_save_rng': True,
            'no_load_optim': True,
            'no_load_rng': True,
            'finetune': True,
            'bridge_backend': strategy.backend.backend_name,
            'tuner_type': (source_args or {}).get('tuner_type', 'full'),
            'task_type': task_type,
            'num_labels': num_labels,
            'mcore_model': self.convert_config.mcore_model,
            'consumed_train_samples': 0,
        }

    # -- LoRA merge --------------------------------------------------------------------------

    def _merge_lora_adapter(self, models: List[Any], source_args: Dict[str, Any],
                            parallel_meta: Dict[str, Any]) -> List[Any]:
        """Rebuild the source checkpoint's LoRA, load its weights, merge, and unload.

        The mechanism is twinkle's (``MegatronModel.add_adapter_to_model`` unwrapped to bare
        models): peft's ``get_peft_model`` on an mcore model produces
        ``mcore_bridge.tuners.LoraParallelLinear`` layers (mcore_bridge patches peft's dispatch
        at import), the adapter weights load through the same torch_dist kernel with the peft
        key filter, and ``merge_and_unload`` folds the deltas in and removes the wrappers.
        Legacy did this with ``Swift.prepare_model`` / ``prepare_mcore_model``; the adapter
        hyper-parameters come from the source args.json either way.
        """
        from peft import get_peft_model
        from twinkle.patch.megatron_peft import MegatronPeft

        # peft-on-mcore compatibility (tied-module detection, merge guard); idempotent.
        MegatronPeft()()
        lora_config = self._lora_config_from_source(models, source_args)
        peft_models = [get_peft_model(model, lora_config, adapter_name='default') for model in models]
        # Load into the BARE models (legacy parity): peft injects the LoRA modules into the shared
        # module tree, so the wrapper and the bare model see the same parameters.
        load_mcore_dist_checkpoint(
            models, self.convert_config.mcore_adapter, parallel_meta, peft_format=True, adapter_name='default')
        logger.info('Merge LoRA...')
        return [peft_model.merge_and_unload() for peft_model in peft_models]

    def _lora_config_from_source(self, models: List[Any], source_args: Dict[str, Any]):
        """Source args.json tuner fields -> a peft ``LoraConfig`` (legacy ``prepare_adapter``)."""
        from peft import LoraConfig
        from twinkle.model.megatron import MegatronModel

        target_modules = source_args.get('target_modules', ['all-linear'])
        if source_args.get('target_regex'):
            # A regex wins over the module list (legacy MegatronTunerMixin.__post_init__).
            target_modules = [source_args['target_regex']]
        elif isinstance(target_modules, str):
            target_modules = [target_modules]
        modules_to_save = list(source_args.get('modules_to_save') or [])
        task_type, _ = self._resolved_task(source_args)
        if task_type == 'seq_cls':
            modules_to_save.append('output_layer')  # legacy get_modules_to_save
        expanded_targets = MegatronModel.get_target_modules(
            models[0],
            target_modules,
            freeze_llm=source_args.get('freeze_llm', False),
            freeze_vit=source_args.get('freeze_vit', True),
            freeze_aligner=source_args.get('freeze_aligner', True))
        modules_to_save = MegatronModel.get_modules_to_save(models[0], modules_to_save)
        return LoraConfig(
            r=source_args.get('lora_rank', 8),
            lora_alpha=source_args.get('lora_alpha', 32),
            lora_dropout=source_args.get('lora_dropout', 0.05),
            bias=source_args.get('lora_bias', 'none'),
            use_rslora=source_args.get('use_rslora', False),
            target_modules=expanded_targets,
            modules_to_save=modules_to_save,
            task_type='CAUSAL_LM')

    # -- HF export --------------------------------------------------------------------------

    def _save_hf_weights(self, bridge, models: List[Any], output_dir: str) -> None:
        """Full HF weights via the bridge backend, mirroring the training path's save call."""
        import inspect

        max_shard_size = self.checkpoint_config.max_shard_size if self.checkpoint_config else '5GB'
        save_kwargs: Dict[str, Any] = {'max_shard_size': max_shard_size}
        # Capability probe, not a version pin: only newer mcore-bridge builds accept the flag.
        # In this direction it is a no-op anyway (no HF load happened, so no source dir was
        # recorded to restore from) -- kept so the call shape matches the training path.
        if 'save_missing_weights' in inspect.signature(bridge.save_weights).parameters:
            save_kwargs['save_missing_weights'] = True
        bridge.save_weights(models, output_dir, **save_kwargs)

    def _write_hf_export_metadata(self, strategy: 'MegatronStrategy', processor, output_dir: str) -> None:
        """Master-rank non-weight half of the HF export: config, tokenizer/processor, sidecars.

        Legacy did this inside a patched ``GPTBridge.save_weights`` (``swift.megatron.init``);
        the bare bridge writes weights only, so the config/processor half lives here.
        Deliberately dropped from the legacy half: the FP8 ``FineGrainedFP8Config`` write-back
        and the remote-code ``custom_object_save`` copy -- both need a meta HF model, and an
        FP8 or remote-code mcore conversion is not a supported input on this path (mcore-bridge
        covers a fixed in-tree family set; the precision test rejects FP8 outright).
        """
        from copy import deepcopy

        import torch.distributed as dist
        from swift.dev.utils import HfConfigFactory, is_master

        if not is_master():
            return
        from swift.dev.model import save_checkpoint

        hf_config = deepcopy(processor.model_info.config)
        config = strategy.config
        if getattr(config, 'mtp_num_layers', None):
            llm_config = HfConfigFactory.get_text_config(hf_config)
            for key in ('num_nextn_predict_layers', 'mtp_num_hidden_layers'):
                if hasattr(llm_config, key):
                    setattr(llm_config, key, config.mtp_num_layers)
                    break
            else:
                llm_config.num_nextn_predict_layers = config.mtp_num_layers
        HfConfigFactory.del_config_attr(hf_config, 'quantization_config')
        hf_config.save_pretrained(output_dir)
        save_checkpoint(
            None,
            processor,
            output_dir,
            model_dirs=[processor.model_info.model_dir],
            additional_saved_files=processor.model_info.additional_saved_files)
        if dist.is_initialized():
            # The precision test re-loads this directory on every rank; the config/processor
            # files must be visible before any non-master rank reads them (legacy barrier order).
            dist.barrier()

    # -- precision test ---------------------------------------------------------------------

    def _run_precision_test(self, strategy: 'MegatronStrategy', hf_model, mg_models: List[Any], template,
                            processor) -> None:
        """Port of legacy ``swift.megatron.utils.convert_utils.test_convert_precision``.

        ``args`` reads are replaced by the objects that own the values now: parallel sizes and
        ``cp_partition_mode`` come off the DeviceMesh / DistributedConfig, task shape and
        model_type off the resolved model info, and the attention backend off the mcore config.
        The packed (padding-free) variant is dropped: conversion always runs unpacked, matching
        legacy ``_CONVERT_KWARGS['padding_free'] = False``. The template wiring uses dev's
        ``template.model`` assignment (done in ``_load_hf_model_template``) in place of legacy's
        ``register_post_encode_hook``.
        """
        import torch
        import torch.distributed as dist
        from megatron.core.tensor_parallel.mappings import (gather_from_sequence_parallel_region,
                                                            gather_from_tensor_model_parallel_region)
        from swift.dev.utils import HfConfigFactory, to_device
        from twinkle import Platform

        # One platform-resolved device per rank (GPU/NPU/MPS); the raw load has no device_map
        # default anymore, so the just-in-time move legacy got from default placement is stated
        # explicitly on both the HF and mcore sides below.
        compute_device = Platform.get_local_device()
        test_convert_dtype = self._dtype(self.convert_config.test_convert_dtype) or torch.float32
        template.set_mode('train')
        mg_model = mg_models[0]
        _test_params_sum(mg_model)

        config = mg_model.config
        is_multimodal = config.is_multimodal
        test_mm_type = getattr(config, 'test_mm_type', 'image') if is_multimodal else 'text'
        mg_language_model = mg_model.language_model if is_multimodal else mg_model
        if mg_language_model.config.fp8 is not None:
            raise ValueError('fp8 models currently do not support testing convert_precision. '
                             'Please set `--test_convert_precision false`.')
        share_embedding = mg_language_model.share_embeddings_and_output_weights
        if hf_model is not None:
            hf_model.eval()
            if dist.get_world_size() == 1:
                _test_params_sum(hf_model)
            inputs = template.encode(_get_examples(test_mm_type), return_length=True)
            hf_inputs = to_device(template.data_collator([inputs]), compute_device)
            HfConfigFactory.set_config_attr(hf_model.config, 'use_cache', False)
            ignore_modules = _precision_ignore_modules(processor, is_multimodal)
            hf_modules = _find_modules(hf_model, ignore_modules=ignore_modules)
            with torch.inference_mode(), _model_cpu_forward_context(
                    hf_modules, test_convert_dtype, compute_device, share_embedding=share_embedding):
                hf_inputs.pop('text_position_ids', None)
                hf_logits = hf_model(**hf_inputs).logits
                hf_logits = hf_logits.to(compute_device)
            hf_model.to('cpu')

        template.use_megatron = True
        inputs = template.encode(_get_examples(test_mm_type), return_length=True)
        mesh = strategy.device_mesh
        tp_size = mesh.tp_world_size or 1
        cp_size = mesh.cp_world_size or 1
        cp_partition_mode = (
            self.distributed_config.cp_partition_mode if self.distributed_config is not None else 'zigzag')
        mg_inputs = to_device(
            template.data_collator(
                [inputs],
                padding_to=_get_padding_to(
                    tp_size,
                    bool(getattr(mesh, 'sequence_parallel', False)),
                    cp_size,
                    config.attention_backend.name,
                    fp8_format=getattr(config, 'fp8', None))),
            compute_device)
        mg_model.eval()
        # thd
        mg_inputs.pop('text_position_ids', None)
        mg_language_model.config.fp8 = None  # compat fp8
        mg_modules = _find_modules(mg_language_model, ignore_modules=['visual'])
        for key in ('labels', 'seq_lens', 'attention_mask_2d'):
            mg_inputs.pop(key, None)
        mg_inputs = _get_batch_on_this_cp_rank(mg_inputs, is_multimodal=is_multimodal, task_type=config.task_type,
                                              cp_partition_mode=cp_partition_mode)
        _param = next(mg_language_model.parameters())
        mg_dtype = _param.dtype
        mg_device = _param.device
        if processor.model_info.model_type == 'minimax_m2':
            # router to bfloat16 (expert_bias). No need to do this when actually training.
            for n, m in mg_language_model.named_modules():
                if n.endswith('router'):
                    m.to(mg_dtype)
        if getattr(config, 'enable_hyper_connections', False):
            for param in mg_language_model.decoder.parameters(recurse=False):
                param.data = param.data.to(compute_device)
        attention_context = (
            _patch_attention_fp32(mg_dtype) if config.attention_backend.name in {'flash', 'fused'} else nullcontext())
        with torch.inference_mode(), _model_cpu_forward_context(
                mg_modules, test_convert_dtype, compute_device, share_embedding=share_embedding,
                target_device=mg_device), attention_context:
            mg_logits = _forward_step_helper(mg_model, mg_inputs, dtype=test_convert_dtype)
            if tp_size > 1 and config.task_type != 'seq_cls':
                if mg_logits is not None:
                    mg_logits = gather_from_tensor_model_parallel_region(mg_logits)
            if cp_size > 1:
                if mg_logits is not None:
                    mg_logits = gather_from_sequence_parallel_region(
                        mg_logits.transpose(0, 1), group=_cp_group())
                    # Contiguous CP already gathers ranks in original token order, so the
                    # zigzag un-balancing must be skipped (it is only correct for zigzag split).
                    if cp_partition_mode != 'contiguous':
                        from megatron.core.ssm.mamba_context_parallel import _undo_attention_load_balancing

                        mg_logits = _undo_attention_load_balancing(mg_logits, cp_size, None)
                    mg_logits = mg_logits.transpose(0, 1)
        mg_logits = _broadcast_mg_logits(mg_logits)
        if hf_model is None:
            return
        if config.task_type == 'seq_cls':
            mg_logits = mg_logits[:, -1]
            mean_diff = (mg_logits - hf_logits).abs().mean().item()
            max_diff = (mg_logits - hf_logits).abs().max().item()
            print(f'mean_diff: {mean_diff}, max_diff: {max_diff}')
        else:
            mg_logits = mg_logits[:, :hf_logits.shape[1]]
            token_mean_diff = (mg_logits - hf_logits).abs().mean(dim=-1)
            mean_diff = token_mean_diff.mean().item()
            max_diff = (mg_logits - hf_logits).abs().max().item()
            loss_mask = (torch.roll(hf_inputs['labels'], -1) != -100)
            mean_diff_with_loss = token_mean_diff[loss_mask].mean().item()
            max_diff_with_loss = (mg_logits - hf_logits)[loss_mask].abs().max().item()
            print(f'token_mean_diff: {token_mean_diff}')
            print(f'mean_diff: {mean_diff}, max_diff: {max_diff}')
            print(f'mean_diff (with loss): {mean_diff_with_loss}, max_diff (with loss): {max_diff_with_loss} '
                  '(Please check that mean_diff (with loss) is less than 0.1).')
            hf_tokens = hf_logits.argmax(-1)
            mg_tokens = mg_logits.argmax(-1)
            print(f'hf_tokens: {hf_tokens[0].tolist()}\nmg_tokens: {mg_tokens[0].tolist()}')
            print(f'token_diff: {(hf_tokens != mg_tokens).sum().item()}')
            print(f'token_diff (with loss): {(hf_tokens[loss_mask] != mg_tokens[loss_mask]).sum().item()}')

    # -- shared helpers -----------------------------------------------------------------------

    def _load_hf_model_template(
        self,
        *,
        load_model: bool,
        model: Optional[str] = None,
        model_type: Optional[str] = None,
        template_type: Optional[str] = None,
        patch_offload: bool = False,
    ) -> Tuple[Any, Any, Any]:
        """Load the HF model (optionally) + template, via dev's own builders.

        ``model_type``/``template_type`` pin the types resolved from the source model when
        ``model`` is a freshly-written dir: swift name-matches a hub id to registered types, but
        an arbitrary output dir falls back to config architectures / candidate templates, which
        is ambiguous for some families. dev Configs are atomic, so both values are threaded in.
        """
        from copy import copy

        from swift.dev.builders import build_template, load_model_processor
        from swift.dev.config import TemplateConfig

        model_config = self.model_config
        load_config = model_config
        override_model = model is not None and model != model_config.model
        override_type = model_type is not None and model_config.model_type is None
        if override_model or override_type:
            load_config = copy(model_config)
            if override_model:
                load_config.model = model
            if override_type:
                load_config.model_type = model_type
        build_config = self.template_config or TemplateConfig()
        if template_type is not None and build_config.template is None:
            build_config = copy(build_config)
            build_config.template = template_type
        kwargs: Dict[str, Any] = {}
        # A raw load with no device_map materializes the whole model in host RAM, and this chain
        # holds it next to the CPU-resident mcore model (use_cpu_initialization) -- twice the
        # model in RAM for HF -> mcore. 'auto' spreads the HF side over GPU/CPU/disk like legacy's
        # default placement did; an explicit model_kwargs['device_map'] wins.
        if load_model and "device_map" not in (model_config.model_kwargs or {}):
            kwargs["device_map"] = "auto"
        # patch_offload turns accelerate's align-device hook into a no-op: the bridge only READS
        # the HF weights, never runs a forward. The precision test is the exception (see caller).
        if load_model and patch_offload:
            from swift.dev.patch.accelerate_offload_hook import AccelerateSkipOffloadHookPatch
            from twinkle.patch import apply_context

            with apply_context(None, AccelerateSkipOffloadHookPatch()):
                hf_model, processor = load_model_processor(load_config, load_model=load_model, **kwargs)
        else:
            hf_model, processor = load_model_processor(load_config, load_model=load_model, **kwargs)
        template = build_template(build_config, processor)
        if hf_model is not None and getattr(template, 'use_model', False):
            template.model = hf_model
        return hf_model, template, processor

    def _set_thread_count(self, model) -> None:
        """Derive the torch-dist shard-writer thread count from the checkpoint size (legacy: ~1 per 10GB).

        Mutating convert_config here keeps a single source of truth for the value, which both the
        patch and any later log line read.
        """
        if self.convert_config.thread_count is not None:
            return
        import torch

        from swift.dev.utils import get_n_params_grads

        dtype = self._dtype(self.model_config.torch_dtype) or torch.float32
        checkpoint_size = sum(get_n_params_grads(model)[0]) * torch.finfo(dtype).bits // 8e9
        self.convert_config.thread_count = max(math.ceil(checkpoint_size / 10), 2)  # 10GB

    @staticmethod
    def _save_args(meta: Dict[str, Any], output_dir: str) -> None:
        """Write args.json next to the weights.

        The content is ``parallel_meta`` (the same dict stored inside the dist checkpoint):
        legacy wrote the whole ``MegatronArguments.__dict__``, but the only readers are
        ``load_source_args`` and legacy ``load_args_config``, which read exactly this key set.
        """
        from swift.dev.utils import is_master

        if not is_master():
            return
        import json

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'args.json')
        logger.info(f'The converted args will be saved in: {path}')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dict(meta), f, ensure_ascii=False, indent=2)

    @staticmethod
    def _dtype(value):
        """dev's string torch_dtype -> the real torch.dtype legacy Arguments/`torch.finfo` require."""
        if value is None:
            return None
        import torch

        if isinstance(value, torch.dtype):
            return value
        return getattr(torch, value)

    @staticmethod
    def _skip_save() -> bool:
        from transformers.utils import strtobool

        return bool(strtobool(os.getenv('SWIFT_TEST_CONVERT_PRECISION', '0')))


# -- precision-test helpers (module-level: no instance state, ported from legacy convert_utils) --

def _get_examples(mm_type: str) -> Dict[str, Any]:
    """One canned sample per modality for the precision test (legacy ``get_examples``)."""
    if mm_type == 'image':
        return {
            'messages': [{
                'role': 'user',
                'content': '<image>describe the image.'
            }, {
                'role':
                'assistant',
                'content':
                'The image depicts a close-up of a kitten with striking features. '
                'The kitten has a white and gray coat with distinct black stripes, '
                'particularly noticeable on its face and ears. Its eyes are large '
                'and expressive, with a captivating blue hue that stands out against '
                "the darker fur around them. The kitten's nose is small and pink, "
                'and it has long, delicate whiskers extending from either side of its mouth. '
                "The background is blurred, drawing attention to the kitten's face and "
                'making it the focal point of the image. The overall impression is '
                'one of cuteness and charm.'
            }],
            'images': ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png']
        }
    if mm_type == 'audio':
        return {
            'messages': [{
                'role': 'user',
                'content': '<audio>Caption the audio.'
            }, {
                'role': 'assistant',
                'content': "The audio contains a male voice speaking the phrase '今天天气真好呀' in Mandarin."
            }],
            'audios': ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav']
        }
    # text
    return {
        'messages': [
            {
                'role': 'user',
                'content': 'Introduction to ms-swift.'
            },
            {
                'role':
                'assistant',
                'content':
                'ms-swift is an official framework provided by the ModelScope community for fine-tuning '
                'and deploying large language models and multi-modal large models.'
            },
        ]
    }


def _to_float_dtype(data: Any, dtype: Any) -> Any:
    """Recursively cast floating tensors to ``dtype`` (legacy ``swift.utils.to_float_dtype``)."""
    import torch
    from collections.abc import Mapping

    if isinstance(data, Mapping):
        return type(data)({k: _to_float_dtype(v, dtype) for k, v in data.items()})
    if isinstance(data, (tuple, list)):
        return type(data)(_to_float_dtype(v, dtype) for v in data)
    if isinstance(data, torch.Tensor) and torch.is_floating_point(data):
        return data.to(dtype=dtype)
    return data


def _find_modules(model, recurse: bool = True, prefix: str = '', ignore_modules=None) -> List[Any]:
    """Leaf-module walk that skips ``ignore_modules`` prefixes (legacy ``_find_modules``)."""
    import torch.nn as nn

    ignore_modules = ignore_modules or []
    for k in ignore_modules:
        if prefix.startswith(k):
            return []
    else:
        named_children = list(model.named_children())

    modules = []
    for n, module in named_children:
        if module.__class__ is nn.ModuleList:
            modules += _find_modules(module, False, prefix=f'{prefix}{n}.', ignore_modules=ignore_modules)
        elif recurse:
            modules += _find_modules(module, prefix=f'{prefix}{n}.', ignore_modules=ignore_modules)
        else:
            modules.append(module)
    if not named_children:
        modules.append(model)
    return modules


def _precision_ignore_modules(processor, is_multimodal: bool) -> List[str]:
    """Vision-tower/aligner module prefixes the HF-side walk must not enter (legacy model_arch)."""
    if not is_multimodal:
        return []
    from swift.dev.model.loader import resolve_loader_cls

    loader_cls = resolve_loader_cls(processor.model_info.model_type, processor.model_info.model_dir)
    if loader_cls is None:
        return []
    arch = loader_cls(processor.model_info).model_arch
    return list(arch.vision_tower) + list(arch.aligner)


@contextmanager
def _model_cpu_forward_context(modules,
                               torch_dtype=None,
                               compute_device=None,
                               share_embedding: bool = False,
                               target_device='cpu'):
    """Keep modules on CPU except during their own forward (legacy ``_model_cpu_forward_context``).

    A pre-hook moves each module to ``compute_device``/``torch_dtype`` just-in-time and a
    post-hook moves it back, so a full-precision spot check never holds the whole model on the
    GPU. The embedding (when tied) and rotary modules stay put: their tensors are shared and
    moving them would break the views other modules hold.
    """
    import torch.nn as nn
    from megatron.core.tensor_parallel import VocabParallelEmbedding

    for module in modules:
        try:
            origin_torch_dtype = next(module.parameters()).dtype
        except StopIteration:
            pass
        else:
            break
    embeddings = None
    if share_embedding:
        embeddings = [module for module in modules if isinstance(module, (nn.Embedding, VocabParallelEmbedding))]

    def _to_cuda_hook(module, args):
        if compute_device is not None or torch_dtype is not None:
            module.to(device=compute_device, dtype=torch_dtype)
            args = _to_float_dtype(args, dtype=torch_dtype)
        return args

    def _to_cpu_hook(module, args, output):
        if share_embedding and module in embeddings or 'rotaryemb' in module.__class__.__name__.lower():
            return
        module.to(device=target_device, dtype=origin_torch_dtype)

    hooks = []
    for module in modules:
        hooks.append(module.register_forward_pre_hook(_to_cuda_hook))
        hooks.append(module.register_forward_hook(_to_cpu_hook))
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


def _test_params_sum(model) -> None:
    """Sanity-sum every parameter: zero / NaN / inf rows are logged before any diff is trusted."""
    import torch
    from megatron.core import mpu
    from twinkle import Platform

    compute_device = Platform.get_local_device()
    total_sum = 0
    zero_count = 0
    n_parameter = 0
    for n, p in model.named_parameters():
        n_parameter += 1
        sum_ = p.to(device=compute_device, dtype=torch.float32).abs().sum().cpu().item()
        if sum_ == 0 and '.lora_B.' not in n:
            zero_count += 1
            logger.warning(f'n: {n}, sum: {sum_}')
        elif math.isnan(sum_) or math.isinf(sum_) or sum_ > 1e10:
            logger.warning(f'n: {n}, sum: {sum_}')
        else:
            total_sum += sum_
    cond = mpu.get_data_parallel_rank() == 0
    logger.info_if(f'n_parameter: {n_parameter}', cond=cond)
    logger.info_if(f'total_sum: {total_sum}', cond=cond)
    logger.info_if(f'zero_count: {zero_count}', cond=cond)


def _broadcast_mg_logits(mg_logits=None, src_rank=None):
    """Bring the last pipeline stage's logits to every rank (legacy ``broadcast_mg_logits``)."""
    import torch
    import torch.distributed as dist
    from twinkle import Platform

    if not dist.is_initialized():
        return
    rank = dist.get_rank()
    if src_rank is None:
        src_rank = dist.get_world_size() - 1
    if rank == src_rank:
        meta = [tuple(mg_logits.shape), str(mg_logits.dtype).split('.', 1)[1]]
    else:
        meta = [None, None]

    dist.broadcast_object_list(meta, src=src_rank)
    shape, dtype = meta
    dtype = getattr(torch, dtype)

    if rank != src_rank:
        mg_logits = torch.empty(shape, dtype=dtype, device=Platform.get_local_device())

    dist.broadcast(mg_logits, src=src_rank)

    return mg_logits


@contextmanager
def _patch_attention_fp32(compute_dtype):
    """Run TE attention in fp32 and cast back (legacy ``_patch_attention_fp32``)."""
    from megatron.core.extensions.transformer_engine import TEDotProductAttention

    forward = TEDotProductAttention.forward

    def new_forward(self, query_layer, key_layer, value_layer, *args, **kwargs):
        torch_dtype = query_layer.dtype
        query_layer = query_layer.to(compute_dtype)
        key_layer = key_layer.to(compute_dtype)
        value_layer = value_layer.to(compute_dtype)
        res = forward(self, query_layer, key_layer, value_layer, *args, **kwargs)
        res = res.to(dtype=torch_dtype)
        return res

    TEDotProductAttention.forward = new_forward
    try:
        yield
    finally:
        TEDotProductAttention.forward = forward


def _forward_step_helper(model, inputs, dtype=None):
    """Single mcore forward across pipeline stages (legacy ``forward_step_helper``)."""
    import torch
    from megatron.core import mpu
    from megatron.core.inference.communication_utils import (recv_from_prev_pipeline_rank_,
                                                             send_to_next_pipeline_rank_)

    config = model.config
    dtype = dtype or config.params_dtype
    if not mpu.is_pipeline_first_stage():
        recv_shape_buffer = torch.empty((3, ), device=torch.cuda.current_device(), dtype=torch.int64)
        recv_from_prev_pipeline_rank_(recv_shape_buffer)
        recv_buffer = torch.empty(recv_shape_buffer.tolist(), device=torch.cuda.current_device(), dtype=dtype)
        recv_from_prev_pipeline_rank_(recv_buffer)
        model.set_input_tensor(recv_buffer)
    output_tensor = model(**inputs)
    if not mpu.is_pipeline_last_stage():
        recv_shape_buffer = torch.tensor(output_tensor.shape, device=torch.cuda.current_device(), dtype=torch.int64)
        send_to_next_pipeline_rank_(recv_shape_buffer)
        send_to_next_pipeline_rank_(output_tensor)
        output_tensor = None

    return output_tensor


def _get_padding_to(tp_size: int, sequence_parallel: bool, cp_size: int, attention_backend_name: str,
                    fp8_format=None, fp4_format=None, fp8_recipe: Optional[str] = None) -> Optional[int]:
    """Sequence-length alignment the kernels require (legacy ``get_padding_to``, args -> params)."""
    padding_to = None
    if tp_size > 1 and sequence_parallel:
        padding_to = tp_size
    if cp_size > 1:
        padding_to = (padding_to or 1) * cp_size
    origin_padding_to = padding_to
    if fp8_recipe == 'blockwise':
        padding_to = (padding_to or 1) * 128
    elif fp8_recipe == 'mxfp8':
        # MXFP8 uses a block size of 32. Under sequence parallel, the sequence is split across
        # TP ranks, so each per-rank shard (seq_len / TP) must itself be divisible by 32. Pad the
        # total length to TP * 32 to guarantee this.
        padding_to = (padding_to or 1) * 32
    elif fp8_format is not None or fp4_format is not None:
        padding_to = (padding_to or 1) * 16
    if attention_backend_name == 'fused':
        padding_to = max(padding_to or 1, ((origin_padding_to) or 1) * 64)
    return padding_to


def _get_batch_on_this_cp_rank(batch: Dict[str, Any], *, is_multimodal: bool, task_type: Optional[str],
                               cp_partition_mode: str = 'zigzag'):
    """Slice the batch into this CP rank's chunks (legacy ``get_batch_on_this_cp_rank``).

    ``cp_partition_mode`` selects zigzag (load-balanced, the default) vs contiguous chunking;
    packed-sequence inputs never occur here because conversion runs unpacked.
    """
    from mcore_bridge import split_cp_inputs
    from megatron.core import mpu
    from transformers.utils import is_torch_npu_available

    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        keys = ['labels', 'position_ids', 'loss_scale']
        if not is_multimodal:
            # Multimodal models will handle CP in input_embeds.
            keys.append('input_ids')
        kwargs = {}
        if cp_partition_mode == 'contiguous':
            kwargs['cp_partition_mode'] = 'contiguous'
        for key, val in batch.items():
            if key not in keys:
                continue
            if task_type in ('seq_cls', 'embedding', 'generative_reranker') and key == 'labels':
                continue
            if val is not None:
                batch[key] = split_cp_inputs(val, None, -1, **kwargs)
        attention_mask = batch.get('attention_mask')
        if is_torch_npu_available() and attention_mask is not None and attention_mask.ndim >= 4:
            batch['attention_mask'] = split_cp_inputs(attention_mask, None, -2, **kwargs)

    return batch


def _cp_group():
    from megatron.core import mpu

    return mpu.get_context_parallel_group()
