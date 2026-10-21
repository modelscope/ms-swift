# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

from transformers import PreTrainedModel

from twinkle import Platform, remote_function
from twinkle.data_format import InputFeature, Trajectory
from twinkle.hub import HubOperation
from twinkle.infra import collect_tensor_dict, get_device_mesh
from twinkle.model.transformers.transformers import TransformersModel as TwinkleTransformersModel
from twinkle.model.transformers.transformers import _default_adapter_name
from twinkle.processor import InputProcessor
from twinkle.utils import get_logger

logger = get_logger()

# Feature keys a sentence-transformers ``Transformer`` module consumes from the encoded batch
# (mirrors legacy's trans_features filter). Everything else is dropped before the ST forward.
_ST_FEATURE_KEYS = ('input_ids', 'attention_mask', 'token_type_ids', 'position_ids', 'inputs_embeds')


class SentenceTransformerModel(TwinkleTransformersModel):
    """Train a ``sentence-transformers`` model on twinkle's embedding pipeline.

    A ``SentenceTransformer`` is a *pipeline* (Transformer -> Pooling -> Normalize) that already
    pools and normalizes, producing ``sentence_embedding`` ``[B, D]`` -- re-pooling would be wrong,
    so this bypasses the per-token pooling of the plain ``task='embedding'`` path. The recipe wires
    the embedding InputProcessor and loss; packing unpacks features per-sequence before the pipeline.
    """

    def __init__(self,
                 model_id: Optional[str] = None,
                 *,
                 config: Any = None,
                 device_mesh: Any = None,
                 mixed_precision: str = 'bf16',
                 strategy: str = 'accelerate',
                 ddp_config: Optional[Dict[str, Any]] = None,
                 fsdp_config: Optional[Dict[str, Any]] = None,
                 grad_scaler_config: Optional[Dict[str, Any]] = None,
                 memory_efficient_init: bool = False,
                 sentence_transformer_kwargs: Optional[Dict[str, Any]] = None,
                 gradient_checkpointing: bool = True,
                 gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None,
                 init_strategy: Optional[str] = None,
                 acc_strategy: str = 'token',
                 model_arch: Optional[Dict[str, Any]] = None,
                 **kwargs):
        # Mirror TwinkleTransformersModel.__init__ plumbing, but build a SentenceTransformer; the
        # super(PreTrainedModel, self) call skips its config-requiring __init__ (same as the parent).
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        self._try_init_process_group()
        super(PreTrainedModel, self).__init__()
        self._default_tokenizer = None
        # Overrides the @remote_class-wrapped base, so no automatic device_mesh injection; resolve the
        # global default to uphold the "a constructed model holds a real mesh" invariant.
        self.device_mesh = device_mesh if device_mesh is not None else get_device_mesh()
        self.mixed_precision = mixed_precision
        self._fsdp_config = dict(fsdp_config or {})
        self._ddp_config = ddp_config or {}
        self._memory_efficient_init = memory_efficient_init
        # Router replay is a MoE-causal-LM concept; irrelevant to sentence encoders.
        self._router_replay_enabled = False
        self._router_replay_applied = False
        # ``_acc_strategy`` / ``model_arch`` are read by inherited TransformersModel code (the
        # optimizer-group and metric paths) but this forked __init__ skips the base one that sets
        # them -- same gap the multi-LoRA variant closes, with the base's role normalization.
        self._acc_strategy = acc_strategy
        self.model_arch = dict(model_arch or {})
        for _role in ('language_model', 'vision_tower', 'aligner'):
            _value = self.model_arch.get(_role)
            self.model_arch[_role] = [_value] if isinstance(_value, str) else list(_value or [])
        self._decide_strategy(strategy)
        self.grad_scaler_config = grad_scaler_config
        if model_id is not None:
            model_id = HubOperation.download_model(model_id)
        self.model_id = model_id
        self.tokenizer_id = kwargs.get('tokenizer_id', self.model_id)

        self.model = self._build_sentence_transformer(model_id, sentence_transformer_kwargs or {})
        # hf_config is read by the processor/loss for e.g. pad-token / dtype; expose the backbone's.
        self.hf_config = getattr(config, 'name_or_path', None) and config or self._backbone_config()
        # Gated worker-side: a driver-side post-construction disable would only touch the Ray
        # proxy stub and silently no-op.
        if gradient_checkpointing:
            self._enable_gradient_checkpointing(gradient_checkpointing_kwargs)
        # Re-initialize weights the checkpoint left uninitialized (inherited method); ST's pipeline
        # is an nn.Module so parameters() reaches the backbone. new_special_tokens is builder-rejected.
        if init_strategy is not None:
            self._apply_init_strategy(init_strategy)

        self.sp_strategy = None
        self._model_wrapped = False
        self.optimizer_group = {_default_adapter_name: self._construct_default_optimizer_group()}
        self.optimizer_group[_default_adapter_name].adapter_name = _default_adapter_name
        self.active_group = _default_adapter_name

    # --- construction helpers -------------------------------------------------

    def _build_sentence_transformer(self, model_id: Optional[str], st_kwargs: Dict[str, Any]):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError('SentenceTransformerModel requires the `sentence-transformers` package. '
                              'Install it with `pip install sentence-transformers`.') from e
        if model_id is None:
            raise ValueError('SentenceTransformerModel requires `model_id`; a sentence-transformers '
                             'model is loaded via SentenceTransformer(model_id), not built from config.')
        st_kwargs.setdefault('trust_remote_code', True)
        return SentenceTransformer(model_id, **st_kwargs)

    def _first_module(self):
        """The leading ``Transformer`` module that owns the HF backbone (``auto_model``)."""
        return self.model[0]

    def _backbone_config(self):
        auto_model = getattr(self._first_module(), 'auto_model', None)
        return getattr(auto_model, 'config', None)

    def _enable_gradient_checkpointing(self, gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None):
        # SentenceTransformer has no top-level gradient_checkpointing_enable; drive the backbone.
        auto_model = getattr(self._first_module(), 'auto_model', None)
        if auto_model is not None and hasattr(auto_model, 'gradient_checkpointing_enable'):
            auto_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    # --- forward --------------------------------------------------------------

    def _encode_sentence_embeddings(self, inputs: Dict[str, Any]):
        """Run the ST module pipeline and return ``[B, D]`` sentence embeddings.

        The ST ``Transformer`` module reads ``input_ids``/``attention_mask`` from the features dict and
        the pipeline writes ``sentence_embedding``; pooling + normalization happen inside ST. Under
        padding_free packing the batch is ``[1, total_tokens]`` -- the ST Pooling module is not
        packed-aware, so unpack the features into ``[n_seqs, max_len]`` first (see below).
        """
        if self.sp_strategy is not None:
            raise NotImplementedError('SentenceTransformerModel does not support sequence parallelism: '
                                      'a whole-sentence pooling head cannot see a sequence shard.')
        features = {k: inputs[k] for k in _ST_FEATURE_KEYS if k in inputs}
        position_ids = inputs.get('position_ids')
        if InputProcessor._is_packed_position_ids(position_ids):
            features = self._unpack_packed_features(features, position_ids)
        out_features = self.model(features)
        return out_features['sentence_embedding']

    @staticmethod
    def _unpack_packed_features(features: Dict[str, Any], position_ids) -> Dict[str, Any]:
        """Split packed ``[1, total]`` features into ``[n_seqs, max_len]`` for the ST pipeline."""
        import torch
        keys_to_unpack = [k for k in ('input_ids', 'attention_mask', 'token_type_ids') if k in features]
        tensors = [features[k] for k in keys_to_unpack]
        # Pad value 0 for all: input_ids pad=0 is fine (attention_mask=0 marks it), position_ids unused after.
        unpacked = InputProcessor._unpack_by_position_ids(position_ids, *tensors, padding_values=[0] * len(tensors))
        result = dict(features)
        for key, tensor in zip(keys_to_unpack, unpacked):
            result[key] = tensor
        # Synthesize correct attention_mask from unpacked input_ids (non-zero = valid).
        if 'attention_mask' not in result and 'input_ids' in result:
            result['attention_mask'] = (result['input_ids'] != 0).long()
        # Drop position_ids: ST's Transformer module derives them internally from attention_mask.
        result.pop('position_ids', None)
        return result

    def _prepare_forward(self, inputs, adapter_name):
        """Shared front half of forward/forward_only: wrap, encode inputs, split off labels."""
        optimizer_config = self.optimizer_group[adapter_name]
        self._lazy_wrap_model()
        if not inputs:
            raise ValueError('inputs empty, check your DataLoader outputs')
        if (isinstance(inputs, dict) and self._not_encoded(inputs)) or (isinstance(inputs, list)
                                                                        and self._not_encoded(inputs[0])):
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            if isinstance(inputs, dict):
                inputs = [inputs]
            inputs = optimizer_config.template.batch_encode(inputs)  # noqa
        processor: InputProcessor = optimizer_config.processor
        assert isinstance(processor, InputProcessor), 'Set a correct `InputProcessor` before forwarding'
        inputs = processor(inputs, sp_strategy=self.sp_strategy, model=self.model, hf_config=self.hf_config)
        return optimizer_config, inputs

    def forward(self, *, inputs: Union[InputFeature, List[InputFeature], List[Trajectory]], **kwargs):
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        kwargs.pop('task', None)  # a sentence encoder is always the embedding task
        self.model.train()
        optimizer_config, inputs = self._prepare_forward(inputs, adapter_name)
        labels = inputs.pop('labels', None)
        optimizer_config.accumulate_metrics(True)

        embeddings = self._encode_sentence_embeddings(inputs)

        inputs['labels'] = labels
        outputs = {'embeddings': embeddings, 'logits': None, 'past_key_values': None}
        optimizer_config.train_status.inputs = inputs
        optimizer_config.train_status.outputs = outputs
        optimizer_config.train_status.forward_kwargs = kwargs
        optimizer_config.train_status.loss_value = outputs.get('aux_loss', 0)
        return {'embeddings': embeddings}

    @remote_function(dispatch='slice_dp', collect=collect_tensor_dict)
    def forward_only(self, *, inputs: Union[InputFeature, List[InputFeature], List[Trajectory]], **kwargs):
        import torch
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        kwargs.pop('task', None)
        self.model.eval()
        optimizer_config, inputs = self._prepare_forward(inputs, adapter_name)
        inputs.pop('labels', None)
        with torch.inference_mode():
            embeddings = self._encode_sentence_embeddings(inputs)
        return {'embeddings': embeddings}

    # --- persistence ----------------------------------------------------------

    # Decorated like the base save (collect='first' returns the rank-0 path; lazy_collect=False blocks
    # through staging): an undecorated override runs on the driver's proxy stub in ray mode and dies on
    # the missing instance state before the async half is ever reached.
    @remote_function(collect='first', lazy_collect=False)
    def save(self, name: Optional[str] = None, output_dir: Optional[str] = None, interval: int = 1, **kwargs):
        """Save in native sentence-transformers format (a directory loadable by ``SentenceTransformer``).

        Only the model is written (rank-0), matching how sentence-transformers checkpoints are consumed;
        optimizer/scheduler resume is out of scope for this encoder recipe. Under ``async_save`` the
        directory write rides the background writer (the bytes are frozen at call time).
        """
        from twinkle.model.base import rotate_checkpoints
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        save_total_limit = kwargs.pop('save_total_limit', None)
        async_save = kwargs.pop('async_save', False)
        optimizer_config = self.optimizer_group[adapter_name]
        if name is None:
            name = f'checkpoint-step-{optimizer_config.cur_step}'
        if output_dir is None:
            output_dir = 'output'
        checkpoint_dir = os.path.join(output_dir, name)
        if interval and optimizer_config.cur_step % interval != 0:
            return checkpoint_dir
        unwrap = getattr(self.strategy, 'unwrap_model', None)
        st_model = unwrap(self.model) if callable(unwrap) else self.model

        def _write():
            if Platform.get_rank() <= 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                st_model.save(checkpoint_dir)
                logger.info(f'Saved SentenceTransformer checkpoint to: {checkpoint_dir}')

        if async_save:
            self.submit_async_write(lambda: (_write(), rotate_checkpoints(output_dir, checkpoint_dir,
                                                                          save_total_limit)))
        else:
            _write()
            rotate_checkpoints(output_dir, checkpoint_dir, save_total_limit)
        return checkpoint_dir
