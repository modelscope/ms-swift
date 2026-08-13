from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

from transformers import PreTrainedModel

from twinkle import Platform, remote_function
from twinkle.data_format import InputFeature, Trajectory
from twinkle.hub import HubOperation
from twinkle.infra import collect_tensor_dict
from twinkle.model.transformers.transformers import TransformersModel as TwinkleTransformersModel
from twinkle.model.transformers.transformers import _default_adapter_name
from twinkle.processor import InputProcessor
from twinkle.utils import get_logger

logger = get_logger()

# Feature keys a sentence-transformers ``Transformer`` module consumes from the encoded batch.
# Everything else (labels, dataset-specific fields) is dropped before the ST forward.
_ST_FEATURE_KEYS = ('input_ids', 'attention_mask', 'token_type_ids', 'position_ids')


class SentenceTransformerModel(TwinkleTransformersModel):
    """Train a ``sentence-transformers`` model on twinkle's embedding pipeline.

    Why a dedicated class instead of the plain embedding task:
      - The HF embedding path (``task='embedding'``) keeps a causal-LM backbone, swaps ``lm_head``
        for identity, extracts per-token ``features`` and pools them in
        ``InputProcessor.postprocess_tensor_sp``. A ``SentenceTransformer`` is instead a *pipeline*
        of modules (``Transformer -> Pooling -> Normalize`` and optionally ``Router``/``Dense``) that
        already performs its own pooling and normalization, producing ``features['sentence_embedding']``
        of shape ``[B, D]``. Re-pooling it would be wrong, so this model bypasses the per-token pooling.
      - So the only thing that differs from the base transformers model is *model construction* and the
        *forward output* (write ``outputs['embeddings']`` directly). Optimizer / scheduler / loss /
        metric / step / grad-clip are inherited unchanged.

    Contract kept identical to the HF embedding path so the rest of the stack is reused verbatim:
      - the recipe wires the embedding ``InputProcessor`` (produces ``input_ids``/``attention_mask`` for
        the flattened anchor+positive+negative batch) and an embedding loss (e.g. ``InfoNCE``) via
        ``set_processor`` / ``set_loss``;
      - forward writes the L2-normalized ``outputs['embeddings']`` ``[B, D]`` that the loss reads.

    Scope: targets the standard (non-sequence-parallel, non-packed) embedding-training path, which is
    how sentence-transformers models are trained. Sequence parallel and padding-free packing are not
    composed with a whole-sentence pooling head, so they are explicitly rejected rather than silently
    mis-pooled.
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
                 **kwargs):
        # Mirror TwinkleTransformersModel.__init__ plumbing, but build a SentenceTransformer instead of
        # an AutoModelFor* backbone. super(PreTrainedModel, self) intentionally skips PreTrainedModel's
        # config-requiring __init__ (same as the parent) and initializes the remaining mixins.
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        self._try_init_process_group()
        super(PreTrainedModel, self).__init__()
        self._default_tokenizer = None
        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision
        self._fsdp_config = dict(fsdp_config or {})
        self._ddp_config = ddp_config or {}
        self._memory_efficient_init = memory_efficient_init
        # Router replay is a MoE-causal-LM concept; irrelevant to sentence encoders.
        self._router_replay_enabled = False
        self._router_replay_applied = False
        self._decide_strategy(strategy)
        self.grad_scaler_config = grad_scaler_config
        if model_id is not None:
            model_id = HubOperation.download_model(model_id)
        self.model_id = model_id
        self.tokenizer_id = kwargs.get('tokenizer_id', self.model_id)

        self.model = self._build_sentence_transformer(model_id, sentence_transformer_kwargs or {})
        # hf_config is read by the processor/loss for e.g. pad-token / dtype; expose the backbone's.
        self.hf_config = getattr(config, 'name_or_path', None) and config or self._backbone_config()
        self._enable_gradient_checkpointing()

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

    def _enable_gradient_checkpointing(self):
        # SentenceTransformer has no top-level gradient_checkpointing_enable; drive the backbone.
        auto_model = getattr(self._first_module(), 'auto_model', None)
        if auto_model is not None and hasattr(auto_model, 'gradient_checkpointing_enable'):
            auto_model.gradient_checkpointing_enable()

    # --- forward --------------------------------------------------------------

    def _encode_sentence_embeddings(self, inputs: Dict[str, Any]):
        """Run the ST module pipeline and return ``[B, D]`` sentence embeddings.

        The ST ``Transformer`` module reads ``input_ids``/``attention_mask`` from the features dict and
        the pipeline writes ``sentence_embedding``; pooling + normalization happen inside ST.
        """
        if self.sp_strategy is not None:
            raise NotImplementedError('SentenceTransformerModel does not support sequence parallelism: '
                                      'a whole-sentence pooling head cannot see a sequence shard.')
        position_ids = inputs.get('position_ids')
        if position_ids is not None and self._is_packed_position_ids(position_ids):
            raise NotImplementedError('SentenceTransformerModel does not support padding-free packing: '
                                      'the pooling head expects one sequence per batch row.')
        features = {k: inputs[k] for k in _ST_FEATURE_KEYS if k in inputs}
        out_features = self.model(features)
        return out_features['sentence_embedding']

    @staticmethod
    def _is_packed_position_ids(position_ids) -> bool:
        # A packed [1, total] batch restarts position ids at 0 for each concatenated sequence.
        flat = position_ids.squeeze(0) if position_ids.dim() == 2 else position_ids
        return bool((flat[1:] == 0).any()) if flat.numel() > 1 else False

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

    def save(self, name: Optional[str] = None, output_dir: Optional[str] = None, interval: int = 1, **kwargs):
        """Save in native sentence-transformers format (a directory loadable by ``SentenceTransformer``).

        Only the model is written (rank-0), matching how sentence-transformers checkpoints are consumed;
        optimizer/scheduler resume is out of scope for this encoder recipe.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        if name is None:
            name = f'checkpoint-step-{optimizer_config.cur_step}'
        if output_dir is None:
            output_dir = 'output'
        checkpoint_dir = os.path.join(output_dir, name)
        if interval and optimizer_config.cur_step % interval != 0:
            return checkpoint_dir
        if Platform.get_rank() <= 0:
            unwrap = getattr(self.strategy, 'unwrap_model', None)
            st_model = unwrap(self.model) if callable(unwrap) else self.model
            os.makedirs(checkpoint_dir, exist_ok=True)
            st_model.save(checkpoint_dir)
            logger.info(f'Saved SentenceTransformer checkpoint to: {checkpoint_dir}')
        return checkpoint_dir
