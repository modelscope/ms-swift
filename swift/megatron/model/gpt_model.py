# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, Literal, Optional, Tuple

import torch
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt import GPTModel as McoreGPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import WrappedTensor, deprecate_inference_params
from megatron.training import get_args

from swift.utils import get_logger
from .rope import dynamic_rope_update, get_rope_inv_freq

logger = get_logger()


class OutputLayerLinear(TELinear):

    def forward(self, hidden_states, *args, **kwargs):
        args = get_args()
        if args.sequence_parallel and args.tensor_model_parallel_size > 1:
            hidden_states = gather_from_sequence_parallel_region(hidden_states)
        return super().forward(hidden_states)

    def sharded_state_dict(
            self,
            prefix: str = '',
            sharded_offsets: Tuple[Tuple[int, int, int]] = (),
            metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        res = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        for k, v in res.items():
            if k.endswith('._extra_state'):
                if v.data is not None and v.data.numel() == 0:
                    v.data = None
        return res


class GPTModel(McoreGPTModel):

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal['learned_absolute', 'rope', 'mrope', 'none'] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        hf_rope_scaling: Dict[str, Any] = None,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        vp_stage: Optional[int] = None,
    ):
        if config.multi_latent_attention and config.rope_type == 'yarn':
            config.rope_type = 'rope'  # use transformers implementation
            if hf_rope_scaling and hf_rope_scaling['rope_type'] == 'yarn':
                # softmax_scale
                config.mscale = hf_rope_scaling['mscale']
                config.mscale_all_dim = hf_rope_scaling['mscale_all_dim']
                config.rotary_scaling_factor = hf_rope_scaling['factor']
        self.hf_rope_scaling = hf_rope_scaling
        super().__init__(
            config,
            transformer_layer_spec,
            vocab_size,
            max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )
        if config.multi_latent_attention:
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=config.qk_pos_emb_head_dim,
                rotary_percent=rotary_percent,
                rotary_interleaved=config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                rope_scaling=rope_scaling,
                rope_scaling_factor=rope_scaling_factor,
                use_cpu_initialization=config.use_cpu_initialization,
            )
            # save memory
            for i in range(len(self.decoder.layers)):
                if hasattr(self.decoder.layers[i].self_attention, 'rotary_pos_emb'):
                    del self.decoder.layers[i].self_attention.rotary_pos_emb
        self.attention_scaling = 1.
        new_inv_freq, self.attention_scaling = get_rope_inv_freq()
        self.rotary_pos_emb.inv_freq = new_inv_freq.to(self.rotary_pos_emb.inv_freq.device)
        args = get_args()
        if args.task_type == 'seq_cls' and self.post_process:
            self.output_layer = OutputLayerLinear(
                config.hidden_size,
                args.num_labels,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                parallel_mode=None,
                skip_weight_param_allocation=False,
            )

        if (self.attention_scaling != 1 or position_embedding_type == 'mrope') and config.apply_rope_fusion:
            config.apply_rope_fusion = False
            logger.warning('`apply_rope_fusion` does not support `attention_scaling`. '
                           f'Setting `config.apply_rope_fusion`: {config.apply_rope_fusion}')

    @contextmanager
    def _patch_apply_rotary_pos_emb(self):
        if self.attention_scaling == 1.:
            yield
            return

        from megatron.core.transformer import attention
        origin_apply_rotary_pos_emb = attention.apply_rotary_pos_emb

        def apply_rotary_pos_emb(*args, **kwargs):
            kwargs['mscale'] = self.attention_scaling
            return origin_apply_rotary_pos_emb(*args, **kwargs)

        attention.apply_rotary_pos_emb = apply_rotary_pos_emb
        try:
            yield
        finally:
            attention.apply_rotary_pos_emb = origin_apply_rotary_pos_emb

    def _preprocess(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        decoder_input: torch.Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Preprocesses inputs for the transformer decoder.

        Applies embeddings to input tokens, or uses `decoder_input` from a previous
        pipeline stage. Also sets up rotary positional embeddings.
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.
        in_inference_mode = inference_context is not None and not self.training

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        if decoder_input is not None and self.training and torch.is_grad_enabled() and not decoder_input.requires_grad:
            # fix LoRA incompatibility with gradient checkpointing
            decoder_input = decoder_input.requires_grad_(True)

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        if self.position_embedding_type in {'rope', 'mrope'}:
            if not self.training and self.config.flash_decode and inference_context:
                assert (inference_context.is_static_batching()
                        ), 'GPTModel currently only supports static inference batching.'
                # Flash decoding uses precomputed cos and sin for RoPE
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb_cache.setdefault(
                    inference_context.max_sequence_length,
                    self.rotary_pos_emb.get_cos_sin(inference_context.max_sequence_length),
                )
            else:
                rotary_seq_len = RotaryEmbedding.get_rotary_seq_len(self, inference_context, self.decoder,
                                                                    decoder_input, self.config, packed_seq_params)
                if self.hf_rope_scaling is not None:
                    attention_scaling = dynamic_rope_update(self, self.rotary_pos_emb.inv_freq, rotary_seq_len)
                    if attention_scaling is not None:
                        self.attention_scaling = attention_scaling
                packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
                if self.position_embedding_type == 'mrope':
                    rotary_pos_emb = self.rotary_pos_emb(
                        position_ids,
                        mrope_section=self.mrope_section,
                        packed_seq=packed_seq,
                    )
                else:
                    rotary_pos_emb = self.rotary_pos_emb(
                        rotary_seq_len,
                        packed_seq=packed_seq,
                    )
                    if packed_seq and not self.config.apply_rope_fusion:
                        assert position_ids.shape[0] == 1, f'position_ids.shape: {position_ids.shape}'
                        rotary_pos_emb = rotary_pos_emb[position_ids[0]]

        if (in_inference_mode and ((self.config.enable_cuda_graph and self.config.cuda_graph_scope != 'full_iteration')
                                   or self.config.flash_decode) and rotary_pos_cos is not None
                and inference_context.is_static_batching()):
            current_batch_size = input_ids.shape[0]
            sequence_len_offset = torch.tensor(
                [inference_context.sequence_len_offset] * current_batch_size,
                dtype=torch.int32,
                device=rotary_pos_cos.device,  # Co-locate this with the rotary tensors
            )
        else:
            sequence_len_offset = None

        # Wrap decoder_input to allow the decoder (TransformerBlock) to delete the
        # reference held by this caller function, enabling early garbage collection for
        # inference. Skip wrapping if decoder_input is logged after decoder completion.
        if in_inference_mode and not has_config_logger_enabled(self.config):
            decoder_input = WrappedTensor(decoder_input)

        return decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset

    # Code borrowed from NVIDIA/Megatron-LM
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        decoder_input: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset = (
            self._preprocess(
                input_ids=input_ids,
                position_ids=position_ids,
                decoder_input=decoder_input,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
            ))
        # Run decoder.
        with self._patch_apply_rotary_pos_emb():
            hidden_states = self.decoder(
                hidden_states=decoder_input,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                **(extra_block_kwargs or {}),
                **kwargs,
            )

        args = get_args()
        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels if args.task_type == 'causal_lm' else None,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
        )

    def get_input_tensor(self):
        return self.decoder.input_tensor
