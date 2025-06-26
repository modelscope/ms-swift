# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, Literal, Optional

import torch
from megatron.core import InferenceParams
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt import GPTModel as McoreGPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from swift.utils import get_logger
from .rope import dynamic_rope_update, get_rope_inv_freq

logger = get_logger()


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
    ):
        if config.multi_latent_attention and config.rope_type == 'yarn':
            config.rope_type = 'rope'  # use transformers implementation
            if hf_rope_scaling and hf_rope_scaling['rope_type'] == 'yarn':
                # softmax_scale
                config.mscale = hf_rope_scaling['mscale']
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
            for i in range(config.num_layers):
                if hasattr(self.decoder.layers[i].self_attention, 'rotary_pos_emb'):
                    del self.decoder.layers[i].self_attention.rotary_pos_emb
        self.attention_scaling = 1.
        if self.hf_rope_scaling is not None:
            new_inv_freq, self.attention_scaling = get_rope_inv_freq()
            self.rotary_pos_emb.inv_freq.data.copy_(new_inv_freq)
        if self.attention_scaling != 1 and config.apply_rope_fusion:
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

    # Code borrowed from NVIDIA/Megatron-LM
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
    ) -> torch.Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        if self.position_embedding_type == 'rope':
            if not self.training and self.config.flash_decode and inference_params:
                # Flash decoding uses precomputed cos and sin for RoPE
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb_cache.setdefault(
                    inference_params.max_sequence_length,
                    self.rotary_pos_emb.get_cos_sin(inference_params.max_sequence_length),
                )
            else:
                rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(inference_params, self.decoder, decoder_input,
                                                                        self.config, packed_seq_params)
                if self.hf_rope_scaling is not None:
                    attention_scaling = dynamic_rope_update(self, self.rotary_pos_emb.inv_freq, rotary_seq_len)
                    if attention_scaling is not None:
                        self.attention_scaling = attention_scaling
                rotary_pos_emb = self.rotary_pos_emb(
                    rotary_seq_len,
                    packed_seq=packed_seq_params is not None and packed_seq_params.qkv_format == 'thd',
                )
        if ((self.config.enable_cuda_graph or self.config.flash_decode) and rotary_pos_cos is not None
                and inference_params):
            sequence_len_offset = torch.tensor(
                [inference_params.sequence_len_offset] * inference_params.current_batch_size,
                dtype=torch.int32,
                device=rotary_pos_cos.device,  # Co-locate this with the rotary tensors
            )
        else:
            sequence_len_offset = None

        # Run decoder.
        with self._patch_apply_rotary_pos_emb():
            hidden_states = self.decoder(
                hidden_states=decoder_input,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                **(extra_block_kwargs or {}),
            )

        if not self.post_process:
            return hidden_states

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output)

        if has_config_logger_enabled(self.config):
            payload = OrderedDict({
                'input_ids': input_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'decoder_input': decoder_input,
                'logits': logits,
            })
            log_config_to_disk(self.config, payload, prefix='input_and_logits')

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)

        return loss
