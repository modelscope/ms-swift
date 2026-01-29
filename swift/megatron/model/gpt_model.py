# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import os
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Literal, Optional, Tuple

import megatron.core
import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt import GPTModel as McoreGPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import (gather_from_sequence_parallel_region,
                                                    gather_from_tensor_model_parallel_region,
                                                    reduce_from_tensor_model_parallel_region)
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, MTPLossLoggingHelper, roll_tensor
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import WrappedTensor, deprecate_inference_params
from megatron.training import get_args
from packaging import version

from swift.utils import get_logger
from .rope import dynamic_rope_update, get_rope_inv_freq

logger = get_logger()

mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')


class OutputLayerLinear(TELinear):

    def forward(self, hidden_states, *args, **kwargs):
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
        vocab_size = math.ceil(vocab_size / config.tensor_model_parallel_size) * config.tensor_model_parallel_size
        if config.multi_latent_attention and config.rope_type == 'yarn':
            config.rope_type = 'rope'  # use transformers implementation
            if hf_rope_scaling and hf_rope_scaling['rope_type'] == 'yarn':
                # softmax_scale
                config.mscale = hf_rope_scaling['mscale']
                config.mscale_all_dim = hf_rope_scaling['mscale_all_dim']
                config.rotary_scaling_factor = hf_rope_scaling['factor']
        self.hf_rope_scaling = hf_rope_scaling
        if mcore_013:
            kwargs = {'vp_stage': vp_stage}
        else:
            self.vp_stage = vp_stage
            assert vp_stage is None, 'megatron-core==0.12 does not support vp_stage'
            kwargs = {}
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
            **kwargs,
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
            self.output_layer.weight.average_gradients_across_tp_domain = True
        elif args.task_type == 'embedding' and self.post_process:
            self.output_layer = None

        if (self.attention_scaling != 1 or position_embedding_type == 'mrope') and config.apply_rope_fusion:
            config.apply_rope_fusion = False
            if self.attention_scaling != 1:
                warning_string = 'attention_scaling'
            else:
                warning_string = 'mrope'
            logger.warning(f'`apply_rope_fusion` does not support `{warning_string}`. '
                           f'Setting `config.apply_rope_fusion`: {config.apply_rope_fusion}')
        if self.attention_scaling != 1:
            self._patch_apply_rotary_pos_emb()
        if getattr(self, 'mtp', None) is not None:
            for layer in self.mtp.layers:
                attention = layer.transformer_layer.self_attention
                attention.config = deepcopy(attention.config)
                attention.config.apply_rope_fusion = False

    def _patch_apply_rotary_pos_emb(self):
        from megatron.core.transformer import attention
        origin_apply_rotary_pos_emb = attention.apply_rotary_pos_emb

        def apply_rotary_pos_emb(*args, **kwargs):
            kwargs['mscale'] = self.attention_scaling
            return origin_apply_rotary_pos_emb(*args, **kwargs)

        attention.apply_rotary_pos_emb = apply_rotary_pos_emb
        attention.origin_apply_rotary_pos_emb = origin_apply_rotary_pos_emb

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
                    if attention_scaling is not None and attention_scaling != self.attention_scaling:
                        raise ValueError('Currently does not support changing attention_scaling during training. '
                                         f'args.attention_scaling: {self.attention_scaling}, '
                                         f'current_attention_scaling: {attention_scaling}.')
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

        # MTP: https://github.com/NVIDIA/Megatron-LM/issues/1661
        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
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

    def _postprocess(
        self,
        hidden_states,
        input_ids,
        position_ids,
        labels,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        loss_mask=None,
        decoder_input=None,
        attention_mask=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        runtime_gather_output=None,
        extra_block_kwargs=None,
        inference_context=None,
    ):
        """Postprocesses decoder hidden states to generate logits or compute loss.

        Applies Multi-Token Prediction if enabled, generates output logits through
        the output layer, and computes language model loss when labels are provided.
        """
        if not self.post_process:
            return hidden_states
        args = get_args()
        labels = labels if args.task_type == 'causal_lm' else None
        in_inference_mode = inference_context is not None and not self.training
        if in_inference_mode:
            assert runtime_gather_output, 'Inference must always gather TP logits'

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        if self.mtp_process:
            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                embedding=self.embedding,
                **(extra_block_kwargs or {}),
            )
            hidden_states_list = torch.chunk(hidden_states, 1 + self.config.mtp_num_layers, dim=0)
            hidden_states = hidden_states_list[0]

            if labels is not None:
                from ..trainers.utils import split_cp_inputs
                mtp_labels = labels.clone()
                if loss_mask is None:
                    # if loss_mask is not provided, use all ones as loss_mask
                    if packed_seq_params is None:
                        loss_mask = torch.ones_like(mtp_labels)
                    else:
                        loss_mask = mtp_labels.new_ones((1, packed_seq_params.cu_seqlens_q[-1]))
                cu_seqlens = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
                for mtp_layer_number in range(self.config.mtp_num_layers):
                    # output
                    mtp_logits, _ = self.output_layer(
                        hidden_states_list[mtp_layer_number + 1],
                        weight=output_weight,
                        runtime_gather_output=runtime_gather_output,
                    )
                    # Calc loss for the current Multi-Token Prediction (MTP) layers.
                    mtp_labels, _ = roll_tensor(mtp_labels, shifts=-1, dims=-1, cp_group=self.cp_group)
                    if cu_seqlens is None:
                        loss_mask_, _ = roll_tensor(loss_mask, shifts=-1, dims=-1, cp_group=self.cp_group)
                    else:
                        loss_mask[:, cu_seqlens[:-1]] = 0
                        loss_mask, _ = roll_tensor(loss_mask, shifts=-1, dims=-1)
                        if args.context_parallel_size > 1:
                            loss_mask_ = split_cp_inputs(loss_mask, cu_seqlens, dim=1)
                        else:
                            loss_mask_ = loss_mask.clone()
                    mtp_loss = self.compute_language_model_loss(mtp_labels, mtp_logits)
                    mtp_loss = loss_mask_ * mtp_loss
                    num_tokens = loss_mask_.sum()
                    if self.training:
                        # TODO(shifangx): remove the use of parallel_state here
                        # after moving loss logging to loss_func in pretrain_gpt.py
                        MTPLossLoggingHelper.save_loss_to_tracker(
                            torch.sum(mtp_loss) / num_tokens,
                            mtp_layer_number,
                            self.config.mtp_num_layers,
                            avg_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                        )
                    mtp_loss_scale = self.config.mtp_loss_scaling_factor / self.config.mtp_num_layers
                    if self.config.calculate_per_token_loss:
                        hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_scale * mtp_loss)
                    else:
                        hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_scale * mtp_loss / num_tokens)
        sequence_parallel_override = False
        if in_inference_mode and inference_context.materialize_only_last_token_logits:
            if inference_context.is_static_batching():
                hidden_states = hidden_states[-1:, :, :]
            else:
                if self.output_layer.sequence_parallel:
                    # Perform the sequence parallel gather here instead of after the output layer
                    # because we need to slice the last token logits from the full view of the
                    # packed logits across all requests.
                    # TODO(ksanthanam): Make the equivalent change in the `MambaModel` code after
                    # merging in !3722.
                    hidden_states = gather_from_sequence_parallel_region(hidden_states, group=self.pg_collection.tp)
                    self.output_layer.sequence_parallel = False
                    sequence_parallel_override = True

                # Reshape [B, 1, H] to [1, B, H] → extract each sample’s true last‐token hidden
                # state ([B, H]) → unsqueeze back to [1, B, H]
                # (so that the output layer, which expects S×B×H, receives only the final token)
                hidden_states = inference_context.last_token_logits(hidden_states.squeeze(1).unsqueeze(0)).unsqueeze(1)

        if args.task_type in {'seq_cls', 'embedding'
                              } and args.sequence_parallel and args.tensor_model_parallel_size > 1:
            hidden_states = gather_from_sequence_parallel_region(hidden_states)

        if args.task_type == 'embedding':
            logits = F.normalize(hidden_states, p=2, dim=-1)
        else:
            logits, _ = self.output_layer(
                hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output)
            if args.task_type == 'generative_reranker':
                logits = gather_from_tensor_model_parallel_region(logits)
                positive_token = os.environ.get('GENERATIVE_RERANKER_POSITIVE_TOKEN', 'yes')
                negative_token = os.environ.get('GENERATIVE_RERANKER_NEGATIVE_TOKEN', 'no')
                positive_token_id = self.tokenizer.convert_tokens_to_ids(positive_token)
                negative_token_id = self.tokenizer.convert_tokens_to_ids(negative_token)
                logits = (logits[..., positive_token_id] - logits[..., negative_token_id])[..., None]
        # Restore sequence parallel execution to the output layer if necessary.
        if sequence_parallel_override:
            assert (in_inference_mode and inference_context.is_dynamic_batching()
                    and inference_context.materialize_only_last_token_logits)
            self.output_layer.sequence_parallel = True

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

    def get_input_tensor(self):
        return self.decoder.input_tensor
