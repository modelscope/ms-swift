# Copyright (c) Alibaba, Inc. and its affiliates.
import concurrent.futures
import os
import subprocess
import sys
from contextlib import contextmanager
from copy import copy
from typing import List, Optional, Tuple

import peft
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from tqdm import tqdm

from swift.llm import git_clone_github
from swift.utils import (get_logger, is_flash_attn_3_available, is_megatron_available, safe_ddp_context, split_list,
                         subprocess_run)

logger = get_logger()


def _patch_transformer_engine():
    import transformer_engine
    try:
        from transformer_engine.pytorch.attention import apply_rotary_pos_emb
    except ImportError:
        try:
            transformer_engine.pytorch.attention.apply_rotary_pos_emb = (
                transformer_engine.pytorch.attention.rope.apply_rotary_pos_emb)
            logger.info('Patch apply_rotary_pos_emb successfully applied.')
        except (ImportError, AttributeError):
            pass
    try:
        from transformer_engine.pytorch.attention import _SplitAlongDim
    except ImportError:
        try:
            transformer_engine.pytorch.attention._SplitAlongDim = (transformer_engine.pytorch.utils.SplitAlongDim)
            logger.info('Patch _SplitAlongDim successfully applied.')
        except (ImportError, AttributeError):
            pass


def _patch__batched_p2p_ops():
    from megatron.core.pipeline_parallel import p2p_communication

    _batched_p2p_ops_origin = p2p_communication._batched_p2p_ops

    def _batched_p2p_ops(**kwargs):
        kwargs['group'] = None
        return _batched_p2p_ops_origin(**kwargs)

    p2p_communication._batched_p2p_ops = _batched_p2p_ops


def _patch_mla_attention():
    # support thd
    import megatron.core
    from megatron.core.utils import deprecate_inference_params
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.transformer.multi_latent_attention import MultiLatentAttention, MLASelfAttention
    from megatron.core.tensor_parallel.mappings import (
        gather_from_sequence_parallel_region,
        gather_from_tensor_model_parallel_region,
        scatter_to_sequence_parallel_region,
    )
    megatron_core_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')

    # Code borrowed from NVIDIA/Megatron-LM
    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        """Forward pass for multi-latent attention"""
        assert attention_bias is None, 'Attention bias should not be passed into MLA.'
        assert (rotary_pos_cos is None and rotary_pos_sin is None), 'MLA does not support Flash Decoding'

        # hidden_states: [sq, b, h]

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # query: [96, 1, 16, 128], key:[96, 1, 16, 128], value:[96, 1, 16, 128]
        query, key, value = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            rotary_pos_emb=rotary_pos_emb,
            inference_context=inference_context,
        )

        # ===================================================
        # Adjust key, value for inference
        # ===================================================
        # rotary_pos_emb = None
        if megatron_core_013:
            query, key, value, _, attn_mask_type, _ = self._adjust_key_value_for_inference(
                inference_context, query, key, value, rotary_pos_emb=None)
        else:
            query, key, value, _, attn_mask_type = self._adjust_key_value_for_inference(
                inference_context, query, key, value, rotary_pos_emb=None)

        # TODO: Currently, TE can only accept contiguous tensors for MLA
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # ==================================
        # core attention computation
        # ==================================
        # Need corresponding TE change
        thd_qkv_format = packed_seq_params and packed_seq_params.qkv_format == 'thd'
        v_dim = value.shape[-1]
        if thd_qkv_format and query.shape[-1] != v_dim:
            value = F.pad(value, [0, query.shape[-1] - v_dim])
            self.core_attention.hidden_size_per_attention_head_v = value.shape[-1]
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, packed_seq_params=packed_seq_params)
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                packed_seq_params=packed_seq_params,
                attn_mask_type=attn_mask_type,
            )
        if thd_qkv_format:
            if core_attn_out.ndim == 2:
                core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-1], -1, value.shape[-1])
            if query.shape[-1] != v_dim:
                core_attn_out = core_attn_out[..., :v_dim]
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.linear_proj(core_attn_out)

        return output, bias

    MultiLatentAttention.forward = forward

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        rotary_pos_emb=None,
        *,
        inference_params=None,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # s = sequence length, b = batch size, h = hidden size, n = num attention heads
        # Attention heads [s, b, n*h]
        assert (hidden_states.ndim == 3), f'hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D'

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if packed_seq_params is not None:
            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        if self.config.q_lora_rank is not None:
            # if linear_q_down_proj is ColumnParallelLinear:
            #     q_compressed: [s, b, q_lora_rank / TP]
            # elif linear_q_down_proj is Linear:
            #     q_compressed: [s / TP, b, q_lora_rank]
            q_compressed, _ = self.linear_q_down_proj(hidden_states)

            # When output is sharded (ColumnParallelLinear), two things are needed to be
            # identical to a normal Linear.
            #   1. Manually gather output to restore output dim q_lora_rank;
            #   2. Scatter sequence back to s / TP if sequence-parallel since it was
            #      gathered by ColumnParallelLinear.
            if q_compressed.size(-1) != self.config.q_lora_rank:
                q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
                if self.config.sequence_parallel:
                    q_compressed = scatter_to_sequence_parallel_region(q_compressed)

            q_compressed = self.q_layernorm(q_compressed)
        else:
            q_compressed = hidden_states

        # if linear_kv_down_proj is ColumnParallelLinear:
        #     kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim) / TP]
        # elif linear_kv_down_proj is Linear:
        #     kv_combined: [s / TP, b, (kv_lora_rank + qk_pos_emb_head_dim)]
        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        if kv_combined.size(-1) != self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim:
            # kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim)]
            kv_combined = gather_from_tensor_model_parallel_region(kv_combined)
            # kv_compressed:[s, b, kv_lora_rank], k_pos_emb: [s, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1)
            if self.config.sequence_parallel:
                # kv_compressed:[s / TP, b, kv_lora_rank]
                kv_compressed = scatter_to_sequence_parallel_region(kv_compressed)
        else:
            # kv_compressed:[s / TP, b, kv_lora_rank], k_pos_emb: [s / TP, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1)
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                # k_pos_emb: [s, b, qk_pos_emb_head_dim]
                k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb)

        kv_compressed = self.kv_layernorm(kv_compressed)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================
        def qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb):
            """
            Apply the up projection and RoPE to the query and key.
            When sequence packing enabled, the input tensors adopt a packed shape of [t, ...];
            otherwise, they maintain the unpacked shape [s, b, ...]. In subsequent code comments,
            we uniformly use [num_tokens, ...] to denote [s, b, ...] or [t, ...] for two cases.
            """
            if self.config.q_lora_rank is not None:
                # q_compressed: [num_tokens, q_lora_rank]
                # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
                q, _ = self.linear_q_up_proj(q_compressed)
            else:
                # q_compressed: [num_tokens, hidden_size]
                # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
                q, _ = self.linear_q_proj(q_compressed)

            # q: [num_tokens, n, q_head_dim]
            q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)

            # kv: [num_tokens, n * (qk_head_dim + v_head_dim)]
            kv, _ = self.linear_kv_up_proj(kv_compressed)

            # kv: [num_tokens, n, (qk_head_dim + v_head_dim)]
            kv = kv.view(
                *kv.size()[:-1],
                self.num_attention_heads_per_partition,
                self.config.qk_head_dim + self.config.v_head_dim,
            )

            q_len = q.size()[0]
            if inference_context is not None:
                # add offset to the sequence start for inference
                sequence_start = inference_context.sequence_len_offset
                sequence_end = sequence_start + q_len
                rotary_pos_emb = rotary_pos_emb[sequence_start:sequence_end]
            # Remove the else branch to fix cp.

            # [num_tokens, qk_pos_emb_head_dim] -> [num_tokens, 1, qk_pos_emb_head_dim]
            k_pos_emb = torch.unsqueeze(k_pos_emb, -2)

            # q_no_pe: [num_tokens, n, qk_head_dim]
            # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
            q_no_pe, q_pos_emb = torch.split(q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1)

            # k_no_pe: [num_tokens, n, qk_head_dim]
            # value: [num_tokens, n, v_head_dim]
            k_no_pe, value = torch.split(kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1)
            # This function will be patched and supports mscale.
            from megatron.core.transformer.attention import apply_rotary_pos_emb
            # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
            q_pos_emb = apply_rotary_pos_emb(
                q_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_q,
            )
            # k_pos_emb:[num_tokens, 1, qk_pos_emb_head_dim]
            k_pos_emb = apply_rotary_pos_emb(
                k_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
            )

            # query: [num_tokens, n, (qk_head_dim + v_head_dim)]
            query = torch.cat([q_no_pe, q_pos_emb], dim=-1)

            # key: [num_tokens, n, (qk_head_dim + v_head_dim)]Add commentMore actions
            if k_pos_emb.ndim == 4:
                k_pos_emb = k_pos_emb.expand(-1, -1, self.num_attention_heads_per_partition, -1)
            else:
                assert k_pos_emb.ndim == 3
                k_pos_emb = k_pos_emb.expand(-1, self.num_attention_heads_per_partition, -1)
            key = torch.cat([k_no_pe, k_pos_emb], dim=-1)

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            return query, key, value

        if packed_seq_params is not None:
            # If sequence packing, TE expect [t, h, d] shaped qkv input.
            # In Megatron-Core, the qkv shape is [t, 1, h, d].
            # So we need to reshape qkv from [t, 1, h, d] to [t, h, d].
            q_compressed = q_compressed.squeeze(1)
            kv_compressed = kv_compressed.squeeze(1)
            k_pos_emb = k_pos_emb.squeeze(1)

        if self.recompute_up_proj:
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            query, key, value = self.qkv_up_checkpoint.checkpoint(qkv_up_proj_and_rope_apply, q_compressed,
                                                                  kv_compressed, k_pos_emb, rotary_pos_emb)
        else:
            query, key, value = qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb)

        return query, key, value

    MLASelfAttention.get_query_key_value_tensors = get_query_key_value_tensors


def _patch_peft_BaseTuner():
    from peft.tuners.tuners_utils import BaseTuner
    _origin_get_tied_target_modules = BaseTuner._get_tied_target_modules

    def _get_tied_target_modules(self, model: nn.Module) -> List[str]:
        try:
            return _origin_get_tied_target_modules(self, model)
        except AttributeError:
            tied_target_modules = []
            if model.share_embeddings_and_output_weights:
                for target_module in self.targeted_module_names:
                    if target_module.split('.')[-1] in ['output_layer', 'embedding']:
                        tied_target_modules.append(target_module)
            return tied_target_modules

    BaseTuner._get_tied_target_modules = _get_tied_target_modules


def _patch_TEGroupedLinear():
    from megatron.core.extensions.transformer_engine import TEGroupedLinear

    def sharded_state_dict(
            self,
            prefix: str = '',
            sharded_offsets: Tuple[Tuple[int, int, int]] = (),
            metadata: Optional[dict] = None,
    ):
        return self._sharded_state_dict_grouped(None, prefix, sharded_offsets, metadata)

    TEGroupedLinear.sharded_state_dict = sharded_state_dict


def _patch_peft_ModulesToSaveWrapper():
    if version.parse(peft.__version__) >= version.parse('0.16'):
        from peft.utils import other as peft_module
    else:
        from peft.tuners import tuners_utils as peft_module
    from megatron.core.dist_checkpointing.mapping import ShardedStateDict
    from .utils import tuners_sharded_state_dict

    OriginModulesToSaveWrapper = peft_module.ModulesToSaveWrapper

    class ModulesToSaveWrapper(OriginModulesToSaveWrapper):

        def sharded_state_dict(
                self,
                prefix: str = '',
                sharded_offsets: Tuple[Tuple[int, int, int]] = (),
                metadata: Optional[dict] = None,
        ) -> ShardedStateDict:
            sharded_state_dict = tuners_sharded_state_dict(self, prefix, sharded_offsets, metadata)
            if prefix in {'output_layer.', 'language_model.output_layer.'}:
                for k in list(sharded_state_dict.keys()):
                    if '_extra_state' in k:
                        # Old GPT checkpoints only stored the output layer weight key. So we remove the
                        # _extra_state key but check that it doesn't contain any data anyway
                        output_extra_state = sharded_state_dict.pop(k, None)
                        assert not (output_extra_state and output_extra_state.data
                                    ), f'Expected output layer extra state to be empty, got: {output_extra_state}'
                # fix error
                if f'{prefix}modules_to_save.default.weight' in sharded_state_dict:
                    sharded_state_dict[f'{prefix}weight'] = sharded_state_dict[
                        f'{prefix}modules_to_save.default.weight']
            return sharded_state_dict

    peft_module.ModulesToSaveWrapper = ModulesToSaveWrapper
    peft_module.OriginModulesToSaveWrapper = OriginModulesToSaveWrapper


def _patch_TransformerLayer():
    import megatron.core
    from megatron.training import get_args
    from megatron.core.transformer import TransformerLayer
    _origin_forward = TransformerLayer.forward
    megatron_core_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')

    def forward(self, *_args, **kwargs):
        """
        Perform a forward pass through the transformer layer.

        This method calls the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.
        """
        if not megatron_core_013:
            return _origin_forward(self, *_args, **kwargs)
        hidden_states, context = self._forward_attention(*_args, **kwargs)
        args = get_args()
        mlp_padding_free = args.mlp_padding_free and 'attention_mask' in kwargs
        if mlp_padding_free:
            mask = (kwargs['attention_mask'].sum(dim=(1, 3)) > 0).t()
            hidden_states = hidden_states[mask][:, None]
        output = self._forward_mlp(hidden_states, kwargs.get('inference_context', None))
        if mlp_padding_free:
            new_output = hidden_states.new_zeros((*mask.shape, output.shape[-1]))
            new_output[mask] = output.squeeze(1)
            output = new_output
        return output, context

    TransformerLayer.forward = forward


def _patch_compile_helpers():
    from megatron.core.datasets import utils

    def compile_helpers():
        command = ['make', '-C', os.path.abspath(os.path.dirname(utils.__file__))]
        if subprocess.run(command).returncode != 0:
            logger.warning('Failed to compile the C++ dataset helper functions')

    utils.compile_helpers = compile_helpers


def _patch_flash_attn():
    # flash_attention_3
    if is_flash_attn_3_available():
        import flash_attn_interface
        sys.modules['flash_attn_3.flash_attn_interface'] = flash_attn_interface


def _patch_torch_FileSystemReader():
    from torch.distributed.checkpoint.filesystem import FileSystemReader
    from torch.futures import Future
    _origin_read_data = FileSystemReader.read_data
    _origin__slice_file = FileSystemReader._slice_file
    READER_MAX_WORKERS = int(os.environ.get('MCORE_READER_MAX_WORKERS', '16'))

    @contextmanager
    def _patch__slice_file(prog_bar):

        def _slice_file(self, *args, **kwargs):
            prog_bar.update()
            return _origin__slice_file(self, *args, **kwargs)

        FileSystemReader._slice_file = _slice_file
        try:
            yield
        finally:
            FileSystemReader._slice_file = _origin__slice_file

    def read_data(self, plan, planner):

        def _worker(plan_shard):
            _origin_read_data(self, plan_shard, planner)

        prog_bar = tqdm(total=len(plan.items), dynamic_ncols=True, desc='Loading: ')
        plan_shards = split_list(plan.items, READER_MAX_WORKERS, contiguous=False)
        with _patch__slice_file(prog_bar):
            with concurrent.futures.ThreadPoolExecutor(max_workers=READER_MAX_WORKERS) as pool:
                futures = []
                for i in range(READER_MAX_WORKERS):
                    plan_shard = copy(plan)
                    plan_shard.items = plan_shards[i]
                    futures.append(pool.submit(_worker, plan_shard))
                concurrent.futures.wait(futures)
        prog_bar.close()
        fut: Future = Future()
        fut.set_result(None)
        return fut

    FileSystemReader.read_data = read_data


def _patch_TELinear():
    from megatron.core.extensions.transformer_engine import TELinear

    def __repr__(self):
        return (f'{type(self).__name__}(in_features={self.in_features}, '
                f'out_features={self.out_features}, bias={self.use_bias}, TP={self.tp_size})')

    TELinear.__repr__ = __repr__


def _patch_mrope():
    from megatron.core.models.common.embeddings.rotary_pos_embedding import MultimodalRotaryEmbedding
    from megatron.core import parallel_state
    from megatron.core.models.common.embeddings.rope_utils import (get_pos_emb_on_this_cp_rank,
                                                                   _apply_rotary_pos_emb_bshd)
    from megatron.core.models.common.embeddings import rope_utils
    from megatron.training import get_args

    # Code borrowed from huggingface/transformers
    def apply_interleaved_mrope(freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    # Code borrowed from NVIDIA/Megatron-LM
    def forward(self, position_ids, mrope_section: List[int], packed_seq: bool = False) -> torch.Tensor:
        seq = position_ids.to(device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        # shape (3, bs, dim, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(3, seq.shape[1], -1, 1)
        # shape (3, bs, 1, seq_length)
        seq_expanded = seq[:, :, None, :].float()
        # shape (3, bs, seq_length, dim)
        freqs = (inv_freq_expanded @ seq_expanded).transpose(2, 3)
        args = get_args()
        if args.mrope_interleaved:
            freqs = apply_interleaved_mrope(freqs, mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            # first part even vector components, second part odd vector components,
            #  2 * dim in dimension size
            if not self.rotary_interleaved:
                emb = torch.cat((freqs, freqs), dim=-1)  # shape (3, bs, seq_length, 2 * dim)
            else:
                bs = freqs.shape[1]
                emb = torch.stack((freqs.reshape(3, bs, -1, 1), freqs.reshape(3, bs, -1, 1)),
                                  dim=-1).view(3, bs, freqs.shape[2], -1)

            # generate freqs with mrope_section
            # shape (bs, seq_length, 2 * dim)
            mrope_section = mrope_section * 2
            emb = torch.cat([m[i % 3] for i, m in enumerate(emb.split(mrope_section, dim=-1))], dim=-1)

        # shape (seq_length, bs, 1, 2 * dim)
        emb = emb[..., None, :].transpose(0, 1).contiguous()
        if parallel_state.get_context_parallel_world_size() > 1 and not packed_seq:
            # slice rotary_pos_emb along sequence dimension and select the parition of the current
            # CP rank
            emb = get_pos_emb_on_this_cp_rank(emb, 0, parallel_state.get_context_parallel_group())
        return emb

    MultimodalRotaryEmbedding.forward = forward
    _origin_apply_rotary_pos_emb_thd = rope_utils._apply_rotary_pos_emb_thd

    def _apply_rotary_pos_emb_thd(
        t: torch.Tensor,
        cu_seqlens: torch.Tensor,
        freqs: torch.Tensor,
        rotary_interleaved: bool = False,
        multi_latent_attention: bool = False,
        mscale: float = 1.0,
        cp_group: torch.distributed.ProcessGroup = None,
    ) -> torch.Tensor:
        """A baseline implementation of applying RoPE for `thd` format.

        Args:
            t (Tensor): Input tensor T is of shape [t, h, d]
            cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
            with shape [b + 1] and dtype torch.int32.
            freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]
            cp_group (torch.distributed.ProcessGroup): The context parallel group

        Returns:
            Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
        """
        args = get_args()
        if args.position_embedding_type != 'mrope':
            return _origin_apply_rotary_pos_emb_thd(
                t,
                cu_seqlens,
                freqs,
                rotary_interleaved=rotary_interleaved,
                multi_latent_attention=multi_latent_attention,
                mscale=mscale,
                cp_group=cp_group,
            )

        if cp_group is None:
            raise ValueError('cp_group must be provided for THD format RoPE')
        cp_size = cp_group.size()
        cu_seqlens = cu_seqlens // cp_size
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        return torch.cat([
            _apply_rotary_pos_emb_bshd(
                x.unsqueeze(1),
                f,
                rotary_interleaved=rotary_interleaved,
                multi_latent_attention=multi_latent_attention,
                mscale=mscale,
            ) for x, f in zip(torch.split(t, seqlens), torch.split(freqs, seqlens))
        ]).squeeze(1)

    rope_utils._apply_rotary_pos_emb_thd = _apply_rotary_pos_emb_thd


def _patch_megatron():
    _patch_flash_attn()
    _patch_transformer_engine()
    _patch_TELinear()
    _patch__batched_p2p_ops()
    _patch_mla_attention()
    _patch_TEGroupedLinear()
    _patch_TransformerLayer()
    _patch_compile_helpers()
    _patch_mrope()
    from swift.megatron import tuners  # patch lora
    try:
        _patch_torch_FileSystemReader()
        logger.info('Patch FileSystemReader successfully applied.')
    except Exception:
        pass
    try:
        _patch_peft_BaseTuner()
        _patch_peft_ModulesToSaveWrapper()
        logger.info('Patch peft successfully applied.')
    except Exception:
        pass

    import megatron.core
    logger.info(f'megatron.core.__version__: {megatron.core.__version__}')


def init_megatron_env() -> None:
    if 'MEGATRON_LM_PATH' not in os.environ:
        # TODO: Synchronization issues may occur in DDP scenarios
        # if the distributed environment has not been initialized.
        os.environ['MEGATRON_LM_PATH'] = git_clone_github(
            'https://github.com/NVIDIA/Megatron-LM', branch='core_r0.13.0')
    with safe_ddp_context(hash_id='megatron-lm'):
        if not is_megatron_available():
            subprocess_run([sys.executable, '-m', 'pip', 'install', '-e', os.environ['MEGATRON_LM_PATH']])
    sys.path.insert(0, os.environ['MEGATRON_LM_PATH'])
    _patch_megatron()
