# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from megatron.core.parallel_state import get_tensor_model_parallel_rank
from megatron.core.tensor_parallel import all_gather_last_dim_from_tensor_parallel_region
from megatron.core.transformer.attention import SelfAttention

try:
    from megatron.core.extensions.transformer_engine import SplitAlongDim
except ImportError:
    SplitAlongDim = None


class GatedSelfAttention(SelfAttention):

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, *args, **kwargs):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        assert self.config.num_query_groups is not None
        if getattr(self, 'world_size', None) is not None and self.config.num_query_groups < self.world_size:
            # Note that weights are interleaved in the following manner:
            # q1 q2 k1 v1 | q3 q4 k2 v2 | q5 q6 k3 v3 | ...
            # When tp_size > num_kv_heads, we split "q1 q2 k1 v1" over multiple
            # ranks, so a rank does not have a clean partitioning of just the q_heads
            # it needs. Instead, we perform the following steps:
            # 1. Assemble the full "q1 q2 k1 v1 | q3 q4 k2 v2 | q5 q6 k3 v3 | ..."
            #    through an AG.
            # 2. Pull out the right slice (e.g., "q1 q2 k1 v1" or "q3 q4 k2 v2").
            # 3. Split q_heads (e.g., q1, q2), k_heads (e.g., k1), v_heads (e.g., v1).
            # 4. Further index into query to get only the q_heads that this rank is
            #    responsible for (e.g., q1).
            mixed_qkv = all_gather_last_dim_from_tensor_parallel_region(mixed_qkv)
            idx = get_tensor_model_parallel_rank() // (self.world_size // self.config.num_query_groups)
            size = mixed_qkv.size()[-1] // self.config.num_query_groups
            mixed_qkv = mixed_qkv[:, :, idx * size:(idx + 1) * size]

        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            ((self.num_attention_heads_per_partition // self.num_query_groups_per_partition * 2 + 2)
             * self.hidden_size_per_attention_head),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        split_arg_list = [
            (self.num_attention_heads_per_partition // self.num_query_groups_per_partition
             * self.hidden_size_per_attention_head * 2),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:

            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:

            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        if getattr(self, 'world_size', None) is not None and self.config.num_query_groups < self.world_size:
            # query above corresponds to (num_q_heads / num_kv_heads) q_heads.
            # Index appropriately into query to get (num_q_heads / tp_size) q_heads.
            # This is step 4 in the list of steps above.
            idx = get_tensor_model_parallel_rank() % (self.world_size // self.config.num_query_groups)
            size = query.shape[2] // (self.world_size // self.config.num_query_groups)
            query = query[:, :, idx * size:(idx + 1) * size, :]
        query, gate = query[:, :, ::2], query[:, :, 1::2]
        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value, gate
