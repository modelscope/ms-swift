"""
Router Replay Utilities
Utilities for handling router replay functionality in Megatron models.
"""

import torch
from mcore_bridge import split_cp_inputs
from megatron.core import mpu
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from swift.utils import get_logger
from swift.utils.torch_utils import get_current_device

logger = get_logger()

try:
    from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
    ROUTER_REPLAY_AVAILABLE = True
except ImportError:
    logger.warning('RouterReplay not available in current megatron-core version')
    RouterReplay = None
    RouterReplayAction = None
    MoEAlltoAllTokenDispatcher = None
    ROUTER_REPLAY_AVAILABLE = False

device_name = get_current_device()


def is_moe_layer(tf_config, layer_idx):
    moe_layer_freq = getattr(tf_config, 'moe_layer_freq', None)
    if isinstance(moe_layer_freq, int):
        return layer_idx % moe_layer_freq == 0
    elif isinstance(moe_layer_freq, list):
        return moe_layer_freq[layer_idx] == 1
    else:
        raise ValueError(f'Unsupported moe_layer_freq type: {type(moe_layer_freq)}')


def get_moe_num_layers_to_build(tf_config, vp_stage=None, pp_rank=None):
    """Count the number of MoE layers assigned to the current rank.
    When ``moe_layer_freq`` is 1 or unset, every transformer layer is an MoE
    layer, so the count equals the total layer count. Otherwise only layers
    whose global index satisfies the frequency predicate are counted.
    Args:
        config: Megatron TransformerConfig providing layer layout information.
        vp_stage: Virtual-pipeline stage index (None defaults to current).
        pp_rank: Pipeline-parallel rank (None defaults to current).
    Returns:
        Number of MoE layers on the specified rank/stage.
    """
    total_layers = get_num_layers_to_build(tf_config, vp_stage=vp_stage, pp_rank=pp_rank)

    layer_offset = get_transformer_layer_offset(tf_config, vp_stage=vp_stage)
    local_global_indices = range(layer_offset, layer_offset + total_layers)

    num_moe_layers = sum(1 for idx in local_global_indices if is_moe_layer(tf_config, idx))

    return num_moe_layers


def get_local_layer_range(tf_config, vp_rank=None, only_moe_layer=True):
    vp_size = tf_config.virtual_pipeline_model_parallel_size
    if vp_size is not None:
        vp_rank = 0 if vp_rank is None else vp_rank
        offset = 0
        for pre_vp_stage in range(vp_size):
            if pre_vp_stage == vp_rank:
                break
            num_layers_to_build = get_moe_num_layers_to_build(
                tf_config, pre_vp_stage) if only_moe_layer else get_num_layers_to_build(tf_config, pre_vp_stage)
            offset += num_layers_to_build
    else:
        offset = 0
    count = get_moe_num_layers_to_build(tf_config, vp_rank) if only_moe_layer else get_num_layers_to_build(
        tf_config, vp_rank)
    return offset, count


def get_local_topk_idx_for_current_rank(global_topk_idx, tf_config, packed_seq_params=None):
    if global_topk_idx is None:
        return None
    # 1. pp slice
    # For the hybrid model, global_topk_idx contains data from all layers
    # because vLLM reports routed_experts across all transformer layers(including dense).
    # However megatron only has routers for MoE layers.
    # So local_topk_idx should filter only data from the MoE layer.
    layer_offset = get_transformer_layer_offset(tf_config, vp_stage=0)
    offset, count = get_local_layer_range(
        tf_config, tf_config.virtual_pipeline_model_parallel_size, only_moe_layer=False)
    num_layers = offset + count
    moe_layer_idx = torch.tensor([
        layer_idx for layer_idx in range(layer_offset, layer_offset + num_layers) if is_moe_layer(tf_config, layer_idx)
    ],
                                 dtype=torch.long,
                                 device=global_topk_idx.device)
    local_topk_idx = torch.index_select(global_topk_idx, dim=2, index=moe_layer_idx)
    # 2. cp slice
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        local_topk_idx = split_cp_inputs(local_topk_idx, getattr(packed_seq_params, 'cu_seqlens_q', None), 1)
    # 3. sp slice
    local_topk_idx = scatter_to_sequence_parallel_region(local_topk_idx.transpose(0, 1)).transpose(0, 1)
    return local_topk_idx


def get_router_replay_data(tf_config, vp_rank=None):
    router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
    layers_topk_idx = []
    for router in router_instances_list:
        layers_topk_idx.append(router.recorded_topk_idx.to(torch.uint8))
    # layer_num, seq_len, topk -> 1, seq_len, layer_num, topk
    layers_topk_idx = torch.stack(layers_topk_idx).transpose(0, 1).unsqueeze(0).to(device_name)
    return layers_topk_idx


def set_router_replay_data(layers_topk_idx, tf_config, vp_rank=None):
    # bs, seq_len, layer_num, topk -> layer_num, total_seq_len, topk
    layers_topk_idx_reshape = layers_topk_idx.flatten(0, 1).transpose(0, 1).to(device_name)
    offset, count = get_local_layer_range(tf_config, vp_rank)
    router_instances_list = RouterReplay.global_router_replay_instances[offset:offset + count]
    for i, router in enumerate(router_instances_list):
        router.set_target_indices(layers_topk_idx_reshape[i + offset].to(torch.int64))


class RouterReplayHelper:
    """Helper class to query router replay state and locate local RouterReplay instances."""

    @staticmethod
    def get_micro_batch_router_list(tf_config, vp_rank=None):
        """
        Return the list of RouterReplay instances corresponding to the current micro-batch and local
        (pp_rank, vp_stage) layer range.

        When virtual pipeline (VPP) is enabled, the local range for the PP rank is expanded to include
        all VP stages by multiplying the per-VP count by vp_size. The returned slice is taken from the
        global RouterReplay.global_router_replay_instances list.

        Args:
            tf_config: Configuration object used to compute layer assignments.
            vp_rank (Optional[int]): Explicit virtual pipeline stage to query. If None, the current VP
                rank from Megatron parallel state is used when available.
        Returns:
            list: A contiguous sublist of RouterReplay.router_instances for the local layer range.
        """
        offset, count = get_local_layer_range(tf_config, vp_rank)
        router_instances_list = RouterReplay.global_router_replay_instances[offset:offset + count]
        return router_instances_list

    @staticmethod
    def is_r2_record_action(tf_config, vp_rank=None) -> bool:
        """Return True if the current router_replay_action is RECORD (R2) for the local router instances.

        This inspects the first local RouterReplay instance's router_replay_action and compares it to
        RouterReplayAction.RECORD.
        """
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return (router_instances_list and router_instances_list[0].router_replay_action == RouterReplayAction.RECORD)

    @staticmethod
    def is_replay_forward_action(tf_config, vp_rank=None) -> bool:
        """Return True if the current router_replay_action is REPLAY_FORWARD for the local router instances.

        This inspects the first local RouterReplay instance's router_replay_action and compares it to
        RouterReplayAction.REPLAY_FORWARD.
        """
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return (router_instances_list
                and router_instances_list[0].router_replay_action == RouterReplayAction.REPLAY_FORWARD)

    @staticmethod
    def is_replay_backward_action(tf_config, vp_rank=None) -> bool:
        """Return True if the current router_replay_action is REPLAY_BACKWARD for the local router instances.

        This inspects the first local RouterReplay instance's router_replay_action and compares it to
        RouterReplayAction.REPLAY_BACKWARD.
        """
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return (router_instances_list
                and router_instances_list[0].router_replay_action == RouterReplayAction.REPLAY_BACKWARD)


def apply_router_replay_patch():
    """
    Applies the monkey patch for MoE Router Replay functionality.
    """
    logger.info('Applying Router Replay Patch...')

    assert ROUTER_REPLAY_AVAILABLE, \
        'The routing replay is not supported. Please upgrade megatron-core to 0.16.0 or higher'

    # Patch MoEAlltoAllTokenDispatcher.preprocess to handle router replay
    # When router replay is enabled, duplicate indices in top_indices can cause
    # routing_map.sum() < num_tokens * topk, leading to split size mismatch in alltoall.
    if MoEAlltoAllTokenDispatcher is not None and not hasattr(MoEAlltoAllTokenDispatcher, '_preprocess_patched'):
        original_preprocess = MoEAlltoAllTokenDispatcher.preprocess

        def patched_preprocess(self, routing_map):
            """Patched preprocess that handles router replay correctly for alltoall dispatcher."""
            # Call original preprocess
            result = original_preprocess(self, routing_map)
            # Fix num_out_tokens when router replay is enabled
            if (getattr(self.config, 'moe_enable_routing_replay', False) and not self.drop_and_pad
                    and self.config.moe_expert_capacity_factor is None
                    and not (getattr(self.config, 'moe_router_padding_for_quantization', None)
                             or getattr(self.config, 'moe_router_padding_for_fp8', None))):
                # With router replay, duplicate indices can reduce the actual routed
                # token count, so derive it from the routing map instead.
                self.num_out_tokens = int(routing_map.sum().item())
            return result

        MoEAlltoAllTokenDispatcher.preprocess = patched_preprocess
        MoEAlltoAllTokenDispatcher._preprocess_patched = True
