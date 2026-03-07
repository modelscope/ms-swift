"""
Router Replay Utilities
Utilities for handling router replay functionality in Megatron models.
"""

import torch
from megatron.core import mpu
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from swift.megatron.trainers.utils import split_cp_inputs
from swift.megatron.utils import RouterReplay, RouterReplayAction
from swift.utils.torch_utils import get_current_device

device_name = get_current_device()


def get_local_layer_range(tf_config, vp_rank=None):
    vp_size = tf_config.virtual_pipeline_model_parallel_size
    if vp_size is not None:
        vp_rank = 0 if vp_rank is None else vp_rank
        offset = 0
        for pre_vp_stage in range(vp_size):
            if pre_vp_stage == vp_rank:
                break
            num_layers_to_build = get_num_layers_to_build(tf_config, pre_vp_stage)
            offset += num_layers_to_build
    else:
        offset = 0
    count = get_num_layers_to_build(tf_config, vp_rank)
    return offset, count


def get_local_topk_idx_for_current_rank(global_topk_idx, tf_config, packed_seq_params=None):
    if global_topk_idx is None:
        return None
    # 1. pp slice
    layer_offset = get_transformer_layer_offset(tf_config, vp_stage=0)
    offset, count = get_local_layer_range(tf_config, tf_config.virtual_pipeline_model_parallel_size)
    num_layers = offset + count
    local_topk_idx = torch.narrow(global_topk_idx, dim=2, start=layer_offset, length=num_layers)
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
    router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
    offset, _ = get_local_layer_range(tf_config, vp_rank)
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
