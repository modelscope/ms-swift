# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import inspect
import os
import sys
import torch
from contextlib import contextmanager
from importlib import metadata
from packaging import version
from typing import Optional

from swift.utils.logger import get_logger

logger = get_logger()

def _env_enabled(name: str, default: str = '1') -> bool:
    return os.environ.get(name, default) == '1'


def _get_package_version(package_name: str) -> Optional[version.Version]:
    try:
        return version.parse(metadata.version(package_name))
    except (metadata.PackageNotFoundError, version.InvalidVersion):
        return None


def patch_vllm_ascend_runtime() -> None:
    """Apply SWIFT runtime compatibility patches for vLLM-Ascend on NPU.

    The caller is responsible for guarding this function by device type.  This
    module intentionally imports vLLM/vLLM-Ascend modules lazily so CUDA/GPU
    paths do not enter Ascend-only code.
    """
    if not _env_enabled('SWIFT_PATCH_VLLM_ASCEND_RUNTIME'):
        return

    # Read versions for diagnostics and future version-gated branches.  The
    # concrete patches below still use symbol checks because local source builds
    # may report dev versions while carrying fixes from adjacent releases.
    vllm_version = _get_package_version('vllm')
    vllm_ascend_version = _get_package_version('vllm-ascend')
    logger.debug('Applying vLLM-Ascend runtime patches: vllm=%s, vllm-ascend=%s', vllm_version, vllm_ascend_version)

    _patch_flash_attn_optional_import()
    _patch_vllm_ascend_moe_comm_lazy_init()
    _patch_vllm_ascend_mem_get_info()
    _patch_vllm_ascend_memory_profiling()
    _patch_vllm_ascend_device_op_nonquant_routing()
    _patch_vllm_ascend_unquant_moe_layout()
    patch_vllm_ascend_colocate_runtime()


def patch_vllm_ascend_colocate_runtime() -> None:
    """Apply vLLM-Ascend patches needed by colocated training."""
    _patch_vllm_ascend_colocate_memory_profiling()


def _patch_flash_attn_optional_import() -> None:
    module = sys.modules.get('flash_attn')
    if module is None or hasattr(module, '__path__'):
        return
    for module_name in list(sys.modules):
        if module_name == 'flash_attn' or module_name.startswith('flash_attn.'):
            sys.modules.pop(module_name, None)


def _patch_vllm_ascend_moe_comm_lazy_init() -> None:
    try:
        from vllm_ascend.ascend_forward_context import MoECommType
        from vllm_ascend.ops.fused_moe import moe_comm_method
    except (ImportError, AttributeError):
        return
    if getattr(moe_comm_method, '_swift_lazy_moe_comm_patched', False):
        return

    required_classes = {
        MoECommType.ALLTOALL: 'AlltoAllCommImpl',
        MoECommType.ALLGATHER: 'AllGatherCommImpl',
        MoECommType.MC2: 'MC2CommImpl',
        MoECommType.FUSED_MC2: 'FusedMC2CommImpl',
    }
    if not all(hasattr(moe_comm_method, cls_name) for cls_name in required_classes.values()):
        return

    moe_comm_method._MoECommMethods = {}
    moe_comm_method._SwiftMoECommConfig = None

    def get_moe_comm_method(moe_comm_type):
        if moe_comm_type is None:
            return None
        moe_comm = moe_comm_method._MoECommMethods.get(moe_comm_type)
        if moe_comm is not None:
            return moe_comm
        assert moe_comm_method._SwiftMoECommConfig is not None, 'MoE communication method is not set up'
        moe_comm_cls = getattr(moe_comm_method, required_classes[moe_comm_type])
        moe_comm = moe_comm_cls(moe_comm_method._SwiftMoECommConfig)
        moe_comm_method._MoECommMethods[moe_comm_type] = moe_comm
        return moe_comm

    def setup_moe_comm_method(moe_config):
        moe_comm_method._SwiftMoECommConfig = moe_config
        moe_comm_method._MoECommMethods.clear()

    moe_comm_method.get_moe_comm_method = get_moe_comm_method
    moe_comm_method.setup_moe_comm_method = setup_moe_comm_method
    fused_moe = sys.modules.get('vllm_ascend.ops.fused_moe.fused_moe')
    if fused_moe is not None:
        fused_moe.setup_moe_comm_method = setup_moe_comm_method
    moe_comm_method._swift_lazy_moe_comm_patched = True


def _patch_vllm_ascend_mem_get_info() -> None:
    try:
        from vllm_ascend.platform import NPUPlatform
    except (ImportError, AttributeError):
        return
    if getattr(NPUPlatform, '_swift_mem_get_info_patched', False):
        return

    @classmethod
    def mem_get_info(cls, device=None):
        if device is None:
            return torch.npu.mem_get_info()
        try:
            return torch.npu.mem_get_info(device=device)
        except TypeError:
            return torch.npu.mem_get_info()

    NPUPlatform.mem_get_info = mem_get_info
    NPUPlatform._swift_mem_get_info_patched = True


def _patch_vllm_ascend_memory_profiling() -> None:
    try:
        from vllm.logger import logger as vllm_logger
        from vllm_ascend.worker import worker as ascend_worker
    except (ImportError, AttributeError, RuntimeError):
        return

    origin_memory_profiling = getattr(ascend_worker, 'memory_profiling', None)
    if origin_memory_profiling is None or getattr(origin_memory_profiling,
                                                 '_swift_ascend_memory_profiling_patched', False):
        return

    @contextmanager
    def memory_profiling(baseline_snapshot, weights_memory: int = 0):
        with origin_memory_profiling(baseline_snapshot, weights_memory=weights_memory) as profile_result:
            yield profile_result

        if profile_result.after_profile.free_memory < baseline_snapshot.free_memory:
            return

        # In colocated GRPO, the training side may release NPU memory while
        # vLLM-Ascend is profiling KV cache capacity. vLLM-Ascend assumes
        # external memory is stable and asserts when free memory increases.
        profile_result.after_profile.free_memory = max(baseline_snapshot.free_memory - 1, 0)
        profile_result.after_profile.cuda_memory = (
            profile_result.after_profile.total_memory - profile_result.after_profile.free_memory)
        profile_result.non_torch_increase = max(profile_result.non_torch_increase, 0)
        profile_result.torch_peak_increase = max(profile_result.torch_peak_increase, 0)
        profile_result.non_kv_cache_memory = (
            profile_result.non_torch_increase + profile_result.torch_peak_increase + profile_result.weights_memory)
        vllm_logger.warning_once(
            'Patched vLLM-Ascend memory profiling because free memory increased during profiling. '
            'This is expected in colocated training when another rank releases memory.')

    memory_profiling._swift_ascend_memory_profiling_patched = True
    memory_profiling._swift_origin = origin_memory_profiling
    ascend_worker.memory_profiling = memory_profiling


def _patch_vllm_ascend_device_op_nonquant_routing() -> None:
    """Use the stable torch-npu routing op for non-quantized MoE when needed.

    Some released vLLM-Ascend versions route the non-quantized MoE case
    (``scale is None`` and ``quant_mode == -1``) through
    ``npu_moe_init_routing_custom`` / ``aclnnMoeInitRoutingCustom``, which is
    not stable for the parameter combination used by Qwen-style MoE rollout.

    This is intentionally gated by implementation detection instead of a fixed
    version threshold: source builds or future/backported versions may already
    dispatch the non-quantized path to ``torch_npu.npu_moe_init_routing_v2``.
    When that fixed branch is present, skip patching and keep the upstream
    implementation intact.

    For versions that still use the custom op, probe it on the first
    non-quantized routing call and cache the result.  The probe synchronizes once
    because Ascend custom-op launch failures are asynchronous and may otherwise
    surface later in an unrelated operator.  If the custom op is available, keep
    using the original implementation.  If it raises a known routing-custom
    failure, fall back to ``torch_npu.npu_moe_init_routing_v2`` for later calls.
    Other errors are re-raised.
    """
    try:
        import torch_npu
        from vllm_ascend.device import device_op
    except (ImportError, AttributeError):
        return

    adaptor_cls = getattr(device_op, 'BaseDeviceAdaptor', None)
    if adaptor_cls is None:
        return
    origin_routing = getattr(adaptor_cls, 'npu_moe_init_routing', None)
    if origin_routing is None or getattr(origin_routing, '_swift_nonquant_routing_patched', False):
        return
    try:
        origin_source = inspect.getsource(origin_routing)
    except (OSError, TypeError):
        origin_source = ''
    if 'npu_moe_init_routing_v2' in origin_source and 'quant_mode == -1' in origin_source:
        return
    custom_routing_available = None

    def is_nonquant_routing(routing_kwargs) -> bool:
        return routing_kwargs['scale'] is None and routing_kwargs['quant_mode'] == -1

    def is_custom_routing_failure(error: RuntimeError) -> bool:
        message = str(error)
        return ('MoeInitRoutingCustom' in message or 'npu_moe_init_routing_custom' in message
                or 'aclnnMoeInitRoutingCustom' in message)

    def npu_moe_init_routing_v2(hidden_states, topk_ids, routing_kwargs):
        active_num = routing_kwargs['active_num']
        expert_num = routing_kwargs['expert_num']
        active_expert_range = routing_kwargs['active_expert_range']
        return torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            scale=None,
            offset=None,
            active_num=0 if active_num is None else active_num,
            expert_capacity=-1,
            expert_num=expert_num,
            drop_pad_mode=0,
            expert_tokens_num_type=routing_kwargs['expert_tokens_num_type'],
            expert_tokens_num_flag=routing_kwargs['expert_tokens_num_flag'],
            active_expert_range=[0, expert_num] if active_expert_range is None else active_expert_range,
            quant_mode=routing_kwargs['quant_mode'],
            row_idx_type=0,
        )

    def patched_npu_moe_init_routing(hidden_states, topk_ids, *args, **kwargs):
        routing_kwargs = {
            'scale': None,
            'active_num': None,
            'expert_num': None,
            'expert_tokens_num_type': 1,
            'expert_tokens_num_flag': True,
            'active_expert_range': None,
            'quant_mode': -1,
        }
        param_names = list(routing_kwargs)
        for index, value in enumerate(args):
            if index < len(param_names):
                routing_kwargs[param_names[index]] = value
        routing_kwargs.update(kwargs)

        nonlocal custom_routing_available
        if not is_nonquant_routing(routing_kwargs):
            return origin_routing(hidden_states, topk_ids, *args, **kwargs)
        if custom_routing_available is False:
            return npu_moe_init_routing_v2(hidden_states, topk_ids, routing_kwargs)
        if custom_routing_available is True:
            return origin_routing(hidden_states, topk_ids, *args, **kwargs)

        try:
            result = origin_routing(hidden_states, topk_ids, *args, **kwargs)
            torch.npu.synchronize()
            custom_routing_available = True
            return result
        except RuntimeError as error:
            if not is_custom_routing_failure(error):
                raise
            custom_routing_available = False
            logger.warning_once(
                'Falling back to torch_npu.npu_moe_init_routing_v2 because '
                'vLLM-Ascend npu_moe_init_routing_custom failed in non-quantized MoE routing.')
            return npu_moe_init_routing_v2(hidden_states, topk_ids, routing_kwargs)

    patched_npu_moe_init_routing._swift_nonquant_routing_patched = True
    patched_npu_moe_init_routing._swift_origin = origin_routing
    adaptor_cls.npu_moe_init_routing = staticmethod(patched_npu_moe_init_routing)


def _patch_vllm_ascend_unquant_moe_layout() -> None:
    """Patch vLLM-Ascend unquantized gated-MoE layout before grouped matmul.

    Some vLLM-Ascend versions pass already-processed unquantized MoE weights
    into ``unquant_apply_mlp`` with ``need_trans=False``.  For Qwen-style gated
    MoE, ``w1`` is the concatenated gate/up projection:

        expected  w1: [experts, hidden, 2 * intermediate]
        expected  w2: [experts, intermediate, hidden]
        reversed  w1: [experts, 2 * intermediate, hidden]
        reversed  w2: [experts, hidden, intermediate]

    ``torch_npu.npu_grouped_matmul`` needs the K dimension to match the input
    hidden size.  If the pair is clearly the reversed gated-MoE layout, transpose
    both weights before calling the original implementation.  If the shapes do
    not match this gated-MoE pattern, do not guess; leave the original
    vLLM-Ascend path untouched.
    """
    if not _env_enabled('SWIFT_PATCH_VLLM_ASCEND_MOE_LAYOUT'):
        return
    try:
        from vllm_ascend.ops.fused_moe import moe_mlp
    except (ImportError, AttributeError, RuntimeError):
        return
    origin_apply_mlp = getattr(moe_mlp, 'unquant_apply_mlp', None)
    if origin_apply_mlp is None or getattr(origin_apply_mlp, '_swift_ascend_moe_layout_patched', False):
        return

    def _is_expected_gated_unquant_moe_layout(w1, w2, input_k: int) -> bool:
        return (w1.shape[1] == input_k and w2.shape[2] == input_k and w1.shape[2] % 2 == 0
                and w1.shape[2] // 2 == w2.shape[1])

    def _is_reversed_gated_unquant_moe_layout(w1, w2, input_k: int) -> bool:
        return (w1.shape[2] == input_k and w2.shape[1] == input_k and w1.shape[1] % 2 == 0
                and w1.shape[1] // 2 == w2.shape[2])

    def patched_unquant_apply_mlp(hidden_states, w1, w2, *args, **kwargs):
        need_trans = kwargs.get('need_trans', args[6] if len(args) > 6 else True)
        if not need_trans and w1.dim() == 3 and w2.dim() == 3:
            input_k = hidden_states.shape[-1]
            # This compatibility branch is only for gated MoE layouts where
            # w1 stores concatenated gate/up projections:
            #   w1=[experts, hidden, 2*intermediate],
            #   w2=[experts, intermediate, hidden].
            # Other MoE implementations may not have the 2x relation, so leave
            # them to vLLM-Ascend's original implementation instead of guessing.
            if _is_expected_gated_unquant_moe_layout(w1, w2, input_k):
                pass
            elif _is_reversed_gated_unquant_moe_layout(w1, w2, input_k):
                w1 = w1.transpose(1, 2)
                w2 = w2.transpose(1, 2)
        return origin_apply_mlp(hidden_states, w1, w2, *args, **kwargs)

    patched_unquant_apply_mlp._swift_ascend_moe_layout_patched = True
    patched_unquant_apply_mlp._swift_origin = origin_apply_mlp
    moe_mlp.unquant_apply_mlp = patched_unquant_apply_mlp


def _patch_vllm_ascend_colocate_memory_profiling() -> None:
    try:
        from vllm.utils.mem_constants import GiB_bytes
        from vllm.utils.mem_utils import memory_profiling
        from vllm_ascend.worker import worker as ascend_worker
    except (ImportError, AttributeError, RuntimeError):
        return

    NPUWorker = getattr(ascend_worker, 'NPUWorker', None)
    if NPUWorker is None:
        return
    origin_determine = getattr(NPUWorker, 'determine_available_memory', None)
    if origin_determine is None or getattr(origin_determine, '_swift_colocate_memory_patched', False):
        return

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        GiB = lambda b: b / GiB_bytes
        with memory_profiling(
                self.init_snapshot,
                weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()

        free_gpu_memory = profile_result.after_profile.free_memory
        if self.init_snapshot.free_memory <= free_gpu_memory:
            ascend_worker.logger.warning(
                'vLLM-Ascend memory profiling observed increased free memory '
                'during colocate initialization: initial %.2f GiB, current %.2f GiB. '
                'Continuing with profiled non-KV memory instead of failing.',
                GiB(self.init_snapshot.free_memory), GiB(free_gpu_memory))

        self.available_kv_cache_memory_bytes = self.requested_memory - profile_result.non_kv_cache_memory
        ascend_worker.logger.debug(profile_result)
        ascend_worker.logger.info_once(
            'Available KV cache memory: %.2f GiB', GiB(self.available_kv_cache_memory_bytes), scope='local')
        return int(self.available_kv_cache_memory_bytes)

    determine_available_memory._swift_colocate_memory_patched = True
    determine_available_memory._swift_origin = origin_determine
    NPUWorker.determine_available_memory = determine_available_memory


def patch_vllm_ascend_moe_expert_weight_loader(experts, name: str, param) -> None:
    """Patch one vLLM-Ascend processed MoE expert parameter loader.

    vLLM-Ascend transposes unquantized MoE weights after the initial model load
    so grouped matmul can consume them efficiently.  During GRPO colocate weight
    sync, however, SWIFT still sends regular 2D HF/Megatron weights, for example:

        gate_proj/up_proj: [intermediate, hidden] -> w13_weight
        down_proj       : [hidden, intermediate] -> w2_weight

    The target vLLM-Ascend parameters are already processed 3D tensors, e.g.:

        w13_weight: [local_experts, hidden, 2 * intermediate_per_tp]
        w2_weight : [local_experts, intermediate_per_tp, hidden]

    This wrapper keeps the normal vLLM loader for ordinary cases, and only
    handles the processed 3D vLLM-Ascend layout when a 2D runtime-sync tensor is
    loaded into w13_weight or w2_weight.
    """
    if 'w13_weight' not in name and 'w2_weight' not in name:
        return
    quant_method = getattr(experts, 'quant_method', None)
    quant_method_module = type(quant_method).__module__ if quant_method is not None else ''
    if not quant_method_module.startswith('vllm_ascend'):
        return

    def make_ascend_moe_weight_loader(experts, origin_weight_loader):

        def load_processed_ascend_weight(param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
            quant_method = getattr(experts, 'quant_method', None)
            quant_method_module = type(quant_method).__module__ if quant_method is not None else ''
            # Only the GRPO runtime-sync path needs special handling here:
            # SWIFT provides normal 2D HF/Megatron tensors, while the target
            # vLLM-Ascend MoE parameter has already been converted to a 3D
            # per-local-expert layout.  Initial checkpoint load and other
            # layouts continue to use the original vLLM loader.
            is_runtime_sync_into_processed_param = (param.data.dim() == 3 and loaded_weight.dim() == 2
                                                    and quant_method_module.startswith('vllm_ascend'))
            if not is_runtime_sync_into_processed_param:
                return origin_weight_loader(param, loaded_weight, weight_name, shard_id, expert_id, return_success)

            is_w13_shard = shard_id in {'w1', 'w3'} and 'w13_weight' in weight_name
            is_w2_shard = shard_id == 'w2' and 'w2_weight' in weight_name

            # Runtime sync may see a Parameter whose data was restored to the
            # pre-processed orientation.  Rebuild the vLLM-Ascend orientation
            # before copying the incoming HF/Megatron 2D shard.
            if is_w13_shard:
                if param.data.shape[-1] == loaded_weight.shape[-1] and param.data.shape[-2] != loaded_weight.shape[-1]:
                    param.data = param.data.transpose(1, 2).contiguous()
            elif is_w2_shard:
                if param.data.shape[-2] == loaded_weight.shape[0] and param.data.shape[-1] != loaded_weight.shape[0]:
                    param.data = param.data.transpose(1, 2).contiguous()

            local_expert_id = experts._map_global_expert_id_to_local_expert_id(expert_id)
            if local_expert_id == -1:
                return False if return_success else None

            tp_rank = experts.tp_rank
            param_data = param.data[local_expert_id]

            if is_w13_shard:
                # Example:
                #   loaded gate/up shard: [intermediate, hidden]
                #   target w13 slot    : [hidden, 2 * intermediate_per_tp]
                #
                # TP slices the intermediate dimension.  w1 occupies the first
                # half of w13_weight and w3 occupies the second half, so copy a
                # transposed TP slice into the selected half.
                shard_size = param_data.shape[1] // 2
                loaded_weight = loaded_weight.narrow(0, shard_size * tp_rank, shard_size)
                offset = 0 if shard_id == 'w1' else shard_size
                param_data[:, offset:offset + shard_size].copy_(loaded_weight.transpose(0, 1).contiguous())
                return True if return_success else None

            if is_w2_shard:
                # Example:
                #   loaded down shard: [hidden, intermediate]
                #   target w2 slot  : [intermediate_per_tp, hidden]
                #
                # TP slices the intermediate dimension on loaded_weight dim 1;
                # vLLM-Ascend stores the processed local shard transposed.
                shard_size = param_data.shape[0]
                loaded_weight = loaded_weight.narrow(1, shard_size * tp_rank, shard_size)
                param_data.copy_(loaded_weight.transpose(0, 1).contiguous())
                return True if return_success else None

            return origin_weight_loader(param, loaded_weight, weight_name, shard_id, expert_id, return_success)

        load_processed_ascend_weight._swift_ascend_moe_weight_loader = True
        return load_processed_ascend_weight

    if not hasattr(experts, 'weight_loader'):
        return
    weight_loader = getattr(param, 'weight_loader', experts.weight_loader)
    if not getattr(weight_loader, '_swift_ascend_moe_weight_loader', False):
        param.weight_loader = make_ascend_moe_weight_loader(experts, weight_loader)


__all__ = [
    'patch_vllm_ascend_colocate_runtime',
    'patch_vllm_ascend_moe_expert_weight_loader',
    'patch_vllm_ascend_runtime',
]
