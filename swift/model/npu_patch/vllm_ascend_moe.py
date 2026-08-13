# Copyright (c) ModelScope Contributors. All rights reserved.
"""vLLM-Ascend MoE patches used by SWIFT NPU rollout.

There are two independent responsibilities in this file:

* runtime routing: avoid the unstable custom non-quantized MoE routing op on
  stacks where vLLM-Ascend still dispatches that branch to
  ``aclnnMoeInitRoutingCustom``;
* weight sync: adapt 2D HF/Megatron MoE expert weights to the already-processed
  3D vLLM-Ascend expert parameter layout during GRPO colocate updates.

Both patches are guarded by vLLM-Ascend implementation checks and only touch the
specific MoE paths they need.
"""
from __future__ import annotations

import inspect
import torch

from swift.utils.logger import get_logger

logger = get_logger()

_VLLM_ASCEND_MOE_PROCESSED_WEIGHT_LOADED_ATTR = '_swift_vllm_ascend_moe_processed_weight_loaded'
_VLLM_ASCEND_MOE_POST_LOAD_PATCHED_ATTR = '_swift_vllm_ascend_moe_post_load_patched'


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

    Do not probe the custom op by calling it first.  On Ascend, a missing custom
    binary can be reported asynchronously: even if Python catches the immediate
    RuntimeError and falls back, the failed launch can poison the stream and hang
    later at an unrelated event synchronization.  Therefore, when source
    inspection shows that the non-quantized branch still routes to the custom op,
    dispatch that branch directly to ``torch_npu.npu_moe_init_routing_v2``.
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
    origin_signature = inspect.signature(origin_routing)
    routing_defaults = {
        'scale': None,
        'active_num': None,
        'expert_num': None,
        'expert_tokens_num_type': 1,
        'expert_tokens_num_flag': True,
        'active_expert_range': None,
        'quant_mode': -1,
    }
    missing_params = set(routing_defaults).difference(origin_signature.parameters)
    if missing_params:
        raise RuntimeError('Unsupported vLLM-Ascend npu_moe_init_routing signature: '
                           f'signature={origin_signature}, missing={sorted(missing_params)}.')

    def is_nonquant_routing(routing_kwargs) -> bool:
        return routing_kwargs['scale'] is None and routing_kwargs['quant_mode'] == -1

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
        try:
            bound = origin_signature.bind(hidden_states, topk_ids, *args, **kwargs)
        except TypeError as e:
            raise RuntimeError('Failed to bind vLLM-Ascend npu_moe_init_routing arguments: '
                               f'signature={origin_signature}, args={args}, kwargs={kwargs}.') from e
        bound.apply_defaults()
        routing_kwargs = {key: bound.arguments.get(key, default) for key, default in routing_defaults.items()}

        if not is_nonquant_routing(routing_kwargs):
            return origin_routing(hidden_states, topk_ids, *args, **kwargs)
        logger.warning_once(
            'Using torch_npu.npu_moe_init_routing_v2 for vLLM-Ascend non-quantized MoE routing. '
            'The installed vLLM-Ascend implementation still dispatches this branch to '
            'npu_moe_init_routing_custom, whose missing custom-op binary fails asynchronously on this stack.')
        return npu_moe_init_routing_v2(hidden_states, topk_ids, routing_kwargs)

    patched_npu_moe_init_routing._swift_nonquant_routing_patched = True
    patched_npu_moe_init_routing._swift_origin = origin_routing
    adaptor_cls.npu_moe_init_routing = staticmethod(patched_npu_moe_init_routing)


def _patch_vllm_ascend_moe_sleep_layout(worker_cls=None) -> None:
    """Keep processed MoE parameters in their runtime layout after wake-up.

    vLLM-Ascend 0.18 transposes unquantized ``w13_weight``/``w2_weight``
    parameters in ``NPUWorker.wake_up``.  This prepares the parameters for a
    checkpoint-format full-weight reload, but corrupts inference when the
    caller only reloads a LoRA adapter: untouched routed-expert weights remain
    in the checkpoint layout and grouped matmul observes a hidden-size
    mismatch.

    The allocator restores the original parameter storage before that
    transpose.  Preserve and reinstall those original Parameters so sleep/wake
    itself is layout-neutral.  SWIFT's full-weight sync already handles both
    processed and preprocessed expert layouts explicitly.
    """
    if worker_cls is None:
        try:
            from vllm_ascend.worker.worker import NPUWorker
        except (ImportError, AttributeError):
            return
        worker_cls = NPUWorker

    origin_wake_up = getattr(worker_cls, 'wake_up', None)
    if origin_wake_up is None or getattr(origin_wake_up, '_swift_moe_sleep_layout_patched', False):
        return

    try:
        origin_source = inspect.getsource(origin_wake_up)
    except (OSError, TypeError):
        return
    layout_rewrite_tokens = ('w2_weight', 'w13_weight', 'transpose(1, 2)', 'setattr(parent_module, param_name')
    if not all(token in origin_source for token in layout_rewrite_tokens):
        return

    def wake_up(self, tags=None):
        restore_weights = tags is None or 'weights' in tags
        model = self.model_runner.model
        saved_moe_parameters = {}
        if restore_weights:
            saved_moe_parameters = {
                name: param
                for name, param in model.named_parameters() if 'w13_weight' in name or 'w2_weight' in name
            }

        result = origin_wake_up(self, tags=tags)

        restored = 0
        for name, original_param in saved_moe_parameters.items():
            try:
                current_param = model.get_parameter(name)
            except AttributeError:
                continue
            expected_transposed_shape = (
                original_param.shape[:-2] + (original_param.shape[-1], original_param.shape[-2]))
            if current_param is original_param or current_param.shape != expected_transposed_shape:
                continue
            parts = name.split('.')
            parent_module = model.get_submodule('.'.join(parts[:-1]))
            setattr(parent_module, parts[-1], original_param)
            restored += 1

        if restored:
            logger.warning_once(
                f'Preserved the vLLM-Ascend runtime layout of {restored} FusedMoE parameters across sleep/wake.')
        return result

    wake_up._swift_origin = origin_wake_up
    wake_up._swift_moe_sleep_layout_patched = True
    worker_cls.wake_up = wake_up


def patch_vllm_ascend_moe_runtime() -> None:
    """Apply MoE runtime patches that are independent of GRPO weight sync."""
    _patch_vllm_ascend_device_op_nonquant_routing()
    _patch_vllm_ascend_moe_sleep_layout()


def _is_vllm_ascend_unquantized_fused_moe_method(quant_method) -> bool:
    """Return whether the layer uses Ascend's unquantized FusedMoE method."""
    quant_method_module = type(quant_method).__module__ if quant_method is not None else ''
    if not quant_method_module.startswith('vllm_ascend'):
        return False
    try:
        from vllm.model_executor.layers.fused_moe import UnquantizedFusedMoEMethod
    except (ImportError, AttributeError):
        return False
    return isinstance(quant_method, UnquantizedFusedMoEMethod)


def _patch_vllm_ascend_moe_post_load(experts) -> None:
    """Skip post-load only for an expert layer already written in runtime layout."""
    quant_method = getattr(experts, 'quant_method', None)
    if not _is_vllm_ascend_unquantized_fused_moe_method(quant_method):
        return
    if getattr(quant_method, _VLLM_ASCEND_MOE_POST_LOAD_PATCHED_ATTR, False):
        return

    origin_process_weights = getattr(quant_method, 'process_weights_after_loading', None)
    if origin_process_weights is None:
        return

    def process_weights_after_loading(layer):
        if getattr(layer, _VLLM_ASCEND_MOE_PROCESSED_WEIGHT_LOADED_ATTR, False):
            setattr(layer, _VLLM_ASCEND_MOE_PROCESSED_WEIGHT_LOADED_ATTR, False)
            return
        return origin_process_weights(layer)

    quant_method.process_weights_after_loading = process_weights_after_loading
    setattr(quant_method, _VLLM_ASCEND_MOE_POST_LOAD_PATCHED_ATTR, True)


def expand_fused_moe_expert_names_for_vllm_ascend(name: str):
    """Map Transformers fused Qwen MoE expert names to vLLM checkpoint names.

    FSDP2 can expose Qwen-style MoE expert weights as fused tensors:

        mlp.experts.gate_up_proj: [experts, 2 * intermediate, hidden]
        mlp.experts.down_proj   : [experts, hidden, intermediate]

    vLLM's Qwen MoE ``load_weights`` path expects checkpoint-style names such as
    ``mlp.experts.0.gate_proj.weight`` / ``up_proj`` / ``down_proj`` and maps
    those names onto its internal ``w13_weight`` / ``w2_weight`` parameters.
    Use expert 0 only as a name anchor; the paired vLLM-Ascend weight-loader
    patch below copies all local experts from the full 3D tensor.
    """
    gate_up_suffix = '.mlp.experts.gate_up_proj'
    down_suffix = '.mlp.experts.down_proj'
    if name.endswith(gate_up_suffix):
        prefix = name[:-len('gate_up_proj')]
        return [
            f'{prefix}0.gate_proj.weight',
            f'{prefix}0.up_proj.weight',
        ]
    if name.endswith(down_suffix):
        prefix = name[:-len('down_proj')]
        return [f'{prefix}0.down_proj.weight']
    return None


def expand_fused_moe_expert_weight_for_vllm_ascend(name: str, param):
    """Expand one FSDP2 fused Qwen MoE expert tensor for vLLM-Ascend weight sync."""
    if not isinstance(param, torch.Tensor) or param.dim() != 3:
        return None
    expanded_names = expand_fused_moe_expert_names_for_vllm_ascend(name)
    if expanded_names is None:
        return None
    if name.endswith('.mlp.experts.gate_up_proj'):
        gate_proj, up_proj = param.chunk(2, dim=1)
        return [
            (expanded_names[0], gate_proj.contiguous()),
            (expanded_names[1], up_proj.contiguous()),
        ]
    if name.endswith('.mlp.experts.down_proj'):
        return [(expanded_names[0], param)]
    return None


def patch_vllm_ascend_moe_expert_weight_loader(experts,
                                               name: str,
                                               param,
                                               *,
                                               load_preprocessed_weight: bool = False) -> None:
    """Patch one processed vLLM-Ascend MoE expert parameter loader.

    vLLM-Ascend transposes unquantized MoE weights after each model load
    so grouped matmul can consume them efficiently.  During GRPO weight sync,
    however, SWIFT can send regular HF/Megatron expert weights, for example:

        gate_proj/up_proj: [intermediate, hidden] -> w13_weight
        down_proj       : [hidden, intermediate] -> w2_weight

    FSDP2 Qwen MoE may expose the same weights as fused 3D tensors.  SWIFT
    expands those tensors to checkpoint-style gate/up/down names before calling
    vLLM ``load_weights``:

        gate_proj/up_proj: [experts, intermediate, hidden]
        down_proj       : [experts, hidden, intermediate]

    Full-weight server reload still writes the pre-processed layout and then
    calls ``process_weights_after_loading`` once, letting vLLM-Ascend transpose
    complete weights afterwards:

        w13_weight before process: [local_experts, 2 * intermediate_per_tp, hidden]
        w2_weight before process : [local_experts, hidden, intermediate_per_tp]

    Megatron colocate runtime sync loads into the already-processed layout used
    by the existing Megatron rollout path:

        w13_weight after process: [local_experts, hidden, 2 * intermediate_per_tp]
        w2_weight after process : [local_experts, intermediate_per_tp, hidden]

    ``load_preprocessed_weight`` selects the server full-reload target.
    Colocate runtime sync keeps the processed target because current
    vLLM-Ascend non-quantized grouped matmul consumes the [hidden, I_tp]
    direction in this path.  A layer-local marker is set only after this
    wrapper actually copies such a tensor.  Its unquantized FusedMoE post-load
    wrapper consumes that marker and skips only the redundant transpose for
    this expert layer; all other model post-load processing still runs.

    This wrapper keeps the normal vLLM loader for initial checkpoint load,
    quantized experts, and non-Ascend backends.  It only handles the 3D
    vLLM-Ascend expert tensors when a 2D or fused 3D runtime-sync tensor is
    loaded into ``w13_weight`` or ``w2_weight``.
    """
    if 'w13_weight' not in name and 'w2_weight' not in name:
        return
    if not _is_vllm_ascend_unquantized_fused_moe_method(getattr(experts, 'quant_method', None)):
        return
    _patch_vllm_ascend_moe_post_load(experts)
    setattr(experts, _VLLM_ASCEND_MOE_PROCESSED_WEIGHT_LOADED_ATTR, False)

    def make_ascend_moe_weight_loader(experts, origin_weight_loader):

        def load_processed_ascend_weight(param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
            # Only the GRPO runtime-sync path needs special handling here.
            # SWIFT provides HF/Megatron tensors, while vLLM-Ascend stores MoE
            # experts as 3D per-local-expert tensors.  Initial checkpoint load
            # and other layouts continue to use the original vLLM loader.
            is_runtime_sync_into_processed_param = param.data.dim() == 3 and loaded_weight.dim() in {2, 3}
            if not is_runtime_sync_into_processed_param:
                return origin_weight_loader(param, loaded_weight, weight_name, shard_id, expert_id, return_success)

            is_w13_shard = shard_id in {'w1', 'w3'} and 'w13_weight' in weight_name
            is_w2_shard = shard_id == 'w2' and 'w2_weight' in weight_name

            loaded_expert_sample = loaded_weight[0] if loaded_weight.dim() == 3 else loaded_weight

            def prepare_preprocessed_target_layout():
                """Write weights before vLLM-Ascend post-load processing."""
                if is_w13_shard and param.data.shape[1] == loaded_expert_sample.shape[-1]:
                    param.data = param.data.transpose(1, 2).contiguous()
                elif is_w2_shard and param.data.shape[2] == loaded_expert_sample.shape[0]:
                    param.data = param.data.transpose(1, 2).contiguous()

            def prepare_processed_target_layout():
                """Write weights into the vLLM-Ascend runtime layout."""
                if (is_w13_shard and param.data.shape[-1] == loaded_expert_sample.shape[-1]
                        and param.data.shape[-2] != loaded_expert_sample.shape[-1]):
                    param.data = param.data.transpose(1, 2).contiguous()
                elif (is_w2_shard and param.data.shape[-2] == loaded_expert_sample.shape[0]
                      and param.data.shape[-1] != loaded_expert_sample.shape[0]):
                    param.data = param.data.transpose(1, 2).contiguous()

            tp_rank = experts.tp_rank

            def copy_preprocessed_expert(local_expert_id: int, loaded_expert_weight) -> bool:
                """Copy expert weights into the pre-process vLLM-Ascend layout."""
                param_data = param.data[local_expert_id]
                if is_w13_shard:
                    # Target: [2 * intermediate_per_tp, hidden].
                    shard_size = param_data.shape[0] // 2
                    loaded_expert_weight = loaded_expert_weight.narrow(0, shard_size * tp_rank, shard_size)
                    offset = 0 if shard_id == 'w1' else shard_size
                    param_data[offset:offset + shard_size].copy_(loaded_expert_weight.contiguous())
                    return True

                if is_w2_shard:
                    # Target: [hidden, intermediate_per_tp].
                    shard_size = param_data.shape[1]
                    loaded_expert_weight = loaded_expert_weight.narrow(1, shard_size * tp_rank, shard_size)
                    param_data.copy_(loaded_expert_weight.contiguous())
                    return True

                return False

            def copy_processed_expert(local_expert_id: int, loaded_expert_weight) -> bool:
                """Copy expert shards into the processed vLLM-Ascend layout."""
                param_data = param.data[local_expert_id]
                if is_w13_shard:
                    # Target: [hidden, 2 * intermediate_per_tp].
                    shard_size = param_data.shape[1] // 2
                    loaded_expert_weight = loaded_expert_weight.narrow(0, shard_size * tp_rank, shard_size)
                    offset = 0 if shard_id == 'w1' else shard_size
                    param_data[:, offset:offset + shard_size].copy_(loaded_expert_weight.transpose(0, 1).contiguous())
                    return True

                if is_w2_shard:
                    # Target: [intermediate_per_tp, hidden].
                    shard_size = param_data.shape[0]
                    loaded_expert_weight = loaded_expert_weight.narrow(1, shard_size * tp_rank, shard_size)
                    param_data.copy_(loaded_expert_weight.transpose(0, 1).contiguous())
                    return True

                return False

            if load_preprocessed_weight:
                prepare_preprocessed_target_layout()
                copy_one_expert = copy_preprocessed_expert
            else:
                prepare_processed_target_layout()
                copy_one_expert = copy_processed_expert

            if loaded_weight.dim() == 3:
                copied = False
                for global_expert_id, loaded_expert_weight in enumerate(loaded_weight):
                    local_expert_id = experts._map_global_expert_id_to_local_expert_id(global_expert_id)
                    if local_expert_id == -1:
                        continue
                    copied = copy_one_expert(local_expert_id, loaded_expert_weight) or copied
                if copied and not load_preprocessed_weight:
                    setattr(experts, _VLLM_ASCEND_MOE_PROCESSED_WEIGHT_LOADED_ATTR, True)
                return copied if return_success else None

            local_expert_id = experts._map_global_expert_id_to_local_expert_id(expert_id)
            if local_expert_id == -1:
                return False if return_success else None

            if copy_one_expert(local_expert_id, loaded_weight):
                if not load_preprocessed_weight:
                    setattr(experts, _VLLM_ASCEND_MOE_PROCESSED_WEIGHT_LOADED_ATTR, True)
                return True if return_success else None

            return origin_weight_loader(param, loaded_weight, weight_name, shard_id, expert_id, return_success)

        load_processed_ascend_weight._swift_ascend_moe_weight_loader = True
        load_processed_ascend_weight._swift_origin_weight_loader = origin_weight_loader
        load_processed_ascend_weight._swift_load_preprocessed_weight = load_preprocessed_weight
        return load_processed_ascend_weight

    if not hasattr(experts, 'weight_loader'):
        return
    weight_loader = getattr(param, 'weight_loader', experts.weight_loader)
    origin_weight_loader = getattr(weight_loader, '_swift_origin_weight_loader', weight_loader)
    if (not getattr(weight_loader, '_swift_ascend_moe_weight_loader', False)
            or getattr(weight_loader, '_swift_load_preprocessed_weight', None) != load_preprocessed_weight):
        param.weight_loader = make_ascend_moe_weight_loader(experts, origin_weight_loader)


__all__ = [
    'expand_fused_moe_expert_names_for_vllm_ascend',
    'expand_fused_moe_expert_weight_for_vllm_ascend',
    'patch_vllm_ascend_moe_expert_weight_loader',
    'patch_vllm_ascend_moe_runtime',
]
