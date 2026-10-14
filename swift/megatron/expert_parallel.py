# Copyright (c) ModelScope Contributors. All rights reserved.
"""Compatibility with Megatron's gradient-statistics fixes (#5916 and #3213)."""
import inspect
import sys
from functools import wraps
from packaging import version

from swift.utils import get_logger

logger = get_logger()


def _legacy_expert_topology():
    import megatron.core

    return version.parse(megatron.core.__version__) < version.parse('0.19.0')


def _patch_uninitialized_tp_weight(layer, partition_dim):
    original = layer.__init__
    if getattr(original, '_swift_tp_metadata', False):
        return
    signature = inspect.signature(original)

    @wraps(original)
    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        weight = getattr(self, 'weight', None)
        if weight is not None and not hasattr(weight, 'tensor_model_parallel'):
            from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes

            # Checkpoint loading skips initialization, but these weights are still
            # independent TP shards. Preserve the constructor's stride (e.g. GLU=2).
            stride = signature.bind(self, *args, **kwargs).arguments.get('stride', 1)
            set_tensor_model_parallel_attributes(weight, True, partition_dim, stride)

    init._swift_tp_metadata = True
    layer.__init__ = init


def _patch_duplicated_te_parameters(layer):
    original = layer.__init__
    if getattr(original, '_swift_duplicated_metadata', False):
        return
    signature = inspect.signature(original)

    @wraps(original)
    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        arguments = signature.bind(self, *args, **kwargs).arguments
        if arguments.get('parallel_mode') == 'duplicated' and not arguments.get('is_expert', False):
            # TE only sees a local linear; MCore knows these are ordinary TP replicas.
            # Upstream #3213 restores this flag starting with MCore 0.17.
            for param in self.parameters():
                param.tensor_model_parallel = False

    init._swift_duplicated_metadata = True
    layer.__init__ = init


def patch_expert_parallel():
    import megatron.core
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.tensor_parallel import layers

    original_filter = layers.param_is_not_tensor_parallel_duplicate
    if 'expert_tp_group' not in inspect.signature(original_filter).parameters:

        def param_is_not_tensor_parallel_duplicate(param, tp_group=None, expert_tp_group=None):
            """Return True when this rank's parameter contributes to gradient statistics.

            param carries shard/topology metadata; tp_group and expert_tp_group identify
            ordinary/expert tensor replicas. False excludes a duplicate from statistics only.
            """
            # Independent shards all contribute; replicas are counted on one TP/ETP rank.
            if getattr(param, 'tensor_model_parallel', False):
                return True
            if not getattr(param, 'allreduce', True):
                if expert_tp_group is None:
                    expert_tp_group = parallel_state.get_expert_tensor_parallel_group()
                tp_group = expert_tp_group
            rank = tp_group.rank() if tp_group is not None else parallel_state.get_tensor_model_parallel_rank()
            return rank == 0

        # Some callers imported the function before Swift initialized Megatron.
        for name, module in tuple(sys.modules.items()):
            if module is None or not name.startswith(('megatron.', 'mindspeed.')):
                continue
            if getattr(module, 'param_is_not_tensor_parallel_duplicate', None) is original_filter:
                module.param_is_not_tensor_parallel_duplicate = param_is_not_tensor_parallel_duplicate
        logger.info('Backported Megatron expert TP gradient-statistics filtering (#5916).')

    if not _legacy_expert_topology():
        return
    if version.parse(megatron.core.__version__) < version.parse('0.17.0'):
        from megatron.core.extensions.transformer_engine import TELinear

        _patch_duplicated_te_parameters(TELinear)
    native_layers = (
        (layers.VocabParallelEmbedding, 0),
        (layers.ColumnParallelLinear, 0),
        (layers.RowParallelLinear, 1),
    )
    for layer, dim in native_layers:
        _patch_uninitialized_tp_weight(layer, dim)
    original_copy = layers.copy_tensor_model_parallel_attributes
    if getattr(original_copy, '_swift_expert_metadata', False):
        return

    @wraps(original_copy)
    def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
        original_copy(destination_tensor, source_tensor)
        if hasattr(source_tensor, 'allreduce'):
            destination_tensor.allreduce = source_tensor.allreduce

    copy_tensor_model_parallel_attributes._swift_expert_metadata = True
    layers.copy_tensor_model_parallel_attributes = copy_tensor_model_parallel_attributes
    tensor_parallel.copy_tensor_model_parallel_attributes = copy_tensor_model_parallel_attributes


def prepare_expert_parallel_parameters(model):
    if not _legacy_expert_topology():
        return
    from megatron.core.transformer.moe.moe_layer import BaseMoELayer

    for module in model.modules():
        if not isinstance(module, BaseMoELayer) or module.experts is None:
            continue
        config = module.config
        etp = config.expert_tensor_parallel_size or config.tensor_model_parallel_size
        use_expert_groups = config.expert_model_parallel_size > 1 or etp != config.tensor_model_parallel_size
        # Expert ownership works for GPU TE, MindSpeed and native expert implementations,
        # including adapters. Router/shared-expert params belong to ordinary TP/DP.
        # Legacy constructors only check EP, missing EP=1 with ETP different from TP.
        for param in module.experts.parameters():
            param.allreduce = not use_expert_groups
