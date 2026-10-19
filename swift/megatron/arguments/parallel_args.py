# Copyright (c) ModelScope Contributors. All rights reserved.
"""Basic parallel-size checks, without initializing a training backend."""
import os


def get_parallel_world_size(skip_megatron_init=False, distributed=None):
    """Return the training world size, or None for a driver without a known size."""
    if skip_megatron_init:
        return None
    if distributed is not None and distributed.is_available() and distributed.is_initialized():
        return distributed.get_world_size()
    for name in ('WORLD_SIZE', '_PATCH_WORLD_SIZE'):
        value = os.environ.get(name)
        # Keep the same empty-value fallback as swift.utils.env.get_dist_setting.
        if value:
            try:
                return int(value)
            except ValueError:
                raise ValueError(f'{name} must be a positive integer, got {value!r}.') from None
    # Do not mistake a launcher/driver for a single training worker.
    if any(
            name in os.environ
            for name in ('RANK', 'LOCAL_RANK', 'NPROC_PER_NODE', 'NNODES', 'LOCAL_WORLD_SIZE', 'LOCAL_SIZE')):
        return None
    return 1


def validate_parallel_args(args, world_size=None):
    """Check the sizes used by SWIFT's initialize_model_parallel call."""
    names = (
        'tensor_model_parallel_size',
        'pipeline_model_parallel_size',
        'context_parallel_size',
        'expert_model_parallel_size',
        'expert_tensor_parallel_size',
    )
    sizes = {name: getattr(args, name) for name in names}
    # DP is derived by SWIFT, not a new CLI argument. Validate it if already set.
    for name in ('data_parallel_size', 'expert_data_parallel_size', 'virtual_pipeline_model_parallel_size'):
        value = getattr(args, name, None)
        if value is not None:
            sizes[name] = value
    if world_size is not None:
        sizes['world_size'] = world_size
    for name, value in sizes.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{name} must be a positive integer, got {value!r}.')
    if world_size is None:
        return
    # CP belongs to the dense grid; the expert grid has its own TP and DP.
    # VPP schedules virtual stages and does not multiply the physical world size.
    for factors in (
        ('tensor_model_parallel_size', 'pipeline_model_parallel_size', 'context_parallel_size'),
        ('expert_tensor_parallel_size', 'expert_model_parallel_size', 'pipeline_model_parallel_size'),
    ):
        product = 1
        for name in factors:
            product *= sizes[name]
        if world_size % product:
            expression = ' * '.join(f'{name} ({sizes[name]})' for name in factors)
            raise ValueError(f'Invalid Megatron parallel configuration: world_size ({world_size}) must be '
                             f'divisible by {expression} = {product}.')
