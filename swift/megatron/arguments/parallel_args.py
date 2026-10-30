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
    """Diagnose Megatron's existing dense/expert divisibility checks earlier."""
    if world_size is None:
        return
    sizes = {
        name: getattr(args, name)
        for name in ('tensor_model_parallel_size', 'pipeline_model_parallel_size', 'context_parallel_size',
                     'expert_model_parallel_size', 'expert_tensor_parallel_size')
    }
    # Match Megatron Core without changing the caller's argument object.
    if sizes['expert_tensor_parallel_size'] is None:
        sizes['expert_tensor_parallel_size'] = sizes['tensor_model_parallel_size']
    # DP is derived and VPP may be normalized from the pipeline layout. Neither
    # receives a new validation policy here; existing backend checks still run.
    for grid, factors in (
        ('dense', ('tensor_model_parallel_size', 'pipeline_model_parallel_size', 'context_parallel_size')),
        ('expert', ('expert_tensor_parallel_size', 'expert_model_parallel_size', 'pipeline_model_parallel_size')),
    ):
        expression = ' * '.join(f'{name} ({sizes[name]!r})' for name in factors)
        try:
            product = sizes[factors[0]] * sizes[factors[1]] * sizes[factors[2]]
            remainder = world_size % product
        except (TypeError, ZeroDivisionError) as exc:
            reason = ('the parallel-size product is zero, so the world size cannot be divided by it' if isinstance(
                exc, ZeroDivisionError) else 'the supplied values cannot be used in Megatron parallel-size arithmetic')
            raise ValueError(f'Invalid Megatron parallel configuration: world_size ({world_size!r}), '
                             f'{expression}; {reason}. Original error: {exc}') from exc
        if remainder != 0:
            fallback = (' expert_tensor_parallel_size=None resolves to tensor_model_parallel_size.'
                        if grid == 'expert' and args.expert_tensor_parallel_size is None else '')
            raise ValueError(f'Invalid Megatron parallel configuration: world_size ({world_size}) must be '
                             f'divisible by {expression} = {product}. '
                             f'Remainder: {world_size} % {product} = {remainder}; ranks cannot be '
                             f'partitioned into equal-sized {grid} parallel groups.{fallback}')
