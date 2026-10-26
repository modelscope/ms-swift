# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pass colocate trainer-offload switches through to the twinkle model."""
from typing import Any


def offload_for_rollout(model: Any, rollout_config: Any) -> None:
    """Move the trainer pieces ``rollout_config`` asked to move. Both false leaves it resident.

    vLLM sleep/wake is separate and stays at the call site. ``reload_to_gpu`` later restores
    only what this call actually moved.
    """
    model.offload_to_cpu(
        offload_model=bool(rollout_config.offload_model),
        offload_optimizer=bool(rollout_config.offload_optimizer),
    )
