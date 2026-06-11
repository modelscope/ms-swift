# Copyright (c) ModelScope Contributors. All rights reserved.
"""GRPO batch data structures.

Defines the post-collation data structure that separates RL signals from
model forward inputs. Before collation, data remains as ``List[Dict]``
(determined by template/model/dataset). After collation, this module
provides ``GRPOBatch`` — the typed contract for what the loss function needs.
"""
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

import torch


@dataclass
class GRPOBatch:
    """Batch data for GRPO loss computation (post-collation, batch-level).

    1. ``completion_mask``, ``truncated_mask``, ``seq_lengths`` — derived from
       collated ``labels`` right after ``data_collator``.
    2. ``old_per_token_logps``, ``ref_per_token_logps`` — computed via
       model/ref forward on the collated batch.
    3. ``advantages`` — computed from gathered rewards.
    4. ``rollout_per_token_logps``, ``num_items_in_batch`` — optional,
       filled when rollout IS / DAPO is enabled.
    """
    completion_mask: torch.Tensor                           # [B, T]
    truncated_mask: torch.Tensor                            # [B]
    seq_lengths: torch.Tensor                               # [B] or [B+n] for padding_free

    old_per_token_logps: Optional[torch.Tensor] = None      # [B, T]
    ref_per_token_logps: Optional[torch.Tensor] = None      # [B, T]
    rollout_per_token_logps: Optional[torch.Tensor] = None  # [B, T]
    advantages: Optional[torch.Tensor] = None               # [B]
    num_items_in_batch: Optional[torch.Tensor] = None       # scalar
    logits_to_keep: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict (for backward compat with code that expects dict)."""
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'GRPOBatch':
        """Extract RL fields from a mixed dict, leaving the rest untouched.

        Returns ``(rl_batch, model_inputs)`` where ``model_inputs`` is the
        dict with RL keys removed.
        """
        rl_field_names = {f.name for f in fields(GRPOBatch)}
        rl_kwargs = {}
        model_inputs = {}
        for k, v in d.items():
            if k in rl_field_names:
                rl_kwargs[k] = v
            else:
                model_inputs[k] = v
        return GRPOBatch(**rl_kwargs), model_inputs


# Keys that belong to GRPOBatch, not to model forward inputs.
GRPO_RL_KEYS = frozenset(f.name for f in fields(GRPOBatch))
