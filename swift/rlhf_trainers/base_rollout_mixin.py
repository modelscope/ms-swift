# Copyright (c) ModelScope Contributors. All rights reserved.
"""Backend-agnostic rollout sample helpers shared by HF and Megatron trainers.

This mixin contains only stateless / backend-independent sample handling
methods.  Backend-specific rollout infrastructure (vLLM engine setup, weight
sync, distributed groups) stays in the respective mixins.
"""
import copy
from typing import Any, Dict, List

from swift.rl_core.data import OnPolicySample


class BaseRolloutTrainerMixin:
    """Base mixin with backend-agnostic per-sample rollout utilities."""

    sample_cls = OnPolicySample

    def to_samples(self, rows: List[Any]) -> List[OnPolicySample]:
        """Convert dataloader/rollout dict rows into per-sample objects.

        Rows that are already samples pass through unchanged.
        """
        return [r if isinstance(r, OnPolicySample) else self.sample_cls.from_row(r) for r in rows]

    @staticmethod
    def _split_data_by_steps(samples: List[OnPolicySample], steps: int) -> List[List[OnPolicySample]]:
        """Split a list of samples into ``steps`` chunks with balanced sizes."""
        if steps <= 1:
            return [samples]

        chunk_size = len(samples) // steps
        remainder = len(samples) % steps
        chunks: List[List[OnPolicySample]] = []
        start_idx = 0
        for i in range(steps):
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size
            chunks.append(samples[start_idx:end_idx])
            start_idx = end_idx
        return chunks

    def _set_inputs_system(self, samples: List[OnPolicySample]) -> List[OnPolicySample]:
        """Insert default system message if not present."""
        if not self.template.template_meta.default_system:
            return samples
        if all(s.messages[0]['role'] == 'system' for s in samples):
            return samples
        for s in samples:
            if s.messages[0]['role'] != 'system':
                s.messages.insert(0, {'role': 'system', 'content': self.template.template_meta.default_system})
        return samples

    def _postprocess_rollout_outputs(self, samples: List[OnPolicySample], outputs: List[Any]) -> List[OnPolicySample]:
        """Merge rollout outputs back onto samples (deepcopy to match HF path)."""
        assert len(samples) == len(outputs), (f'samples ({len(samples)}) and outputs ({len(outputs)}) mismatch')
        results = []
        for s, out in zip(samples, outputs):
            s = copy.deepcopy(s)
            s.apply_rollout_output(rollout_output=out)
            results.append(s)
        return results
