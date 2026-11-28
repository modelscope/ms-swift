# Copyright (c) ModelScope Contributors. All rights reserved.
"""GAE followed by optional policy signals on the same response-token frame."""

from __future__ import annotations
from typing import TYPE_CHECKING

from twinkle import remote_class
from twinkle.advantage import GAEAdvantage
from twinkle.advantage.base import _absmean, _mean, _zero_row_fraction

if TYPE_CHECKING:
    from swift.dev.config import RLConfig


@remote_class(execute="first")
class PolicyGAE(GAEAdvantage):
    """Keep value targets from GAE; teacher regularization changes only actor advantages."""

    def __init__(self, rl_config: RLConfig, **kwargs):
        super().__init__(**kwargs)
        self.rl_config = rl_config

    def _estimate(self, trajs, rollout_logps, scores, *, ref_out, teacher_out, old_out, value_out, global_step=0):
        cfg = self.rl_config
        old_logps = self._lane("old_out", old_out, "logps", trajs) if old_out is not None else rollout_logps
        rows, summary = super()._estimate(
            trajs,
            old_logps,
            scores,
            ref_out=ref_out,
            value_out=value_out,
            teacher_out=None,
            old_out=None,
            global_step=global_step,
        )
        if teacher_out is not None:
            teacher_logps = self._lane("teacher_out", teacher_out, "logps", trajs)
            updates = []
            for index, (row, teacher, old) in enumerate(zip(rows, teacher_logps, old_logps)):
                if len(teacher) != len(row["advantages"]) or len(old) != len(teacher):
                    raise ValueError("Teacher, policy and GAE advantages must have identical response-token frames.")
                advantage = row["advantages"]
                if cfg.teacher_kl_coef and cfg.sdar_loss_coef <= 0 and not summary["unscored_flags"][index]:
                    advantage = [a + cfg.teacher_kl_coef * (t - p) for a, t, p in zip(advantage, teacher, old)]
                updates.append(dict(row, advantages=advantage, teacher_logps=teacher))
            rows = updates
        elif cfg.teacher_kl_coef or cfg.sdar_loss_coef > 0:
            raise ValueError("Configured teacher signal requires a teacher lane.")
        if cfg.log_rollout_offpolicy_metrics:
            summary["rollout_correction"] = self._correction(old_logps, rollout_logps)
        advantages = [row["advantages"] for row in rows]
        summary.update(
            mean_advantage=_mean(advantages),
            absmean_advantage=_absmean(advantages),
            zero_advantage_rows=_zero_row_fraction(advantages),
        )
        return rows, summary
