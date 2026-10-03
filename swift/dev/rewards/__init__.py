# Copyright (c) ModelScope Contributors. All rights reserved.
"""Reward plugins for swift.dev: outcome rewards (ORM) and process-reward step scorers (PRM).

Internalized from ``swift.rewards`` so swift.dev carries no runtime dependency on legacy swift packages.

``ORM`` is the historical name of swift's reward plugin base (see :mod:`swift.dev.plugin`); ``orms`` is the
registry the ``reward`` extension point adopts, and :data:`REWARD` is that point. Consumed by
:mod:`swift.dev.reward`.

:mod:`swift.dev.rewards.prm` declares the sibling ``prm_scorer`` extension point: :class:`PRMScorer` is the
general base that segments a response into reasoning steps and broadcasts each step's PRM score onto its
tokens, :class:`DelimiterPRMScorer` is the default (one step per delimiter-separated line), ``prm_scorers``
is the registry it adopts, and :data:`PRM_STEP_SCORER` is that point. Consumed by the GRPO loop to build a
per-token process reward.
"""
from .orm import ORM, REWARD, orms
from .prm import PRM_STEP_SCORER, DelimiterPRMScorer, PRMScorer, prm_scorers

__all__ = [
    'ORM', 'REWARD', 'orms', 'PRMScorer', 'DelimiterPRMScorer', 'prm_scorers', 'PRM_STEP_SCORER'
]
