# Copyright (c) ModelScope Contributors. All rights reserved.
"""PRM step segmentation: the plugins that decide when a rollout step closes a segment.

The driver's PRM channel scores the trajectory-so-far at every closed segment and stamps the
per-segment scores into the score rows; WHERE those boundaries fall is this module's concern. A
plugin resolves to a ``prm_segment(sample, traj) -> bool`` predicate -- the one the StreamingDriver
consults after every step, where ``True`` closes the running segment. ``traj`` arrives in whatever
form the driver holds it. PRMSegmentWorker evaluates content-based predicates at the compute
boundary; the driver receives only their boolean verdicts.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

from swift.dev.utils.plugin import PluginRegistry, SwiftPlugin

from twinkle import remote_class, remote_function
from twinkle.data_format import last_assistant_text

__all__ = ['PRMSegmentScorer', 'TurnPRMScorer', 'DelimiterPRMScorer', 'prm_scorers',
           'PRM_STEP_SCORER', 'build_prm_segment']


@remote_class(execute='first')
class PRMSegmentWorker:
    """Evaluate a segmentation predicate on trajectory data and publish only its boolean verdict."""

    def __init__(self, predicate: Callable[[Any, Any], bool]):
        self.predicate = predicate

    @remote_function(dispatch='all', execute='first', collect='first', lazy_collect=False)
    def __call__(self, sample: Any, traj: Any) -> bool:
        return bool(self.predicate(sample, traj))


class PRMSegmentScorer(SwiftPlugin):
    """Base of the PRM segmentation plugins: build the driver's ``prm_segment`` predicate.

    ``build`` returns ``None`` for "the driver's own default" (every step closes a segment), or a
    ``prm_segment(sample, traj) -> bool`` callable. ``args`` is the run's RewardConfig, so a
    plugin reads its own knobs (``prm_step_delimiter`` and friends) off it.
    """

    def build(self) -> Optional[Callable[[Any, Any], bool]]:
        raise NotImplementedError


class TurnPRMScorer(PRMSegmentScorer):
    """The multi-turn default: one assistant turn (thought + action) is one PRM segment.

    Agentic-RL PRM research scores a tool-using trajectory one assistant turn at a time -- a
    turn's thought+action is the unit a PRM judges, and the tool observation it triggers is
    context for the NEXT turn. That is exactly the driver's default (every step closes a
    segment), so the predicate is ``None`` -- no extra mechanism.
    """

    name = 'turn'

    def build(self) -> Optional[Callable[[Any, Any], bool]]:
        return None


class DelimiterPRMScorer(PRMSegmentScorer):
    """A segment closes at the turn whose assistant text contains the delimiter.

    ``prm_step_delimiter`` ('\n' default): the step just finished closes the running segment when
    its newest assistant text contains the delimiter; turns without one accumulate into the next
    closing turn. Known limit versus the legacy token-space splitter: the streaming driver cuts
    segments at STEP boundaries only, so several delimited steps inside ONE response are not
    expressible -- a single-turn rollout degenerates to one segment (closed at done), and
    multi-turn delimited reasoning (the delimiter marking turn ends) is the supported shape.
    """

    name = 'delimiter'

    def __init__(self, args: Optional[Any] = None, **kwargs):
        super().__init__(args, **kwargs)
        #: Step boundary in the decoded assistant text, read off the run's RewardConfig.
        self.delimiter = getattr(args, 'prm_step_delimiter', None) or '\n'

    def build(self) -> Optional[Callable[[Any, Any], bool]]:
        delimiter = self.delimiter

        def prm_segment(sample: Any, traj: Any) -> bool:
            return delimiter in last_assistant_text(traj)

        return prm_segment


prm_scorers: Dict[str, Type[PRMSegmentScorer]] = {
    'turn': TurnPRMScorer,
    'delimiter': DelimiterPRMScorer,
}

#: The 'prm_scorer' extension point: a run picks a segmentation with ``RewardConfig.prm_scorer``
#: (a registered name, a class, or a callable), resolved through :meth:`PluginRegistry.resolve`.
PRM_STEP_SCORER = PluginRegistry.register_kind(
    'prm_scorer', PRMSegmentScorer, config_field='prm_scorer', entries=prm_scorers)


def build_prm_segment(reward_config: Optional[Any], rollout_config: Optional[Any], *,
                      remote_group: Optional[str] = None) -> Optional[Callable[[Any, Any], bool]]:
    """The driver's ``prm_segment`` predicate, or ``None`` (the driver default) without --prm.

    ``prm_scorer`` unset picks by rollout shape, mirroring the legacy default: 'turn' for a
    multi-turn rollout (``max_turns`` set), 'delimiter' otherwise. A registered name / class
    resolves to its plugin; a plain callable is used as the predicate itself (the
    ``prm_segment(sample, traj) -> bool`` contract).
    """
    if reward_config is None or not reward_config.prm:
        return None
    spec = reward_config.prm_scorer
    if spec is None:
        spec = 'turn' if getattr(rollout_config, 'max_turns', None) is not None else 'delimiter'
    plugin = PluginRegistry.resolve(PRM_STEP_SCORER, spec, config=reward_config)
    if isinstance(plugin, PRMSegmentScorer):
        if reward_config.prm_step_delimiter is not None and isinstance(plugin, TurnPRMScorer):
            raise ValueError(
                f'prm_step_delimiter={reward_config.prm_step_delimiter!r} only shapes the delimiter '
                "segmentation, but the resolved prm_scorer is 'turn': pass --prm_scorer delimiter "
                'or drop the delimiter.')
        predicate = plugin.build()
    elif callable(plugin):
        predicate = plugin
    else:
        raise ValueError(f'prm_scorer {spec!r} resolved to {plugin!r}: expected a PRMSegmentScorer '
                         'plugin or a (sample, traj) -> bool callable.')
    if predicate is None:
        return None
    import uuid

    return PRMSegmentWorker(predicate, remote_group=remote_group, instance_id=uuid.uuid4().hex)
