# Copyright (c) ModelScope Contributors. All rights reserved.
"""Owned resource declarations; algorithms do not define separate policy lifecycles."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from twinkle.advantage.base import Advantage
    from twinkle.dataloader import DataLoader
    from twinkle.processor import RoutingReplay, TrajectoryFilter, TrajectoryUpdate
    from twinkle.reward import FunctionWorker, GenerativeWorker
    from twinkle_agentic.async_rl import ClusterAdapter
    from twinkle_agentic.rollout import MultiTurnRollout

    from swift.dev.recipe.distill.worker import DistillWorker
    from swift.dev.recipe.online import RolloutSampler, WeightSync
    from swift.dev.recipe.preparation import ModelProcessor, TrainingModel
    from swift.dev.reward.prm import PRMSegmentWorker
    from swift.dev.template import Template
    from swift.dev.utils.scoring import PrivilegedView


@dataclass(kw_only=True)
class OfflineResources:
    model: TrainingModel
    processor: ModelProcessor
    template: Template
    dataloader: DataLoader
    eval_dataloader: Optional[DataLoader]
    total_opt_steps: int
    task_type: Optional[str]
    forward_task: str


@dataclass(kw_only=True)
class OnlineResources:
    model: TrainingModel
    sampler: RolloutSampler
    rollout: MultiTurnRollout
    engine: WeightSync
    sampling_params: dict
    colocate: bool


@dataclass(kw_only=True)
class PolicyResources(OnlineResources):
    advantage: Advantage
    critic: Optional[TrainingModel] = None
    adapter: Optional[ClusterAdapter] = None
    reference: Optional[TrainingModel] = None
    teacher: Optional[TrainingModel] = None
    orm_scorers: list[FunctionWorker | GenerativeWorker] = field(default_factory=list)
    orm_names: list[str] = field(default_factory=list)
    prm_scorers: list[FunctionWorker] = field(default_factory=list)
    prm_names: list[str] = field(default_factory=list)
    privileged: Optional[PrivilegedView] = None
    routing: Optional[RoutingReplay] = None
    traj_filter: Optional[TrajectoryFilter] = None
    prm_segment: Optional[PRMSegmentWorker] = None
    traj_update: Optional[TrajectoryUpdate] = None


@dataclass(kw_only=True)
class DistillResources(OnlineResources):
    teachers: list[TrainingModel]
    worker: DistillWorker
    adapter: Optional[ClusterAdapter] = None


@dataclass(kw_only=True)
class RFTResources(OnlineResources):
    orm_scorers: list[FunctionWorker | GenerativeWorker]
    orm_scorer_names: list[str]
