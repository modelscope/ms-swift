# Copyright (c) ModelScope Contributors. All rights reserved.
"""Built-in component bindings: thin adapters over real twinkle / swift.dev objects.

Importing this package registers every binding into
``swift.dev.webui.component.COMPONENT_CLASSES``. Each class carries ``type``
matching a registry descriptor and binds ONE native surface -- construction
here, behaviour stays native. Heavy payloads move as references: bindings pass
them through, never materialising token/logprob batches in this process.
"""
from . import data, resources, signals, training  # noqa: F401 -- registration side effect


def _register_all() -> None:
    from ..component import register_component
    from .data import (ChallengerComponent, DataloaderComponent, GenerateStage,
                       ScoreStage, SelectStage, StreamingDriverComponent)
    from .resources import (ApiEndpointComponent, CodeComponent, DatasetComponent,
                            GroupRuleComponent, ModelComponent, RolloutComponent,
                            SamplerComponent, SandboxEnvComponent, PolicyEndpointComponent,
                            ToolsetComponent)
    from .signals import (AdvantageStage, ModelForwardStage, ReduceStage,
                          RewardShapeStage, SignalCombineStage)
    from .tasks import (DeployTask, EvalTask, HubPushTask, InferTask, MergeTask,
                        QuantizeTask)
    from .training import (CheckpointStage, TrainStepStage, WeightAverageStage,
                           WeightSyncStage)
    for cls in (
            ModelComponent, DatasetComponent, SamplerComponent,
            RolloutComponent, GroupRuleComponent, ToolsetComponent, SandboxEnvComponent,
            PolicyEndpointComponent, ApiEndpointComponent, CodeComponent,
            DataloaderComponent, StreamingDriverComponent, ChallengerComponent,
            GenerateStage, ScoreStage, SelectStage,
            ModelForwardStage, ReduceStage, RewardShapeStage, AdvantageStage,
            SignalCombineStage, TrainStepStage, WeightSyncStage, WeightAverageStage,
            CheckpointStage, MergeTask, QuantizeTask, EvalTask, InferTask,
            DeployTask, HubPushTask):
        register_component(cls)


_register_all()
