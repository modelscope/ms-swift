# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from .utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .version import __version__, __release_datetime__
    from .tuners import (Adapter, AdapterConfig, AdapterModule, SwiftModel, LoRA, LoRAConfig, SWIFT_MAPPING,
                         AdaLoraConfig, IA3Config, LoftQConfig, LoHaConfig, LoKrConfig, LoraConfig, OFTConfig,
                         PeftConfig, PeftModel, PeftModelForCausalLM, ResTuningConfig, SideConfig,
                         PeftModelForSeq2SeqLM, PeftModelForSequenceClassification, PeftModelForTokenClassification,
                         PrefixTuningConfig, PromptEncoderConfig, PromptLearningConfig, PromptTuningConfig,
                         get_peft_config, get_peft_model, get_peft_model_state_dict, Prompt, PromptConfig, PromptModule,
                         SwiftConfig, SwiftOutput, Swift, SwiftTuners, LongLoRAConfig, LongLoRA, LongLoRAModelType,
                         SCETuning, SCETuningConfig)
    from .hub import snapshot_download, push_to_hub, push_to_hub_async, push_to_hub_in_queue
    from .trainers import (EvaluationStrategy, FSDPOption, HPSearchBackend, HubStrategy, IntervalStrategy,
                           SchedulerType, ShardedDDPOption, TrainingArguments, Seq2SeqTrainingArguments, Trainer,
                           Seq2SeqTrainer)
    from .utils import get_logger
else:
    _import_structure = {
        'version': ['__release_datetime__', '__version__'],
        'hub': ['snapshot_download', 'push_to_hub', 'push_to_hub_async', 'push_to_hub_in_queue'],
        'tuners': [
            'Adapter', 'AdapterConfig', 'AdapterModule', 'SwiftModel', 'LoRA', 'LoRAConfig', 'SWIFT_MAPPING',
            'LoraConfig', 'AdaLoraConfig', 'IA3Config', 'LoftQConfig', 'LoHaConfig', 'LoKrConfig', 'OFTConfig',
            'PeftConfig', 'ResTuningConfig', 'SideConfig', 'PeftModel', 'PeftModelForCausalLM', 'PeftModelForSeq2SeqLM',
            'PeftModelForSequenceClassification', 'PeftModelForTokenClassification', 'PrefixTuningConfig',
            'PromptEncoderConfig', 'PromptLearningConfig', 'PromptTuningConfig', 'get_peft_config', 'get_peft_model',
            'get_peft_model_state_dict', 'Prompt', 'PromptConfig', 'PromptModule', 'SwiftConfig', 'SwiftOutput',
            'Swift', 'SwiftTuners', 'LongLoRAConfig', 'LongLoRA', 'LongLoRAModelType', 'SCETuning', 'SCETuningConfig'
        ],
        'trainers': [
            'EvaluationStrategy', 'FSDPOption', 'HPSearchBackend', 'HubStrategy', 'IntervalStrategy', 'SchedulerType',
            'ShardedDDPOption', 'TrainingArguments', 'Seq2SeqTrainingArguments', 'Trainer', 'Seq2SeqTrainer'
        ],
        'utils': ['get_logger']
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
