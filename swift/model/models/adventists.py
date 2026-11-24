# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import sys
from transformers import PretrainedConfig, PreTrainedModel

from swift.template import TemplateType
from swift.utils import Processor, git_clone_github
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model


class SophonRealtimeLoader(ModelLoader):
    """Thinker of Sophon-Realtime v1: Qwen3-ASR encoder -> projector -> Qwen3-32B.

    The modeling code lives in https://github.com/adventists-ai/sophon-realtime (finetune/sophon_swift);
    the weights are the `base/` folder of the ModelScope repo.
    """

    def _import_sophon_swift(self):
        local_repo_path = self.local_repo_path
        if not local_repo_path:
            local_repo_path = git_clone_github('https://github.com/adventists-ai/sophon-realtime')
        path = os.path.join(local_repo_path, 'finetune')
        if path not in sys.path:
            sys.path.append(path)
        import sophon_swift
        sophon_swift.register_auto()
        return sophon_swift

    def get_config(self, model_dir: str) -> PretrainedConfig:
        sophon_swift = self._import_sophon_swift()
        return sophon_swift.DuplexJevConfig.from_pretrained(model_dir)

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        sophon_swift = self._import_sophon_swift()
        return sophon_swift.SophonRealtimeProcessor.from_pretrained(model_dir, config)

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        sophon_swift = self._import_sophon_swift()
        self.auto_model_cls = self.auto_model_cls or sophon_swift.DuplexJevForConditionalGeneration
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.sophon_realtime,
        [ModelGroup([
            Model('AdventistsAI/Sophon-Realtime-v1:base'),
        ])],
        SophonRealtimeLoader,
        template=TemplateType.sophon_realtime,
        model_arch=ModelArch.sophon_realtime,
        architectures=['DuplexJevForConditionalGeneration'],
        requires=['transformers>=4.51', 'librosa'],
        tags=['audio'],
    ))
