# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
from functools import wraps

from transformers import AutoModel, PreTrainedModel

from swift.template import TemplateType
from swift.utils import git_clone_github, safe_snapshot_download
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..patcher import patch_output_clone
from ..register import ModelLoader, register_model


class GotOCR2Loader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        self.auto_model_cls = AutoModel
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.got_ocr2, [
            ModelGroup([
                Model('stepfun-ai/GOT-OCR2_0', 'stepfun-ai/GOT-OCR2_0'),
            ]),
        ],
        GotOCR2Loader,
        template=TemplateType.got_ocr2,
        model_arch=ModelArch.got_ocr2,
        architectures=['GOTQwenForCausalLM'],
        tags=['vision']))


class GotOCR2HfLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers.models.got_ocr2 import GotOcr2ForConditionalGeneration
        GotOcr2ForConditionalGeneration._no_split_modules = ['GotOcr2VisionLayer']
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.got_ocr2_hf, [
            ModelGroup([
                Model('stepfun-ai/GOT-OCR-2.0-hf', 'stepfun-ai/GOT-OCR-2.0-hf'),
            ]),
        ],
        GotOCR2HfLoader,
        template=TemplateType.got_ocr2_hf,
        model_arch=ModelArch.llava_hf,
        architectures=['GotOcr2ForConditionalGeneration'],
        tags=['vision']))


class StepAudioLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        local_repo_path = self.local_repo_path
        if not local_repo_path:
            local_repo_path = git_clone_github('https://github.com/stepfun-ai/Step-Audio.git')
        sys.path.append(local_repo_path)
        from tokenizer import StepAudioTokenizer
        encoder_path = safe_snapshot_download('stepfun-ai/Step-Audio-Tokenizer', check_local=True)
        model = super().get_model(model_dir, *args, **kwargs)
        model.encoder = StepAudioTokenizer(encoder_path)
        # from tts import StepAudioTTS
        # if not os.path.exists('speakers'):
        #     shutil.copytree(os.path.join(local_repo_path, 'speakers'), 'speakers')
        # decoder_path = safe_snapshot_download('stepfun-ai/Step-Audio-TTS-3B', check_local=True)
        # model.decoder = StepAudioTTS(decoder_path, model.encoder)
        return model


register_model(
    ModelMeta(
        MLLMModelType.step_audio, [
            ModelGroup([
                Model('stepfun-ai/Step-Audio-Chat', 'stepfun-ai/Step-Audio-Chat'),
            ]),
        ],
        StepAudioLoader,
        template=TemplateType.step_audio,
        architectures=['Step1ForCausalLM'],
        requires=['funasr', 'sox', 'conformer', 'openai-whisper', 'librosa'],
        tags=['audio']))


def _patch_step_audio2_mini(model):
    if hasattr(model.__class__, 'origin_forward'):
        return

    model.__class__.origin_forward = model.__class__.forward

    @wraps(model.__class__.origin_forward)
    def _forward(self, *args, **kwargs):
        labels = kwargs.get('labels')
        output = self.origin_forward(*args, **kwargs)
        if labels is not None and output.loss is None:
            output['loss'] = self.loss_function(
                logits=output.logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size)
        return output

    model.__class__.forward = _forward


class StepAudio2MiniLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        model = super().get_model(model_dir, *args, **kwargs)
        patch_output_clone(model.model.embed_tokens)
        _patch_step_audio2_mini(model)
        return model


register_model(
    ModelMeta(
        MLLMModelType.step_audio2_mini,
        [ModelGroup([
            Model('stepfun-ai/Step-Audio-2-mini', 'stepfun-ai/Step-Audio-2-mini'),
        ])],
        StepAudio2MiniLoader,
        template=TemplateType.step_audio2_mini,
        model_arch=ModelArch.step_audio2_mini,
        architectures=['StepAudio2ForCausalLM'],
        requires=['transformers==4.53.3', 'torchaudio', 'librosa'],
        tags=['audio'],
    ))
