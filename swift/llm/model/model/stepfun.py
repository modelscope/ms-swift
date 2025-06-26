# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import sys

from transformers import AutoModel

from swift.llm import TemplateType
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal,
                        get_model_tokenizer_with_flash_attn, register_model)
from ..utils import git_clone_github, safe_snapshot_download


def get_model_tokenizer_got_ocr2(*args, **kwargs):
    kwargs['automodel_class'] = AutoModel
    model, tokenizer = get_model_tokenizer_with_flash_attn(*args, **kwargs)
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.got_ocr2, [
            ModelGroup([
                Model('stepfun-ai/GOT-OCR2_0', 'stepfun-ai/GOT-OCR2_0'),
            ]),
        ],
        TemplateType.got_ocr2,
        get_model_tokenizer_got_ocr2,
        model_arch=ModelArch.got_ocr2,
        architectures=['GOTQwenForCausalLM'],
        tags=['vision']))


def get_model_tokenizer_got_ocr2_hf(model_dir, *args, **kwargs):
    from transformers.models.got_ocr2 import GotOcr2ForConditionalGeneration
    GotOcr2ForConditionalGeneration._no_split_modules = ['GotOcr2VisionLayer']
    model, processor = get_model_tokenizer_multimodal(model_dir, *args, **kwargs)
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.got_ocr2_hf, [
            ModelGroup([
                Model('stepfun-ai/GOT-OCR-2.0-hf', 'stepfun-ai/GOT-OCR-2.0-hf'),
            ]),
        ],
        TemplateType.got_ocr2_hf,
        get_model_tokenizer_got_ocr2_hf,
        model_arch=ModelArch.llava_hf,
        architectures=['GOTQwenForCausalLM'],
        tags=['vision']))


def get_model_tokenizer_step_audio(*args, **kwargs):
    local_repo_path = kwargs.get('local_repo_path')
    if not local_repo_path:
        local_repo_path = git_clone_github('https://github.com/stepfun-ai/Step-Audio.git')
    sys.path.append(local_repo_path)
    from tokenizer import StepAudioTokenizer
    encoder_path = safe_snapshot_download('stepfun-ai/Step-Audio-Tokenizer', check_local=True)
    model, tokenizer = get_model_tokenizer_with_flash_attn(*args, **kwargs)
    if model is not None:
        model.encoder = StepAudioTokenizer(encoder_path)
        # from tts import StepAudioTTS
        # if not os.path.exists('speakers'):
        #     shutil.copytree(os.path.join(local_repo_path, 'speakers'), 'speakers')
        # decoder_path = safe_snapshot_download('stepfun-ai/Step-Audio-TTS-3B', check_local=True)
        # model.decoder = StepAudioTTS(decoder_path, model.encoder)
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.step_audio, [
            ModelGroup([
                Model('stepfun-ai/Step-Audio-Chat', 'stepfun-ai/Step-Audio-Chat'),
            ]),
        ],
        TemplateType.step_audio,
        get_model_tokenizer_step_audio,
        architectures=['Step1ForCausalLM'],
        requires=['funasr', 'sox', 'conformer', 'openai-whisper', 'librosa'],
        tags=['audio']))
