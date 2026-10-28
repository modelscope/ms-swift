# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import numpy as np
import os
import pytest
import torch

from swift.template.templates.nvidia import NemoSpeechLMTemplate, _nemo_speechlm_audio_tokens

ESTIMATOR = {
    'preprocessor': {
        'n_fft': 512,
        'hop_length': 160,
        'stft_pad_amount': 256,
    },
    'subsampling': {
        'type': 'feature_stacking',
        'factor': 8,
    },
}


def test_audio_token_estimator():
    assert _nemo_speechlm_audio_tokens(16000, ESTIMATOR) == 13
    assert _nemo_speechlm_audio_tokens(32000, ESTIMATOR) == 25
    assert _nemo_speechlm_audio_tokens(1, ESTIMATOR) == 1


def test_fleurs_cantonese_preprocessor():
    from swift.dataset.dataset.mllm import FleursCantonesePreprocessor

    row = {
        'raw_transcription': '自卸卡車被用來堵住隧道的各個入口處。',
        'audios': {
            'bytes': b'wav',
            'path': 'audio.wav',
        },
    }
    result = FleursCantonesePreprocessor().preprocess(row)
    assert result['messages'][0] == {
        'role': 'user',
        'content': FleursCantonesePreprocessor.prompt,
    }
    assert result['messages'][1] == {
        'role': 'assistant',
        'content': row['raw_transcription'],
    }
    assert result['audios'] == [b'wav']


def test_packing_is_rejected():
    template = NemoSpeechLMTemplate.__new__(NemoSpeechLMTemplate)
    with pytest.raises(NotImplementedError, match='packing false'):
        template.packing_row([])


def test_audio_collator_flattens_and_pads():
    template = NemoSpeechLMTemplate.__new__(NemoSpeechLMTemplate)
    batch = [{'audio_signal': [torch.zeros(160), torch.ones(80)]}, {'audio_signal': [torch.full((40, ), 2.)]}]
    result = template._data_collator_mm_data(batch)
    assert result['audio_signal'].shape == (3, 160)
    assert result['audio_signal_length'].tolist() == [160, 80, 40]
    assert result['audio_signal'][1, 80:].count_nonzero() == 0
    assert result['audio_signal'][2, 40:].count_nonzero() == 0


@pytest.mark.skipif(
    not os.environ.get('NEMO_SPEECHLM_MODEL_DIR'), reason='Set NEMO_SPEECHLM_MODEL_DIR for tokenizer integration.')
def test_config_runtime_vocab_is_idempotent(tmp_path):
    from swift.model.models.nvidia import NemoSpeechLMLoader

    loader = NemoSpeechLMLoader.__new__(NemoSpeechLMLoader)
    config = loader.get_config(os.environ['NEMO_SPEECHLM_MODEL_DIR'])
    assert config.speechlm_output_vocab_size == 131072
    assert config.llm_config.vocab_size == 131072
    config.llm_config.vocab_size = 131073
    config.save_pretrained(tmp_path)
    with open(tmp_path / 'config.json', encoding='utf-8') as f:
        assert json.load(f)['model_type'] == 'nemo_speechlm'
    reloaded = loader.get_config(tmp_path)
    assert reloaded.speechlm_output_vocab_size == 131072
    assert reloaded.llm_config.vocab_size == 131073


@pytest.mark.skipif(
    not os.environ.get('NEMO_SPEECHLM_MODEL_DIR'), reason='Set NEMO_SPEECHLM_MODEL_DIR for tokenizer integration.')
def test_real_tokenizer_audio_expansion():
    from swift.model import get_model_processor
    from swift.template import get_template

    _, processor = get_model_processor(os.environ['NEMO_SPEECHLM_MODEL_DIR'], load_model=False)
    template = get_template(processor, template_type='nemo_speechlm')
    template.set_mode('train')
    encoded = template.encode({
        'messages': [{
            'role': 'user',
            'content': 'Transcribe <audio>'
        }, {
            'role': 'assistant',
            'content': 'hello'
        }],
        'audios': [np.zeros(16000, dtype=np.float32)],
    })
    audio_token_id = processor.model_info.config.audio_token_id
    assert processor.model_info.config.llm_config.vocab_size == 131073
    assert encoded['input_ids'].count(audio_token_id) == 13
    assert len(encoded['labels']) == len(encoded['input_ids'])
    batch = template.data_collator([encoded])
    assert batch['audio_signal'].shape == (1, 16000)
    assert batch['audio_signal_length'].tolist() == [16000]
