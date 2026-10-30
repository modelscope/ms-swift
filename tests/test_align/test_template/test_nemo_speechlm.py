# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import numpy as np
import os
import pytest
import torch

from swift.template.templates.nvidia import NemoSpeechLMTemplate, _nemotron_transcribe_audio_tokens


def test_transformers_registration():
    from swift.model.register import MODEL_MAPPING

    meta = MODEL_MAPPING['nemo_speechlm']
    assert meta.architectures == ['Nemotron3_5TranscribeForConditionalGeneration']
    assert not any(requirement.startswith('nemo') for requirement in meta.requires)


def test_audio_token_count_from_mask():
    mask = torch.zeros(4, 105, dtype=torch.bool)
    mask[0, :100] = True
    mask[1, :101] = True
    mask[2, :104] = True
    mask[3, :105] = True
    counts = _nemotron_transcribe_audio_tokens(mask, 8)
    assert counts.tolist() == [13, 13, 13, 14]


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


def test_packing_is_disabled():
    assert NemoSpeechLMTemplate.support_padding_free is False


def test_audio_collator_flattens_and_pads():
    template = NemoSpeechLMTemplate.__new__(NemoSpeechLMTemplate)
    mask = torch.ones(2, 104, dtype=torch.bool)
    batch = [{
        'input_features': torch.zeros(2, 104, 128),
        'input_features_mask': mask,
    }, {
        'input_features': torch.full((1, 96, 128), 2.),
        'input_features_mask': torch.ones(1, 96, dtype=torch.bool),
    }]
    result = template._data_collator_mm_data(batch)
    assert result['input_features'].shape == (3, 104, 128)
    assert result['input_features_mask'].shape == (3, 104)
    assert result['input_features_mask'][2, 96:].count_nonzero() == 0


@pytest.mark.skipif(
    not os.environ.get('NEMO_SPEECHLM_MODEL_DIR'), reason='Set NEMO_SPEECHLM_MODEL_DIR for tokenizer integration.')
def test_real_tokenizer_audio_expansion():
    from swift.model import get_model_processor
    from swift.template import get_template

    model_dir = os.environ['NEMO_SPEECHLM_MODEL_DIR']
    with open(os.path.join(model_dir, 'config.json'), encoding='utf-8') as f:
        if 'audio_config' not in json.load(f):
            pytest.skip('requires transformed checkpoint metadata')
    _, processor = get_model_processor(model_dir, load_model=False)
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
    assert encoded['input_ids'].count(audio_token_id) == 13
    assert len(encoded['labels']) == len(encoded['input_ids'])
    batch = template.data_collator([encoded])
    assert processor.model_info.config.text_config.vocab_size == 131072
    assert batch['input_features'].shape == (1, 104, 128)
    assert batch['input_features_mask'].shape == (1, 104)
