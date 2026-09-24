# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
import pytest
import torch
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from swift.template import TEMPLATE_MAPPING
from swift.template.base import Template


@pytest.mark.parametrize('strategy', ['left', 'right'])
@pytest.mark.parametrize('padding_side', ['left', 'right'])
@pytest.mark.parametrize('training', [False, True])
def test_paligemma_truncated_batch_forward_and_backward(strategy, padding_side, training):
    from transformers import PaliGemmaConfig, PaliGemmaForConditionalGeneration

    # A small random model exercises the real attention mask, vision projection
    # and loss without downloading pretrained weights. Tokenization is stubbed.
    config = PaliGemmaConfig(
        vision_config=dict(
            model_type='siglip_vision_model',
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=2,
            patch_size=1),
        text_config=dict(
            model_type='gemma',
            vocab_size=100,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=64,
            pad_token_id=0),
        image_token_index=90,
        vocab_size=100,
        projection_dim=16,
        hidden_size=16,
    )
    model = PaliGemmaForConditionalGeneration(config)
    model.train(training)
    meta = TEMPLATE_MAPPING['paligemma']
    template = meta.template_cls(None, meta, max_length=8, truncation_strategy=strategy, padding_side=padding_side)
    template.processor = _Processor('paligemma')
    template.model_info = SimpleNamespace(torch_dtype=torch.float32)
    template.mode = 'train' if training else 'transformers'
    template.placeholder_tokens = [90]
    batch = []
    for has_image in [True, False]:
        ids = [10, 11, 90, 90, 90, 90, 12, 13, 14, 15] if has_image else [10, 12, 13]
        labels = [-100] * 6 + [12, 13, 14, 15] if has_image else [-100, 12, 13]
        base = {'input_ids': ids, 'labels': labels if training else None, 'loss_scale': None}
        with patch.object(template, '_preprocess_inputs'), patch.object(Template, '_encode', return_value=base):
            batch.append(template._encode_truncated(_inputs('image' if has_image else '')))
    collated = template._data_collator(batch)
    output = model(**collated, use_cache=False)
    assert output.logits.shape == (2, 8, 100)
    assert torch.isfinite(output.logits).all()
    if training:
        assert torch.isfinite(output.loss)
        output.loss.backward()
        projector_parameters = [
            parameter for name, parameter in model.named_parameters() if 'multi_modal_projector' in name
        ]
        assert projector_parameters
        for parameter in projector_parameters:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


class _Tokenizer:
    pad_token_id = 0
    tokens = {
        '<image>': 90,
        '<start_of_image>': 99,
        '<start_of_audio>': 98,
        '<|image|>': 99,
        '<|video|>': 97,
        '<im_patch>': 90,
        '<frame_start>': 95,
        '<frame_end>': 96,
        '<|reserved_special_token_0|>': 0,
    }
    sequences = {
        'image': [99, 90, 90, 89],
        '\n\nimage': [99, 90, 90, 89],
        'audio': [98, 91, 91, 88],
        'video': [95, 90, 90, 96],
    }

    def convert_tokens_to_ids(self, token):
        return self.tokens[token]

    def encode(self, text, **kwargs):
        return self.sequences[text].copy()

    def __call__(self, text, **kwargs):
        return {'input_ids': self.encode(text)}


class _Processor:
    pad_token_id = 0
    image_token_id = 90
    audio_token_id = 91
    image_token_ids = [90]
    full_image_sequence = '\n\nimage'
    full_audio_sequence = 'audio'

    def __init__(self, template_type):
        self.tokenizer = _Tokenizer()
        self.template_type = template_type

    def __call__(self, **kwargs):
        return {'pixel_values': torch.ones(1, 3, 2, 2)}

    def image_processor(self, *args, **kwargs):
        result = {'pixel_values': np.ones((1, 3, 2, 2), dtype=np.float32)}
        if self.template_type == 'molmo2':
            result.update(
                pixel_values=torch.ones(1, 3, 2, 2),
                image_grids=torch.tensor([[1, 2, 2]]),
                image_token_pooling=torch.tensor([[0, 1]]),
                image_num_crops=torch.tensor([1]),
            )
        else:
            result['num_crops'] = [1]
        return result

    def feature_extractor(self, *args, **kwargs):
        return {
            'input_features': np.ones((1, 3, 4), dtype=np.float32),
            'input_features_mask': np.ones((1, 3), dtype=np.bool_),
        }

    def video_processor(self, **kwargs):
        return {
            'pixel_values_videos': torch.ones(1, 3, 2, 2),
            'video_grids': torch.tensor([[1, 2, 2]]),
            'video_token_pooling': torch.tensor([[0, 1]]),
            'video_metadata': [SimpleNamespace(timestamps=[0.0])],
        }

    def get_image_tokens(self, image_grid):
        return ['image']

    def get_video_string(self, video_grid, timestamps):
        return 'video'


class _CogModel:
    dtype = torch.float32

    def __init__(self, cross_images):
        self.cross_images = cross_images

    def build_conversation_input_ids(self, processor, *, images, **kwargs):
        result = {'token_type_ids': torch.tensor([0, 1, 1, 0] if images else [0, 0])}
        if images:
            result['images'] = [torch.ones(3, 2, 2)]
            if self.cross_images:
                result['cross_images'] = [torch.ones(3, 4, 4)]
        return result


def _inputs(media):
    return SimpleNamespace(
        images=['image'] if 'image' in media else [],
        audios=[np.zeros(16)] if 'audio' in media else [],
        videos=['video'] if 'video' in media else [],
        messages=[{
            'role': 'user',
            'content': 'query'
        }],
        to_history=lambda: {
            'query': 'query',
            'history': []
        },
    )


@pytest.mark.parametrize('template_type,media', [
    ('paligemma', 'image'),
    ('gemma3_vision', 'image'),
    ('gemma3n', 'image'),
    ('gemma3n', 'audio'),
    ('gemma3n', 'image_audio'),
    ('molmo2', 'image'),
    ('molmo2', 'video'),
    ('molmo2', 'image_video'),
    ('cogvlm', 'image'),
    ('cogvlm2', 'image'),
    ('cogagent_chat', 'image'),
    ('cogagent_vqa', 'image'),
    ('cogvlm2_video', 'video'),
])
@pytest.mark.parametrize('mode', ['train', 'train_with_loss_scale', 'transformers'])
@pytest.mark.parametrize('strategy', ['left', 'right'])
def test_model_token_types_through_encoding_truncation_and_collation(template_type, media, mode, strategy):
    meta = TEMPLATE_MAPPING[template_type]
    template = meta.template_cls(None, meta, truncation_strategy=strategy)
    template.processor = _Processor(template_type)
    template.model_info = SimpleNamespace(torch_dtype=torch.float32)
    template.mode = 'transformers' if mode == 'transformers' else 'train'
    template.placeholder_tokens = template.placeholder_tokens.copy()
    template._init_placeholder_tokens()
    template.boi_token_id = 99
    template.boa_token_id = 98
    is_cog = template_type.startswith('cog')
    template.model = _CogModel(cross_images=template_type.startswith('cogagent'))

    placeholders = []
    expanded = []
    if not is_cog:
        if 'image' in media:
            placeholders += [90, 90] if template_type == 'paligemma' else [99]
            expanded += [90, 90] if template_type == 'paligemma' else [99, 90, 90, 89]
        if 'audio' in media:
            placeholders += [98]
            expanded += [98, 91, 91, 88]
        if 'video' in media:
            placeholders += [97]
            expanded += [95, 90, 90, 96]
    base_ids = [10, 11] + placeholders + [12, 13, 14, 15]
    full_ids = [10, 0, 0, 11, 12, 13, 14, 15] if is_cog else [10, 11] + expanded + [12, 13, 14, 15]
    base = {
        'input_ids': base_ids,
        'labels': [-100] * (len(base_ids) - 4) + [12, 13, 14, 15] if template.is_training else None,
        'loss_scale': [0.] * (len(base_ids) - 4) + [0.5, 1., 1.5, 2.] if mode == 'train_with_loss_scale' else None,
    }
    if template_type == 'paligemma':
        full_types = [0] * (len(full_ids) - 4) + [1] * 4 if template.is_training else [0] * len(full_ids)
    elif is_cog:
        full_types = [0, 1, 1, 0, 0, 0, 0, 0]
    else:
        full_types = [3 if token == 91 else int(token == 90) for token in full_ids]

    # Stub text tokenization and media I/O; run each model's own encoder and collator.
    with patch.object(template, '_preprocess_inputs'), patch.object(
            Template, '_encode', side_effect=lambda inputs: deepcopy(base)), patch(
                'swift.template.templates.glm.load_batch', side_effect=lambda paths, loader: paths):
        full = template._encode_truncated(_inputs(media))
        assert full['input_ids'] == full_ids
        assert full['token_type_ids'] == full_types
        if template.is_training:
            assert full['labels'] == [token if token in {12, 13, 14, 15} else -100 for token in full_ids]
        if mode == 'train_with_loss_scale':
            response_scales = {12: 0.5, 13: 1., 14: 1.5, 15: 2.}
            assert full['loss_scale'] == [response_scales.get(token, 0.) for token in full_ids]

        template.max_length = len(full_ids) - 2
        result = template._encode_truncated(_inputs(media))
        removed = {10, 11} if strategy == 'left' else {14, 15}
        kept = [i for i, token in enumerate(full_ids) if token not in removed]
        expected = {'input_ids': [full_ids[i] for i in kept], 'token_type_ids': [full_types[i] for i in kept]}
        for key, first in (('labels', -100), ('loss_scale', 0)):
            if full.get(key) is not None:
                expected[key] = [full[key][i] for i in kept]
                expected[key][0] = first
        for key, value in expected.items():
            assert result[key] == value, key
        assert result['length'] == len(kept)

        # A shorter text-only row must work before or after the multimodal row.
        base = {
            'input_ids': [10, 11, 12],
            'labels': [-100, -100, 12] if template.is_training else None,
            'loss_scale': [0., 0., 0.5] if mode == 'train_with_loss_scale' else None
        }
        short = template._encode_truncated(_inputs(''))
        short_types = [0, 0, 1] if template_type == 'paligemma' and template.is_training else [0, 0, 0]
        assert short['token_type_ids'] == short_types
        if template_type.startswith('cogagent'):
            # CogAgent cross-attention requires an image for every row.
            for rows in ([result, short], [short, result]):
                with pytest.raises(ValueError, match='CogAgent requires an image for every sample'):
                    template._data_collator(deepcopy(rows))
            short = template._encode_truncated(_inputs('image'))
            assert short['token_type_ids'] == [0, 1, 1, 0, 0]

    for rows in ([result, short], [short, result]):
        batch = template._data_collator(deepcopy(rows))
        for key, pad_value in (('input_ids', 0), ('token_type_ids', 0), ('labels', -100), ('loss_scale', 0)):
            if rows[0].get(key) is None:
                assert key not in batch
                continue
            expected_rows = []
            for row in rows:
                padding = [pad_value] * (len(kept) - len(row[key]))
                expected_rows.append(row[key] + padding if template.is_training else padding + row[key])
            assert batch[key].shape == (2, len(kept))
            assert batch[key].tolist() == expected_rows, key
        for key in ('pixel_values', 'pixel_values_videos', 'input_features', 'input_features_mask', 'image_grids',
                    'video_grids', 'image_token_pooling', 'video_token_pooling', 'image_num_crops'):
            if key in result:
                torch.testing.assert_close(batch[key], result[key])
        if is_cog:
            for key in ('images', 'cross_images'):
                if key not in result:
                    continue
                assert len(batch[key]) == 2
                for i, row in enumerate(rows):
                    assert len(batch[key][i]) == (1 if key in row else 0)
                    if key in row:
                        torch.testing.assert_close(batch[key][i][0], row[key][0][0])


@pytest.mark.parametrize('labels,expected_types', [
    (None, [0, 0, 0]),
    ([-100, -100, -100], [0, 0, 0]),
    ([-100, -100, 12], [0, 0, 1]),
    ([-100, 11, 12], [0, 1, 1]),
    ([10, 11, 12], [1, 1, 1]),
    ([], []),
])
def test_paligemma_prompt_answer_boundary(labels, expected_types):
    meta = TEMPLATE_MAPPING['paligemma']
    template = meta.template_cls(None, meta)
    template.processor = _Processor('paligemma')
    encoded = {'input_ids': [10, 11, 12][:len(expected_types)], 'labels': labels}
    with patch.object(Template, '_encode', return_value=encoded):
        result = template._encode(_inputs(''))
    assert result['token_type_ids'] == expected_types
