import os
import torch
import unittest
from torch import nn
from types import SimpleNamespace
from unittest.mock import patch


class _FakeMossVLProcessor:

    def __init__(self):
        self.video_processor = SimpleNamespace(size={'shortest_edge': 4096, 'longest_edge': 201326592})
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return {
            'input_ids': torch.tensor([[1, 2]]),
            'labels': torch.tensor([[-100, 2]]),
            'pixel_values': torch.zeros(60, 768),
            'grid_thw': torch.tensor([[1, 6, 10]]),
            'cross_attention_mask': torch.ones(1, 1, 2, 1, dtype=torch.bool),
            'media_nums_per_sample': [1],
        }


class TestMossVLRegistration(unittest.TestCase):

    def test_registration(self):
        from swift.model import MODEL_MAPPING, MLLMModelType
        from swift.model.models.moss import MossVLLoader
        from swift.template import TEMPLATE_MAPPING, TemplateType

        model_meta = MODEL_MAPPING[MLLMModelType.moss_vl]
        self.assertIs(model_meta.loader, MossVLLoader)
        self.assertEqual(model_meta.template, TemplateType.moss_vl)
        self.assertEqual(model_meta.model_arch.arch_name, 'moss_vl')
        self.assertIn('MossVLForConditionalGeneration', model_meta.architectures)
        self.assertIn(TemplateType.moss_vl, TEMPLATE_MAPPING)

        model_ids = [model.hf_model_id for group in model_meta.model_groups for model in group.models]
        self.assertIn('OpenMOSS-Team/MOSS-VL-Instruct-0708', model_ids)
        self.assertIn('OpenMOSS-Team/MOSS-VL-Base-0708', model_ids)


class TestMossVLTemplate(unittest.TestCase):

    def test_template_state_and_unsupported_modes(self):
        from swift.template import TEMPLATE_MAPPING, TemplateType
        from swift.template.base import Template
        from swift.template.templates.moss import MossVLTemplate

        self.assertIn('placeholder_tokens', MossVLTemplate.__dict__)
        self.assertIsNot(MossVLTemplate.placeholder_tokens, Template.placeholder_tokens)

        template_meta = TEMPLATE_MAPPING[TemplateType.moss_vl]
        with self.assertRaisesRegex(ValueError, 'truncation_strategy="split"'):
            MossVLTemplate(None, template_meta, truncation_strategy='split')
        with self.assertRaisesRegex(NotImplementedError, 'does not support sequence parallel'):
            MossVLTemplate(None, template_meta, sequence_parallel_size=2)
        template = MossVLTemplate(None, template_meta)
        self.assertEqual(template.sequence_parallel_size, 1)

    def test_video_pixel_limits_are_routed_to_video_processor(self):
        from swift.template.templates.moss import MossVLTemplate

        processor = _FakeMossVLProcessor()
        template = object.__new__(MossVLTemplate)
        template.processor = processor
        template.chat_template_kwargs = {}
        template.mode = 'train'
        template.max_pixels = 1024
        inputs = SimpleNamespace(
            mm_processor_kwargs={},
            chat_template_kwargs={
                'video_min_pixels': 256,
                'video_max_pixels': 16384,
                'video_fps': 1.0,
                'max_frames': 32,
            },
            images=[],
            videos=['video.mp4'],
        )

        # Isolate the host environment so the default vision_chunked_length (64) is not
        # overridden by a MOSSVL_VISION_CHUNKED_LENGTH set on the host.
        env = {k: v for k, v in os.environ.items() if k.lower() != 'mossvl_vision_chunked_length'}
        with patch.dict(os.environ, env, clear=True), \
                patch.object(MossVLTemplate, '_render_text_and_spans', return_value=('prompt', [[0, 6]])):
            encoded = template._encode(inputs)

        self.assertNotIn('video_min_pixels', processor.kwargs)
        self.assertNotIn('video_max_pixels', processor.kwargs)
        self.assertEqual(processor.kwargs['max_pixels'], 1024)
        self.assertEqual(processor.kwargs['video_fps'], 1.0)
        self.assertEqual(processor.kwargs['max_frames'], 32)
        self.assertEqual(processor.kwargs['videos_kwargs']['size'], {
            'shortest_edge': 256,
            'longest_edge': 16384,
        })
        self.assertEqual(encoded['input_ids'], [1, 2])
        self.assertEqual(encoded['vision_chunked_length'], 64)

    def test_video_environment_limits_are_supported(self):
        from swift.template.templates.moss import MossVLTemplate

        processor = _FakeMossVLProcessor()
        template = object.__new__(MossVLTemplate)
        template.processor = processor
        template.chat_template_kwargs = {}
        template.mode = 'train'
        template.max_pixels = None
        inputs = SimpleNamespace(
            mm_processor_kwargs={},
            chat_template_kwargs={},
            images=[],
            videos=['video.mp4'],
        )
        environment = {
            'VIDEO_MIN_PIXELS': '256',
            'VIDEO_MAX_PIXELS': '16384',
            'FPS': '1.0',
            'FPS_MAX_FRAMES': '32',
        }

        with patch.dict(os.environ, environment, clear=True), \
                patch.object(MossVLTemplate, '_render_text_and_spans', return_value=('prompt', [[0, 6]])):
            template._encode(inputs)

        self.assertEqual(processor.kwargs['video_fps'], 1.0)
        self.assertEqual(processor.kwargs['max_frames'], 32)
        self.assertEqual(processor.kwargs['videos_kwargs']['size'], {
            'shortest_edge': 256,
            'longest_edge': 16384,
        })

    def test_truncation_preserves_media_contract(self):
        from swift.template.base import MaxLengthError
        from swift.template.templates.moss import MossVLTemplate

        class FakeTokenizer:
            unk_token_id = -1
            token_ids = {'<|image_pad|>': 20, '<|vision_start|>': 10, '<|vision_end|>': 30}

            def convert_tokens_to_ids(self, token):
                return self.token_ids[token]

        template = object.__new__(MossVLTemplate)
        template.processor = FakeTokenizer()
        template.max_length = 4
        input_ids = [10, 20, 30, 40, 50, 60]
        labels = [-100, -100, -100, 40, 50, 60]
        encoded = {'cross_attention_mask': torch.zeros(1, 1, 6, 2, dtype=torch.bool)}

        truncated_ids, truncated_labels = template._truncate(input_ids, labels, encoded, 'right')
        self.assertEqual(truncated_ids, [10, 20, 30, 40])
        self.assertEqual(truncated_labels, [-100, -100, -100, 40])
        self.assertEqual(tuple(encoded['cross_attention_mask'].shape), (1, 1, 4, 2))

        with self.assertRaises(MaxLengthError):
            template._truncate(input_ids, labels, {'cross_attention_mask': torch.zeros(1, 1, 6, 2)}, 'left')

    def test_cross_attention_masks_are_padded(self):
        from swift.template.base import Template
        from swift.template.templates.moss import MossVLTemplate

        template = object.__new__(MossVLTemplate)
        template.mode = 'train'
        template.padding_side = 'right'
        template.sequence_parallel_size = 1
        batch = [
            {
                'input_ids': [1, 2],
                'cross_attention_mask': torch.zeros(1, 1, 2, 1, dtype=torch.bool),
            },
            {
                'input_ids': [1, 2, 3, 4],
                'cross_attention_mask': torch.zeros(1, 1, 4, 3, dtype=torch.bool),
            },
        ]
        base_result = {
            'input_ids': torch.tensor([[1, 2, 0, 0], [1, 2, 3, 4]]),
        }

        with patch.object(Template, '_data_collator', return_value=base_result):
            result = template._data_collator(batch)

        mask = result['cross_attention_mask']
        self.assertEqual(tuple(mask.shape), (2, 1, 4, 3))
        self.assertFalse(mask[0, 0, :2, 0].any())
        self.assertTrue(mask[0, 0, :2, 1:].all())
        self.assertTrue(mask[0, 0, 2:, :].all())
        self.assertFalse(mask[1].any())

    def test_inference_cross_attention_masks_are_left_padded(self):
        from swift.template.base import Template
        from swift.template.templates.moss import MossVLTemplate

        template = object.__new__(MossVLTemplate)
        template.mode = 'transformers'
        template.padding_side = 'right'
        template.sequence_parallel_size = 1
        batch = [
            {
                'input_ids': [1, 2],
                'cross_attention_mask': torch.zeros(1, 1, 2, 2, dtype=torch.bool),
            },
            {
                'input_ids': [1, 2, 3, 4],
                'cross_attention_mask': torch.zeros(1, 1, 4, 3, dtype=torch.bool),
            },
        ]
        base_result = {
            'input_ids': torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]]),
            'attention_mask': torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]]),
        }

        with patch.object(Template, '_data_collator', return_value=base_result):
            result = template._data_collator(batch)

        mask = result['cross_attention_mask']
        self.assertEqual(tuple(mask.shape), (2, 1, 4, 3))
        self.assertTrue(mask[0, 0, :2].all())
        self.assertFalse(mask[0, 0, 2:, :2].any())
        self.assertTrue(mask[0, 0, 2:, 2].all())
        self.assertFalse(mask[1].any())

    def test_media_metadata_is_collated(self):
        from swift.template.base import Template
        from swift.template.templates.moss import MossVLTemplate

        template = object.__new__(MossVLTemplate)
        batch = [
            {
                'grid_thw': torch.tensor([[1, 2, 3]]),
                'media_nums_per_sample': [1],
                'vision_chunked_length': 64,
            },
            {
                'grid_thw': torch.tensor([[2, 3, 4], [1, 1, 1]]),
                'media_nums_per_sample': [2],
                'vision_chunked_length': 64,
            },
        ]

        with patch.object(Template, '_data_collator_mm_data', return_value={}):
            result = template._data_collator_mm_data(batch)

        self.assertEqual(result['grid_thw'].tolist(), [[1, 2, 3], [2, 3, 4], [1, 1, 1]])
        self.assertEqual(result['media_nums_per_sample'], [1, 2])
        self.assertEqual(result['vision_chunked_length'], 64)

    def test_mixed_media_and_text_only_batch_contract(self):
        from swift.template.base import Template
        from swift.template.templates.moss import MossVLTemplate

        template = object.__new__(MossVLTemplate)
        template.mode = 'train'
        template.padding_side = 'right'
        template.sequence_parallel_size = 1
        batch = [
            {
                'input_ids': [1, 2, 3],
                'grid_thw': torch.tensor([[1, 8, 8], [8, 6, 10]]),
                'media_nums_per_sample': [2],
                'vision_chunked_length': 64,
                'cross_attention_mask': torch.zeros(1, 1, 3, 5, dtype=torch.bool),
            },
            {
                'input_ids': [4, 5],
                # Text-only sample in the native contract: the processor injects a blank
                # image, so media_nums_per_sample is [1] and the cross_attention_mask is
                # fully masked (all True).
                'grid_thw': torch.tensor([[1, 8, 8]]),
                'media_nums_per_sample': [1],
                'vision_chunked_length': 64,
                'cross_attention_mask': torch.ones(1, 1, 2, 1, dtype=torch.bool),
            },
        ]
        with patch.object(Template, '_data_collator_mm_data', return_value={}):
            mm_result = template._data_collator_mm_data(batch)
        base_result = {
            **mm_result,
            'input_ids': torch.tensor([[1, 2, 3], [4, 5, 0]]),
        }

        with patch.object(Template, '_data_collator', return_value=base_result):
            result = template._data_collator(batch)

        self.assertEqual(result['media_nums_per_sample'], [2, 1])
        self.assertEqual(result['grid_thw'].tolist(), [[1, 8, 8], [8, 6, 10], [1, 8, 8]])
        self.assertEqual(result['vision_chunked_length'], 64)
        self.assertEqual(tuple(result['cross_attention_mask'].shape), (2, 1, 3, 5))
        self.assertTrue(result['cross_attention_mask'][1].all())


class TestMossVLLoRATargets(unittest.TestCase):

    def test_non_module_aligner_parameter_is_not_scanned_for_lora(self):
        from swift.utils import get_multimodal_target_regex

        class DummyModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.separator_token = nn.Parameter(torch.zeros(1))
                self.model_meta = SimpleNamespace(
                    model_arch=SimpleNamespace(language_model=[], vision_tower=[], aligner=['separator_token']))
                self.model_info = SimpleNamespace(is_moe_model=False)

        regex = get_multimodal_target_regex(DummyModel(), freeze_llm=True, freeze_vit=True, freeze_aligner=False)
        self.assertEqual(regex, '^()$')

    def test_bare_aligner_parameter_name_is_matched(self):
        from swift.tuner_plugin.lora_llm import is_vit_aligner_param

        model_arch = SimpleNamespace(vision_tower=['model.visual'], aligner=['model.separator_token'])
        self.assertTrue(is_vit_aligner_param(model_arch, 'model.separator_token'))
        self.assertTrue(is_vit_aligner_param(model_arch, 'base_model.model.model.separator_token'))
        self.assertFalse(is_vit_aligner_param(model_arch, 'model.language_model.embed_tokens.weight'))

    def test_separator_token_bc_property(self):
        from swift.model.models.moss import _patch_separator_token_property

        class Inner(nn.Module):

            def __init__(self):
                super().__init__()
                self.separator_token = nn.Parameter(torch.zeros(1))

        class Outer(nn.Module):

            def __init__(self):
                super().__init__()
                self.model = Inner()

        outer = Outer()
        _patch_separator_token_property(outer)
        # `outer.separator_token` must resolve to the inner parameter, so that attribute-path
        # lookups (e.g. the lora_llm unfreeze loop) reach it through the wrapper.
        self.assertIs(outer.separator_token, outer.model.separator_token)
        outer.separator_token.requires_grad_(True)
        self.assertTrue(outer.model.separator_token.requires_grad)
        # Idempotent and does not clobber a class that already provides the accessor.
        _patch_separator_token_property(outer)
        self.assertIs(outer.separator_token, outer.model.separator_token)


if __name__ == '__main__':
    unittest.main()
