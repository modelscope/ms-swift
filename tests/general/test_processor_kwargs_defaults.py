# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
import torch
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from swift.template.base import Template
from swift.template.templates import gemma as gemma_template_module
from swift.template.templates.gemma import Gemma3nTemplate, Gemma3Template, Gemma3VisionTemplate
from swift.template.templates.qwen import Qwen2_5OmniTemplate, Qwen3OmniTemplate


class TestQwenOmniProcessorDefaults(unittest.TestCase):

    def _make_template(self, cls):
        template = object.__new__(cls)
        return template

    def test_omni_v2_5_reads_processor_attributes_when_present(self):
        template = self._make_template(Qwen2_5OmniTemplate)
        processor = SimpleNamespace(
            feature_extractor=SimpleNamespace(sampling_rate=16000),
            seconds_per_chunk=3.5,
            position_id_per_seconds=20,
        )
        with patch.object(Template, 'init_processor', lambda self, proc: setattr(self, 'processor', proc)):
            with patch.dict(sys.modules, {'transformers.models.qwen2_5_omni.processing_qwen2_5_omni': None}):
                template.init_processor(processor)

        self.assertEqual(template.seconds_per_chunk, 3.5)
        self.assertEqual(template.position_id_per_seconds, 20)

    def test_omni_v3_reads_processor_attributes_when_present(self):
        template = self._make_template(Qwen3OmniTemplate)
        processor = SimpleNamespace(
            feature_extractor=SimpleNamespace(sampling_rate=16000),
            seconds_per_chunk=2.0,
            position_id_per_seconds=13.0,
        )
        with patch.object(Template, 'init_processor', lambda self, proc: setattr(self, 'processor', proc)):
            with patch.dict(sys.modules, {'transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe': None}):
                template.init_processor(processor)

        self.assertEqual(template.seconds_per_chunk, 2.0)
        self.assertEqual(template.position_id_per_seconds, 13.0)

    def test_omni_v2_5_falls_back_to_legacy_defaults(self):
        template = self._make_template(Qwen2_5OmniTemplate)
        processor = SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000))
        fake_mod = ModuleType('transformers.models.qwen2_5_omni.processing_qwen2_5_omni')
        fake_mod.Qwen2_5OmniProcessorKwargs = SimpleNamespace(
            _defaults={'videos_kwargs': {
                'seconds_per_chunk': 2.0,
                'position_id_per_seconds': 25
            }})

        with patch.object(Template, 'init_processor', lambda self, proc: setattr(self, 'processor', proc)):
            with patch.dict(sys.modules, {'transformers.models.qwen2_5_omni.processing_qwen2_5_omni': fake_mod}):
                template.init_processor(processor)

        self.assertEqual(template.seconds_per_chunk, 2.0)
        self.assertEqual(template.position_id_per_seconds, 25)

    def test_omni_v3_falls_back_to_legacy_defaults_and_disables_audio_truncation(self):
        template = self._make_template(Qwen3OmniTemplate)
        processor = SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000))
        defaults = {'videos_kwargs': {'seconds_per_chunk': 2.0, 'position_id_per_seconds': 13.0}}
        fake_mod = ModuleType('transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe')
        fake_mod.Qwen3OmniMoeProcessorKwargs = SimpleNamespace(_defaults=defaults)

        with patch.object(Template, 'init_processor', lambda self, proc: setattr(self, 'processor', proc)):
            with patch.dict(sys.modules, {'transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe': fake_mod}):
                template.init_processor(processor)

        self.assertEqual(template.seconds_per_chunk, 2.0)
        self.assertEqual(template.position_id_per_seconds, 13.0)
        self.assertFalse(defaults['audio_kwargs']['truncation'])

    def test_omni_ignores_bool_attributes_and_falls_back(self):
        template = self._make_template(Qwen2_5OmniTemplate)
        processor = SimpleNamespace(
            feature_extractor=SimpleNamespace(sampling_rate=16000),
            seconds_per_chunk=True,
            position_id_per_seconds=False,
        )
        fake_mod = ModuleType('transformers.models.qwen2_5_omni.processing_qwen2_5_omni')
        fake_mod.Qwen2_5OmniProcessorKwargs = SimpleNamespace(
            _defaults={'videos_kwargs': {
                'seconds_per_chunk': 2.0,
                'position_id_per_seconds': 25
            }})

        with patch.object(Template, 'init_processor', lambda self, proc: setattr(self, 'processor', proc)):
            with patch.dict(sys.modules, {'transformers.models.qwen2_5_omni.processing_qwen2_5_omni': fake_mod}):
                template.init_processor(processor)

        self.assertEqual(template.seconds_per_chunk, 2.0)
        self.assertEqual(template.position_id_per_seconds, 25)


class TestGemmaProcessorDefaults(unittest.TestCase):

    def _make_gemma3_vision_template(self, **processor_attrs):
        template = object.__new__(Gemma3VisionTemplate)
        template.boi_token_id = 255999
        template._tokenize = lambda _: [100]
        template._extend_tokens = lambda ids, labels, ls, idx, fn: (ids, labels, ls)
        template.processor = SimpleNamespace(
            full_image_sequence='<image>',
            image_token_id=262144,
            image_processor=MagicMock(return_value={
                'pixel_values': [[[0.0]]],
                'num_crops': [0]
            }),
            **processor_attrs,
        )
        return template

    def _make_gemma3n_template(self, **processor_attrs):
        template = object.__new__(Gemma3nTemplate)
        template.boi_token_id = 255999
        template.boa_token_id = 256000
        template.model_info = SimpleNamespace(torch_dtype=torch.float32)
        template._tokenize = lambda _: [100]
        template._extend_tokens = lambda ids, labels, ls, idx, fn: (ids, labels, ls)
        template.processor = SimpleNamespace(
            full_image_sequence='..<img>',
            full_audio_sequence='<audio>',
            image_token_id=10,
            audio_token_id=11,
            image_processor=MagicMock(return_value={'pixel_values': [[[0.0]]]}),
            feature_extractor=MagicMock(return_value={
                'input_features': [[0.0]],
                'input_features_mask': [[True]]
            }),
            **processor_attrs,
        )
        return template

    def test_gemma3_vision_reads_processor_images_kwargs_without_legacy_import(self):
        template = self._make_gemma3_vision_template(images_kwargs={'do_convert_rgb': True, 'do_pan_and_scan': False})
        encoded = {'input_ids': [255999], 'labels': [-100]}
        inputs = SimpleNamespace(images=['img'])

        with patch.dict(sys.modules, {'transformers.models.gemma3.processing_gemma3': None}):
            with patch.object(Gemma3Template, '_encode', return_value=encoded):
                template._encode(inputs)

        template.processor.image_processor.assert_called_once_with(['img'], do_convert_rgb=True, do_pan_and_scan=False)

    def test_gemma3_vision_falls_back_to_legacy_defaults(self):
        template = self._make_gemma3_vision_template()
        encoded = {'input_ids': [255999], 'labels': [-100]}
        inputs = SimpleNamespace(images=['img'])
        fake_mod = ModuleType('transformers.models.gemma3.processing_gemma3')
        fake_mod.Gemma3ProcessorKwargs = SimpleNamespace(_defaults={'images_kwargs': {'do_pan_and_scan': False}})

        with patch.object(gemma_template_module.transformers, '__version__', '5.22.0'):
            with patch.dict(sys.modules, {'transformers.models.gemma3.processing_gemma3': fake_mod}):
                with patch.object(Gemma3Template, '_encode', return_value=encoded):
                    template._encode(inputs)

        template.processor.image_processor.assert_called_once_with(['img'], do_pan_and_scan=False)

    def test_gemma3n_skips_legacy_import_on_transformers_ge_5_23(self):
        template = self._make_gemma3n_template()
        encoded = {'input_ids': [255999, 256000], 'labels': [-100, -100]}
        inputs = SimpleNamespace(images=['img'], audios=['aud'], messages=[{'role': 'user', 'content': 'hi'}])

        with patch.object(gemma_template_module.transformers, '__version__', '5.23.0'):
            with patch.dict(sys.modules, {'transformers.models.gemma3n.processing_gemma3n': None}):
                with patch.object(Gemma3Template, '_encode', return_value=encoded):
                    template._encode(inputs)

        template.processor.image_processor.assert_called_once_with(['img'])
        template.processor.feature_extractor.assert_called_once_with(['aud'])

    def test_gemma3n_reads_processor_kwargs_when_present(self):
        template = self._make_gemma3n_template(
            images_kwargs={'do_convert_rgb': True},
            audio_kwargs={'sampling_rate': 16000},
        )
        encoded = {'input_ids': [255999, 256000], 'labels': [-100, -100]}
        inputs = SimpleNamespace(images=['img'], audios=['aud'], messages=[{'role': 'user', 'content': 'hi'}])

        with patch.dict(sys.modules, {'transformers.models.gemma3n.processing_gemma3n': None}):
            with patch.object(Gemma3Template, '_encode', return_value=encoded):
                template._encode(inputs)

        template.processor.image_processor.assert_called_once_with(['img'], do_convert_rgb=True)
        template.processor.feature_extractor.assert_called_once_with(['aud'], sampling_rate=16000)

    def test_gemma3n_falls_back_to_legacy_defaults(self):
        template = self._make_gemma3n_template()
        encoded = {'input_ids': [255999, 256000], 'labels': [-100, -100]}
        inputs = SimpleNamespace(images=['img'], audios=['aud'], messages=[{'role': 'user', 'content': 'hi'}])
        fake_mod = ModuleType('transformers.models.gemma3n.processing_gemma3n')
        fake_mod.Gemma3nProcessorKwargs = SimpleNamespace(_defaults={
            'images_kwargs': {
                'do_resize': True
            },
            'audio_kwargs': {
                'padding': True
            },
        })

        with patch.object(gemma_template_module.transformers, '__version__', '5.22.0'):
            with patch.dict(sys.modules, {'transformers.models.gemma3n.processing_gemma3n': fake_mod}):
                with patch.object(Gemma3Template, '_encode', return_value=encoded):
                    template._encode(inputs)

        template.processor.image_processor.assert_called_once_with(['img'], do_resize=True)
        template.processor.feature_extractor.assert_called_once_with(['aud'], padding=True)


if __name__ == '__main__':
    unittest.main()
