# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
import torch
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import transformers
from packaging import version

from swift.model.models.qwen import (_get_output_cache, _get_qwen3_omni_output_cls, _get_qwen3_vl_output_cls,
                                     _output_cache_kwarg)
from swift.template.templates.gemma import Gemma3VisionTemplate
from swift.template.templates.qwen import (_build_media_placeholder_text, _get_omni_video_defaults,
                                            _get_thinker_embed_tokens)
from swift.utils import deep_getattr, get_nested_submodule

transformers_5_23 = version.parse(transformers.__version__) >= version.parse('5.23')


class TestGemma3ImagesKwargs(unittest.TestCase):

    def test_prefers_processor_attribute(self):
        # transformers>=5.23 defines the processing defaults on the processor class.
        images_kwargs = {'size': {'height': 896, 'width': 896}}
        processor = SimpleNamespace(images_kwargs=images_kwargs)
        self.assertIs(Gemma3VisionTemplate._get_images_kwargs(processor), images_kwargs)

    def test_falls_back_to_processor_kwargs_defaults(self):
        from transformers.models.gemma3.processing_gemma3 import Gemma3ProcessorKwargs

        expected = Gemma3ProcessorKwargs._defaults.get('images_kwargs', {})
        self.assertEqual(Gemma3VisionTemplate._get_images_kwargs(SimpleNamespace()), expected)


class TestOmniVideoDefaults(unittest.TestCase):

    def test_reads_processor_attributes(self):
        processor = SimpleNamespace(seconds_per_chunk=2.0, position_id_per_seconds=13.0)
        self.assertEqual(_get_omni_video_defaults(processor, 'omni_v2_5'), (2.0, 13.0))
        self.assertEqual(_get_omni_video_defaults(processor, 'omni_v3'), (2.0, 13.0))

    def test_falls_back_to_processor_kwargs_defaults(self):
        defaults = {'videos_kwargs': {'seconds_per_chunk': 2.0, 'position_id_per_seconds': 25}}
        module = SimpleNamespace(Qwen2_5OmniProcessorKwargs=SimpleNamespace(_defaults=defaults))
        with patch.dict(sys.modules, {'transformers.models.qwen2_5_omni.processing_qwen2_5_omni': module}):
            self.assertEqual(_get_omni_video_defaults(SimpleNamespace(), 'omni_v2_5'), (2.0, 25))

    def test_v3_fallback_disables_audio_truncation(self):
        defaults = {
            'videos_kwargs': {
                'seconds_per_chunk': 2.0,
                'position_id_per_seconds': 13.0,
            },
            'audio_kwargs': {
                'truncation': True,
            },
        }
        module = SimpleNamespace(Qwen3OmniMoeProcessorKwargs=SimpleNamespace(_defaults=defaults))
        with patch.dict(sys.modules, {'transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe': module}):
            self.assertEqual(_get_omni_video_defaults(SimpleNamespace(), 'omni_v3'), (2.0, 13.0))
        self.assertFalse(defaults['audio_kwargs']['truncation'])


@unittest.skipUnless(transformers_5_23, 'transformers>=5.23 dropped rope_deltas from ModelOutput')
class TestQwen3OutputsWithoutRopeDeltas(unittest.TestCase):

    def test_qwen3_vl_output_cls(self):
        from transformers.modeling_outputs import BaseModelOutputWithPast

        self.assertIs(_get_qwen3_vl_output_cls(is_moe=False), BaseModelOutputWithPast)
        self.assertIs(_get_qwen3_vl_output_cls(is_moe=True), BaseModelOutputWithPast)
        self.assertNotIn('rope_deltas', BaseModelOutputWithPast.__dataclass_fields__)

    def test_qwen3_omni_output_cls(self):
        from transformers.modeling_outputs import MoeCausalLMOutputWithPast

        self.assertIs(_get_qwen3_omni_output_cls(), MoeCausalLMOutputWithPast)
        self.assertNotIn('rope_deltas', MoeCausalLMOutputWithPast.__dataclass_fields__)


class TestMovedHelpers(unittest.TestCase):

    def test_mm_submodule_top_level_layout(self):
        # transformers<5.23 keeps visual/audio_tower on the outer thinker class.
        visual = object()
        self.assertIs(get_nested_submodule(SimpleNamespace(visual=visual), 'visual'), visual)

    def test_mm_submodule_nested_thinker_layout(self):
        # transformers>=5.23 nests visual/audio_tower under `model.model`.
        visual = object()
        model = SimpleNamespace(model=SimpleNamespace(visual=visual))
        self.assertIs(get_nested_submodule(model, 'visual'), visual)

    def test_mm_submodule_missing_raises(self):
        with self.assertRaises(AttributeError):
            get_nested_submodule(SimpleNamespace(), 'visual')

    def test_output_cache_field_name(self):
        from transformers.modeling_outputs import BaseModelOutputWithPast

        fields = BaseModelOutputWithPast.__dataclass_fields__
        expected_key = 'cache' if 'cache' in fields else 'past_key_values'
        self.assertEqual(_output_cache_kwarg(BaseModelOutputWithPast, 'c'), {expected_key: 'c'})

    def test_get_output_cache_prefers_new_name(self):
        self.assertEqual(_get_output_cache(SimpleNamespace(cache='c')), 'c')
        self.assertEqual(_get_output_cache(SimpleNamespace(past_key_values='p')), 'p')
        self.assertIsNone(_get_output_cache(SimpleNamespace()))

    def test_media_placeholder_counts(self):
        processor = SimpleNamespace(audio_token='<a>', image_token='<i>', video_token='<v>')
        text = _build_media_placeholder_text(processor, audios=[None, None], images=[None], videos=[None] * 3)
        self.assertEqual(text, '<a><a><i><v><v><v>')
        self.assertEqual(text.count('<a>'), 2)
        self.assertEqual(text.count('<i>'), 1)
        self.assertEqual(text.count('<v>'), 3)

    def test_media_placeholder_without_tokens(self):
        self.assertEqual(_build_media_placeholder_text(SimpleNamespace(), [], [], []), '')
        with self.assertRaises(AttributeError):
            _build_media_placeholder_text(SimpleNamespace(), [None], [], [])

    def test_get_keys_to_not_convert_importable(self):
        try:
            from transformers.quantizers.base import get_keys_to_not_convert
        except ImportError:
            from transformers.integrations import get_keys_to_not_convert
        self.assertTrue(callable(get_keys_to_not_convert))


class TestOmniThinkerLayout(unittest.TestCase):

    def test_thinker_embed_tokens_nested_layout(self):
        embed = object()
        thinker = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(embed_tokens=embed)))
        self.assertIs(_get_thinker_embed_tokens(thinker), embed)

    def test_thinker_embed_tokens_legacy_layout(self):
        embed = object()
        thinker = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed))
        self.assertIs(_get_thinker_embed_tokens(thinker), embed)


class TestOmniLoaderStructure(unittest.TestCase):

    @staticmethod
    def _fake_model():
        visual, audio_tower = object(), object()
        thinker = SimpleNamespace(model=SimpleNamespace(visual=visual, audio_tower=audio_tower))
        config = SimpleNamespace(keys_to_ignore_at_inference=[], talker_config=SimpleNamespace(pad_token_id=0))
        return SimpleNamespace(thinker=thinker, config=config), visual, audio_tower

    @staticmethod
    def _fake_legacy_model():
        visual, audio_tower = object(), object()
        thinker = SimpleNamespace(visual=visual, audio_tower=audio_tower)
        config = SimpleNamespace(keys_to_ignore_at_inference=[], talker_config=SimpleNamespace(pad_token_id=0))
        return SimpleNamespace(thinker=thinker, config=config), visual, audio_tower

    def test_qwen2_5_omni_loader_uses_nested_visual(self):
        from swift.model.models.qwen import Qwen2_5OmniLoader
        from swift.model.register import ModelLoader

        model, visual, _ = self._fake_model()
        calls = []
        with patch.object(ModelLoader, 'get_model', return_value=model), \
                patch('swift.model.models.qwen.use_submodel_func'), \
                patch('swift.model.models.qwen.patch_get_input_embeddings',
                      side_effect=lambda module, name: calls.append((module, name))):
            loader = object.__new__(Qwen2_5OmniLoader)
            loader.auto_model_cls = None
            loader.get_model('dummy')
        self.assertEqual(calls, [(visual, 'patch_embed')])

    def test_qwen3_omni_loader_uses_nested_visual_and_audio(self):
        from swift.model.models.qwen import Qwen3OmniLoader
        from swift.model.register import ModelLoader

        model, visual, audio_tower = self._fake_model()
        calls = []
        with patch.object(ModelLoader, 'get_model', return_value=model), \
                patch('swift.model.models.qwen.use_submodel_func'), \
                patch('swift.model.models.qwen._compat_qwen3_omni_mixed_data'), \
                patch('swift.model.models.qwen.patch_get_input_embeddings',
                      side_effect=lambda module, name: calls.append((module, name))):
            loader = object.__new__(Qwen3OmniLoader)
            loader.auto_model_cls = None
            loader.get_model('dummy', SimpleNamespace(), None, {})
        self.assertEqual(calls, [(visual, 'patch_embed'), (audio_tower, 'conv_out')])

    def test_qwen2_5_omni_loader_legacy_layout(self):
        from swift.model.models.qwen import Qwen2_5OmniLoader
        from swift.model.register import ModelLoader

        model, visual, _ = self._fake_legacy_model()
        calls = []
        with patch.object(ModelLoader, 'get_model', return_value=model), \
                patch('swift.model.models.qwen.use_submodel_func'), \
                patch('swift.model.models.qwen.patch_get_input_embeddings',
                      side_effect=lambda module, name: calls.append((module, name))):
            loader = object.__new__(Qwen2_5OmniLoader)
            loader.auto_model_cls = None
            loader.get_model('dummy')
        self.assertEqual(calls, [(visual, 'patch_embed')])

    def test_qwen3_omni_loader_legacy_layout(self):
        from swift.model.models.qwen import Qwen3OmniLoader
        from swift.model.register import ModelLoader

        model, visual, audio_tower = self._fake_legacy_model()
        calls = []
        with patch.object(ModelLoader, 'get_model', return_value=model), \
                patch('swift.model.models.qwen.use_submodel_func'), \
                patch('swift.model.models.qwen._compat_qwen3_omni_mixed_data'), \
                patch('swift.model.models.qwen.patch_get_input_embeddings',
                      side_effect=lambda module, name: calls.append((module, name))):
            loader = object.__new__(Qwen3OmniLoader)
            loader.auto_model_cls = None
            loader.get_model('dummy', SimpleNamespace(), None, {})
        self.assertEqual(calls, [(visual, 'patch_embed'), (audio_tower, 'conv_out')])


class TestOmniPostEncodeStructure(unittest.TestCase):

    def _post_encode(self, thinker):
        from swift.template.templates.qwen import Qwen2_5OmniTemplate

        template = object.__new__(Qwen2_5OmniTemplate)
        template.mode = 'train'
        template.processor = None
        template._get_inputs_embeds_hf = lambda embeds, *args, **kwargs: embeds
        model = SimpleNamespace(thinker=thinker, config=SimpleNamespace(thinker_config=SimpleNamespace()))
        return template._post_encode(model, {'input_ids': torch.tensor([[1, 2]])})['inputs_embeds']

    def test_post_encode_nested_layout(self):
        embed = lambda ids: ids.float()  # noqa: E731
        thinker = SimpleNamespace(
            model=SimpleNamespace(
                visual=object(),
                audio_tower=object(),
                language_model=SimpleNamespace(embed_tokens=embed),
            ))
        self.assertEqual(tuple(self._post_encode(thinker).shape), (1, 2))

    def test_post_encode_legacy_layout(self):
        embed = lambda ids: ids.float()  # noqa: E731
        thinker = SimpleNamespace(
            model=SimpleNamespace(embed_tokens=embed), visual=object(), audio_tower=object())
        self.assertEqual(tuple(self._post_encode(thinker).shape), (1, 2))


def _tiny_qwen3_omni_thinker():
    from transformers import Qwen3OmniMoeThinkerConfig, Qwen3OmniMoeThinkerForConditionalGeneration

    cfg = Qwen3OmniMoeThinkerConfig(
        text_config=dict(hidden_size=32, intermediate_size=64, moe_intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, num_experts=4, num_experts_per_tok=2,
                         decoder_sparse_step=1, vocab_size=128, max_position_embeddings=128, mlp_only_layers=[]),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4, in_channels=3,
                           out_hidden_size=32, patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
                           deepstack_visual_indexes=[], num_position_embeddings=16),
        audio_config=dict(d_model=32, encoder_layers=1, encoder_attention_heads=4, encoder_ffn_dim=64,
                          num_mel_bins=128, max_source_positions=32, output_dim=32, downsample_hidden_size=32,
                          scale_embedding=True, n_window=50, n_window_infer=200),
        image_token_id=100, video_token_id=101, audio_token_id=102,
        vision_start_token_id=103, vision_end_token_id=104, audio_start_token_id=105, audio_end_token_id=106)
    return Qwen3OmniMoeThinkerForConditionalGeneration(cfg).eval()


def _tiny_qwen25_omni_thinker():
    from transformers import Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration

    cfg = Qwen2_5OmniThinkerConfig(
        text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=128, max_position_embeddings=128),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4, in_channels=3,
                           out_hidden_size=32, patch_size=14, temporal_patch_size=2, spatial_merge_size=2),
        audio_config=dict(d_model=32, encoder_layers=1, encoder_attention_heads=4, encoder_ffn_dim=64,
                          num_mel_bins=128, max_source_positions=32, output_dim=32, downsample_hidden_size=32,
                          scale_embedding=True),
        image_token_id=100, video_token_id=101, audio_token_id=102,
        vision_start_token_id=103, vision_end_token_id=104, audio_start_token_id=105, audio_end_token_id=106)
    return Qwen2_5OmniThinkerForConditionalGeneration(cfg).eval()


class TestOmniModelArch(unittest.TestCase):

    def _assert_paths(self, thinker, arch):
        top = SimpleNamespace(thinker=thinker)
        for path in arch.vision_tower + arch.aligner + arch.language_model:
            self.assertIsNotNone(deep_getattr(top, path, None), f'missing arch path: {path}')

    def test_qwen2_5_omni_paths(self):
        from swift.model.model_arch import MODEL_ARCH_MAPPING, MLLMModelArch

        self._assert_paths(_tiny_qwen25_omni_thinker(), MODEL_ARCH_MAPPING[MLLMModelArch.qwen2_5_omni])

    def test_qwen3_omni_paths(self):
        from swift.model.model_arch import MODEL_ARCH_MAPPING, MLLMModelArch

        self._assert_paths(_tiny_qwen3_omni_thinker(), MODEL_ARCH_MAPPING[MLLMModelArch.qwen3_omni])


class TestCacheAlias(unittest.TestCase):

    def test_vl_wrapper_maps_cache_to_past_key_values(self):
        from swift.model.models.qwen import _compat_qwen3_vl_mixed_data

        captured = {}

        class Origin:

            def __call__(self, **kwargs):
                captured.update(kwargs)
                return 'origin-ok'

        model = SimpleNamespace(
            forward=Origin(),
            language_model=SimpleNamespace(),
            visual=SimpleNamespace(forward=lambda *args, **kwargs: None),
        )
        with patch('swift.model.models.qwen.is_deepspeed_enabled', return_value=False):
            _compat_qwen3_vl_mixed_data(model, None)
            out = model.forward(cache='CACHE')
        self.assertEqual(out, 'origin-ok')
        self.assertEqual(captured['past_key_values'], 'CACHE')

    def test_omni_wrapper_maps_cache_to_past_key_values(self):
        from swift.model.models.qwen import _compat_qwen3_omni_mixed_data

        captured = {}

        class Origin:

            def __call__(self, **kwargs):
                captured.update(kwargs)
                return 'omni-origin-ok'

        model = SimpleNamespace(
            forward=Origin(),
            model=SimpleNamespace(language_model=SimpleNamespace()),
            config=SimpleNamespace(use_cache=False, return_dict=True),
        )
        with patch('swift.model.models.qwen.is_deepspeed_enabled', return_value=False):
            _compat_qwen3_omni_mixed_data(model, None)
            out = model.forward(cache='CACHE')
        self.assertEqual(out, 'omni-origin-ok')
        self.assertEqual(captured['past_key_values'], 'CACHE')


if __name__ == '__main__':
    unittest.main()
