# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import transformers
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from packaging import version

from swift.model.models.qwen import _get_qwen3_omni_output_cls, _get_qwen3_vl_output_cls
from swift.template.templates.qwen import _build_media_placeholder_text
from swift.utils import deep_getattr

transformers_5_23 = version.parse(transformers.__version__) >= version.parse('5.23')


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


@unittest.skipUnless(transformers_5_23, 'transformers>=5.23 dropped rope_deltas from ModelOutput')
class TestQwen3OutputClasses(unittest.TestCase):

    def test_qwen3_vl_output_cls(self):
        from transformers.modeling_outputs import BaseModelOutputWithPast

        self.assertIs(_get_qwen3_vl_output_cls(is_moe=False), BaseModelOutputWithPast)
        self.assertIs(_get_qwen3_vl_output_cls(is_moe=True), BaseModelOutputWithPast)
        self.assertNotIn('rope_deltas', BaseModelOutputWithPast.__dataclass_fields__)

    def test_qwen3_omni_output_cls(self):
        from transformers.modeling_outputs import MoeCausalLMOutputWithPast

        self.assertIs(_get_qwen3_omni_output_cls(), MoeCausalLMOutputWithPast)
        self.assertNotIn('rope_deltas', MoeCausalLMOutputWithPast.__dataclass_fields__)


class TestOmniMediaPlaceholder(unittest.TestCase):

    def test_counts_and_missing_tokens(self):
        processor = SimpleNamespace(audio_token='<a>', image_token='<i>', video_token='<v>')
        text = _build_media_placeholder_text(processor, audios=[None, None], images=[None], videos=[None] * 3)
        self.assertEqual(text, '<a><a><i><v><v><v>')
        self.assertEqual(_build_media_placeholder_text(SimpleNamespace(), [], [], []), '')
        with self.assertRaises(AttributeError):
            _build_media_placeholder_text(SimpleNamespace(), [None], [], [])


class TestOmniLoaderStructure(unittest.TestCase):

    @staticmethod
    def _fake_model(layout):
        visual, audio_tower = object(), object()
        if layout == 'nested':
            thinker = SimpleNamespace(model=SimpleNamespace(visual=visual, audio_tower=audio_tower))
        else:
            thinker = SimpleNamespace(visual=visual, audio_tower=audio_tower)
        config = SimpleNamespace(keys_to_ignore_at_inference=[], talker_config=SimpleNamespace(pad_token_id=0))
        return SimpleNamespace(thinker=thinker, config=config), visual, audio_tower

    def test_qwen2_5_loader_layouts(self):
        from swift.model.models.qwen import Qwen2_5OmniLoader
        from swift.model.register import ModelLoader

        for layout in ('nested', 'legacy'):
            with self.subTest(layout=layout):
                model, visual, _ = self._fake_model(layout)
                calls = []
                with patch.object(ModelLoader, 'get_model', return_value=model), \
                        patch('swift.model.models.qwen.use_submodel_func'), \
                        patch('swift.model.models.qwen.patch_get_input_embeddings',
                              side_effect=lambda module, name: calls.append((module, name))):
                    loader = object.__new__(Qwen2_5OmniLoader)
                    loader.auto_model_cls = None
                    loader.get_model('dummy')
                self.assertEqual(calls, [(visual, 'patch_embed')])

    def test_qwen3_loader_layouts(self):
        from swift.model.models.qwen import Qwen3OmniLoader
        from swift.model.register import ModelLoader

        for layout in ('nested', 'legacy'):
            with self.subTest(layout=layout):
                model, visual, audio_tower = self._fake_model(layout)
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

    @staticmethod
    def _post_encode(thinker):
        from swift.template.templates.qwen import Qwen2_5OmniTemplate

        template = object.__new__(Qwen2_5OmniTemplate)
        template.mode = 'train'
        template.processor = None
        template._get_inputs_embeds_hf = lambda embeds, *args, **kwargs: embeds
        model = SimpleNamespace(thinker=thinker, config=SimpleNamespace(thinker_config=SimpleNamespace()))
        with patch('swift.template.templates.qwen.is_deepspeed_enabled', return_value=False):
            return template._post_encode(model, {'input_ids': torch.tensor([[1, 2]])})['inputs_embeds']

    def test_layouts(self):
        embed = lambda ids: ids.float()  # noqa: E731
        layouts = {
            'nested':
            SimpleNamespace(
                model=SimpleNamespace(
                    visual=object(), audio_tower=object(), language_model=SimpleNamespace(embed_tokens=embed))),
            'legacy':
            SimpleNamespace(model=SimpleNamespace(embed_tokens=embed), visual=object(), audio_tower=object()),
        }
        for layout, thinker in layouts.items():
            with self.subTest(layout=layout):
                self.assertEqual(tuple(self._post_encode(thinker).shape), (1, 2))


class TestOmniModelArch(unittest.TestCase):

    def test_paths_resolve_on_tiny_thinkers(self):
        from swift.model.model_arch import MODEL_ARCH_MAPPING, MLLMModelArch

        cases = {
            MLLMModelArch.qwen2_5_omni: _tiny_qwen25_omni_thinker,
            MLLMModelArch.qwen3_omni: _tiny_qwen3_omni_thinker,
        }
        for arch_name, builder in cases.items():
            with self.subTest(arch=arch_name):
                top = SimpleNamespace(thinker=builder())
                arch = MODEL_ARCH_MAPPING[arch_name]
                for path in arch.vision_tower + arch.aligner + arch.language_model:
                    self.assertIsNotNone(deep_getattr(top, path, None), f'missing arch path: {path}')


class TestCacheAlias(unittest.TestCase):

    @staticmethod
    def _patch_wrapper(kind):
        captured = {}

        class Origin:

            def __call__(self, **kwargs):
                captured.update(kwargs)
                return 'origin-ok'

        if kind == 'vl':
            from swift.model.models.qwen import _compat_qwen3_vl_mixed_data

            model = SimpleNamespace(
                forward=Origin(),
                language_model=SimpleNamespace(),
                visual=SimpleNamespace(forward=lambda *args, **kwargs: None),
            )
            patch_fn = _compat_qwen3_vl_mixed_data
        else:
            from swift.model.models.qwen import _compat_qwen3_omni_mixed_data

            model = SimpleNamespace(
                forward=Origin(),
                model=SimpleNamespace(language_model=SimpleNamespace()),
                config=SimpleNamespace(use_cache=False, return_dict=True),
            )
            patch_fn = _compat_qwen3_omni_mixed_data

        with patch('swift.model.models.qwen.is_deepspeed_enabled', return_value=False):
            patch_fn(model, None)
            out = model.forward(cache='CACHE')
        return out, captured

    def test_wrappers_map_cache_to_past_key_values(self):
        for kind in ('vl', 'omni'):
            with self.subTest(wrapper=kind):
                out, captured = self._patch_wrapper(kind)
                self.assertEqual(out, 'origin-ok')
                self.assertEqual(captured['past_key_values'], 'CACHE')


if __name__ == '__main__':
    unittest.main()
