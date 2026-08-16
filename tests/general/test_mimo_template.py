# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from swift.template.template_inputs import StdTemplateInputs
from swift.template.templates.mimo import MiMoV2Template


class TestMiMoV2Template(unittest.TestCase):

    def setUp(self):
        self.template = object.__new__(MiMoV2Template)
        self.inputs = StdTemplateInputs(
            messages=[{
                'role': 'user',
                'content': '<audio>Transcribe this clip.'
            }], audios=['sample.wav'])

    def test_audio_placeholder_preserves_input(self):
        expected = ['<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>']
        qwen_vl_utils = SimpleNamespace(fetch_image=None, fetch_video=None)
        with patch.dict(sys.modules, {'qwen_vl_utils': qwen_vl_utils}):
            for mode in ['transformers', 'vllm', 'sglang', 'lmdeploy']:
                with self.subTest(mode=mode):
                    self.template.mode = mode
                    self.assertEqual(self.template.replace_tag('audio', 0, self.inputs), expected)
                    self.assertEqual(self.inputs.audios, ['sample.wav'])

    def test_audio_placeholder_is_protected_from_truncation(self):
        self.assertIn('<|audio_pad|>', self.template.placeholder_tokens)


if __name__ == '__main__':
    unittest.main()
