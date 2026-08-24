# Copyright (c) ModelScope Contributors. All rights reserved.
import base64
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from swift.template.template_inputs import StdTemplateInputs
from swift.template.templates.minicpm import MiniCPMO4_5Template
from swift.template.vision_utils import local_video_path


class TestMiniCPMO45Template(unittest.TestCase):

    @staticmethod
    def _make_template(mode):
        template = object.__new__(MiniCPMO4_5Template)
        template.use_audio_in_video = True
        template.max_num_frames = 64
        template.mode = mode
        return template

    @staticmethod
    def _patch_minicpmo(get_video_frame_audio_segments):
        minicpmo = types.ModuleType('minicpmo')
        minicpmo_utils = types.ModuleType('minicpmo.utils')
        minicpmo_utils.get_video_frame_audio_segments = get_video_frame_audio_segments
        minicpmo.utils = minicpmo_utils
        return mock.patch.dict(sys.modules, {'minicpmo': minicpmo, 'minicpmo.utils': minicpmo_utils})

    @staticmethod
    def _make_data_uri(video_bytes=b'video-container-with-audio'):
        return f'data:video/mp4;base64,{base64.b64encode(video_bytes).decode()}'

    def test_vllm_data_uri_video_uses_temporary_file(self):
        video_bytes = b'video-container-with-audio'
        video = self._make_data_uri(video_bytes)
        captured = {}

        def get_video_frame_audio_segments(video_path, *, use_audio, stack_frames):
            captured['video_path'] = video_path
            self.assertTrue(os.path.isfile(video_path))
            with open(video_path, 'rb') as f:
                self.assertEqual(f.read(), video_bytes)
            self.assertTrue(use_audio)
            self.assertEqual(stack_frames, 1)
            return ['frame'], ['audio'], None

        template = self._make_template('vllm')
        inputs = StdTemplateInputs(messages=[], videos=[video])
        with self._patch_minicpmo(get_video_frame_audio_segments):
            context = template.replace_tag('video', 0, inputs)

        self.assertFalse(os.path.exists(captured['video_path']))
        self.assertEqual(inputs.videos, [])
        self.assertEqual(inputs.video_idx, -1)
        self.assertEqual(inputs.images, ['frame'])
        self.assertEqual(inputs.audios, ['audio'])
        self.assertEqual(context, ['(<image>./</image>)\n', '(<audio>./</audio>)'])
        self.assertTrue(all(not isinstance(part, list) or all(token_id >= 0 for token_id in part) for part in context))

    def test_vllm_multiple_videos_are_not_skipped(self):
        video = self._make_data_uri()

        def get_video_frame_audio_segments(video_path, **kwargs):
            return ['frame'], ['audio'], None

        template = self._make_template('vllm')
        inputs = StdTemplateInputs(messages=[], videos=[video, video])
        with self._patch_minicpmo(get_video_frame_audio_segments):
            # Match Template._pre_tokenize's post-callback index increment.
            for _ in range(2):
                template.replace_tag('video', inputs.video_idx, inputs)
                inputs.video_idx += 1

        self.assertEqual(inputs.videos, [])
        self.assertEqual(inputs.video_idx, 0)
        self.assertEqual(inputs.images, ['frame', 'frame'])
        self.assertEqual(inputs.audios, ['audio', 'audio'])

    def test_transformers_preserves_source_video(self):
        video = self._make_data_uri()

        def get_video_frame_audio_segments(video_path, **kwargs):
            return ['frame'], ['audio'], None

        template = self._make_template('transformers')
        inputs = StdTemplateInputs(messages=[], videos=[video])
        with self._patch_minicpmo(get_video_frame_audio_segments):
            context = template.replace_tag('video', 0, inputs)

        self.assertEqual(inputs.videos, [video])
        self.assertEqual(inputs.video_idx, 0)
        self.assertEqual(context, [[-100], '<|audio_start|><|audio_end|>'])

    def test_lmdeploy_consumes_source_video(self):
        video = self._make_data_uri()

        def get_video_frame_audio_segments(video_path, **kwargs):
            return ['frame'], ['audio'], None

        template = self._make_template('lmdeploy')
        inputs = StdTemplateInputs(messages=[], videos=[video])
        with self._patch_minicpmo(get_video_frame_audio_segments):
            context = template.replace_tag('video', 0, inputs)

        self.assertEqual(inputs.videos, [])
        self.assertEqual(inputs.video_idx, -1)
        self.assertEqual(inputs.images, ['frame'])
        self.assertEqual(inputs.audios, ['audio'])
        self.assertEqual(context, [[-100], '<|audio_start|><|audio_end|>'])

    def test_audio_placeholder_matches_backend(self):
        inputs = StdTemplateInputs(messages=[], audios=[object()])
        expected_contexts = {
            'vllm': ['(<audio>./</audio>)'],
            'lmdeploy': ['<|audio_start|><|audio_end|>'],
            'transformers': ['<|audio_start|><|audio_end|>'],
        }

        for mode, expected_context in expected_contexts.items():
            with self.subTest(mode=mode):
                template = self._make_template(mode)
                self.assertEqual(template.replace_tag('audio', 0, inputs), expected_context)

    def test_data_uri_video_cleans_up_temporary_file_on_error(self):
        video = self._make_data_uri(b'invalid-video')
        captured = {}

        def get_video_frame_audio_segments(video_path, **kwargs):
            captured['video_path'] = video_path
            self.assertTrue(os.path.isfile(video_path))
            raise RuntimeError('decode failed')

        template = self._make_template('vllm')
        inputs = StdTemplateInputs(messages=[], videos=[video])
        with self._patch_minicpmo(get_video_frame_audio_segments):
            with self.assertRaisesRegex(RuntimeError, 'decode failed'):
                template.replace_tag('video', 0, inputs)

        self.assertFalse(os.path.exists(captured['video_path']))
        self.assertEqual(inputs.videos, [video])

    def test_local_video_path_preserves_local_file(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(b'local-video')
            source_path = f.name
        try:
            with local_video_path(source_path) as resolved_path:
                self.assertEqual(resolved_path, os.path.abspath(source_path))
            self.assertTrue(os.path.exists(source_path))
        finally:
            os.remove(source_path)


if __name__ == '__main__':
    unittest.main()
