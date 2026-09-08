# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import sys
import types
import unittest
from io import BytesIO
from unittest.mock import patch

import swift.template.vision_utils as vision_utils


class TestAudioSsrf(unittest.TestCase):
    """The ffmpeg (audioread) fallback must never fetch a caller-controlled URL itself.

    Handing a URL straight to ffmpeg would let it re-resolve the host and follow its own redirects behind the
    SSRF guard, over any protocol ffmpeg supports. The fallback must therefore download through the guarded
    `load_file` and only ever open a local file. See `_load_audio_librosa`.
    """

    def setUp(self):
        # Stub librosa/audioread so the test does not need the real (heavy, optional) dependencies. The stubs
        # are injected after `vision_utils` is imported and carry a `__spec__` so importlib stays happy.
        self._seen = []

        librosa = types.ModuleType('librosa')
        librosa.__spec__ = None

        def _load(x, sr=None, mono=True):
            # librosa fails on the downloaded bytes, which is exactly what triggers the ffmpeg fallback.
            if isinstance(x, BytesIO):
                raise RuntimeError('librosa cannot decode; force ffmpeg fallback')
            return ('waveform', sr)

        librosa.load = _load

        audioread = types.ModuleType('audioread')
        audioread.__spec__ = None
        ffdec = types.ModuleType('audioread.ffdec')
        ffdec.__spec__ = None
        seen = self._seen

        class FakeFFmpegAudioFile:

            def __init__(self, path):
                # Record what ffmpeg was asked to open and whether it is a real local file.
                seen.append((path, os.path.isfile(path)))

        ffdec.FFmpegAudioFile = FakeFFmpegAudioFile
        audioread.ffdec = ffdec
        self._modules = {'librosa': librosa, 'audioread': audioread, 'audioread.ffdec': ffdec}

    def test_fallback_downloads_and_hands_ffmpeg_a_local_file(self):
        with patch.dict(sys.modules, self._modules):
            with patch.object(vision_utils, 'load_file', return_value=BytesIO(b'AUDIO')) as mock_load_file:
                res = vision_utils._load_audio_librosa('http://media.example.com/a.opus', 16000)
        self.assertEqual(res, ('waveform', 16000))
        self.assertTrue(mock_load_file.called)  # the URL was fetched through the guarded loader
        self.assertEqual(len(self._seen), 1)
        opened_path, is_file = self._seen[0]
        # ffmpeg must receive a real local file, never the URL itself.
        self.assertNotEqual(opened_path, 'http://media.example.com/a.opus')
        self.assertTrue(is_file)
        # The temporary file is cleaned up once decoding finishes.
        self.assertFalse(os.path.exists(opened_path))

    def test_blocked_url_never_reaches_ffmpeg(self):
        # `load_file` rejects the URL (as the SSRF guard does), so the fallback must propagate the error
        # without ever invoking ffmpeg.
        with patch.dict(sys.modules, self._modules):
            with patch.object(vision_utils, 'load_file', side_effect=ValueError('SSRF blocked')):
                with self.assertRaises(ValueError):
                    vision_utils._load_audio_librosa('http://169.254.169.254/latest/meta-data/', 16000)
        self.assertEqual(self._seen, [])

    def test_non_http_scheme_never_reaches_ffmpeg(self):
        # A scheme string ffmpeg would happily fetch (e.g. ftp://) must not be handed to ffmpeg as a "path".
        with patch.dict(sys.modules, self._modules):
            with patch.object(vision_utils, 'load_file', side_effect=OSError('not a local file')):
                with self.assertRaises((ValueError, OSError)):
                    vision_utils._load_audio_librosa('ftp://internal-service/secret', 16000)
        self.assertEqual(self._seen, [])


if __name__ == '__main__':
    unittest.main()
