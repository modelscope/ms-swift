import os
import tempfile
import unittest

from swift.template.vision_utils import _check_path

ENV_KEY = 'SWIFT_MEDIA_ALLOWED_DIRS'


class TestMediaAllowedDirs(unittest.TestCase):
    """`_check_path` resolves untrusted strings against the filesystem, so it must honour the allowlist."""

    def setUp(self):
        os.environ.pop(ENV_KEY, None)
        self.tmp_dir = tempfile.mkdtemp()
        self.allowed_dir = os.path.join(self.tmp_dir, 'allowed')
        os.makedirs(self.allowed_dir)
        self.inside = os.path.join(self.allowed_dir, 'img.png')
        with open(self.inside, 'wb') as f:
            f.write(b'inside')
        self.outside = os.path.join(self.tmp_dir, 'secret.png')
        with open(self.outside, 'wb') as f:
            f.write(b'outside')

    def tearDown(self):
        os.environ.pop(ENV_KEY, None)

    def test_unset_allowlist_keeps_current_behavior(self):
        self.assertEqual(_check_path(self.outside), self.outside)

    def test_path_inside_allowed_dir_is_returned(self):
        os.environ[ENV_KEY] = self.allowed_dir
        self.assertEqual(_check_path(self.inside), self.inside)

    def test_path_outside_allowed_dir_is_refused(self):
        os.environ[ENV_KEY] = self.allowed_dir
        with self.assertRaises(ValueError):
            _check_path(self.outside)

    def test_traversal_out_of_allowed_dir_is_refused(self):
        os.environ[ENV_KEY] = self.allowed_dir
        with self.assertRaises(ValueError):
            _check_path(os.path.join(self.allowed_dir, '..', 'secret.png'))

    def test_symlink_out_of_allowed_dir_is_refused(self):
        os.environ[ENV_KEY] = self.allowed_dir
        link = os.path.join(self.allowed_dir, 'link.png')
        os.symlink(self.outside, link)
        with self.assertRaises(ValueError):
            _check_path(link)

    def test_multiple_allowed_dirs(self):
        other_dir = os.path.join(self.tmp_dir, 'other')
        os.makedirs(other_dir)
        other = os.path.join(other_dir, 'img.png')
        with open(other, 'wb') as f:
            f.write(b'other')
        os.environ[ENV_KEY] = f'{self.allowed_dir}, {other_dir}'
        self.assertEqual(_check_path(self.inside), self.inside)
        self.assertEqual(_check_path(other), other)
        with self.assertRaises(ValueError):
            _check_path(self.outside)

    def test_sibling_dir_with_shared_prefix_is_refused(self):
        """`/data/allowed-evil` must not be treated as living inside `/data/allowed`."""
        sibling = self.allowed_dir + '-evil'
        os.makedirs(sibling)
        path = os.path.join(sibling, 'img.png')
        with open(path, 'wb') as f:
            f.write(b'sibling')
        os.environ[ENV_KEY] = self.allowed_dir
        with self.assertRaises(ValueError):
            _check_path(path)

    def test_base64_input_is_unaffected(self):
        os.environ[ENV_KEY] = self.allowed_dir
        self.assertIsNone(_check_path('aGVsbG8gd29ybGQ='))

    def test_missing_path_outside_allowed_dir_is_refused_alike(self):
        """An outside path must be refused the same way whether or not it exists, or the error leaks that."""
        os.environ[ENV_KEY] = self.allowed_dir
        missing = os.path.join(self.tmp_dir, 'no-such-file.png')
        with self.assertRaises(ValueError) as existing_ctx:
            _check_path(self.outside)
        with self.assertRaises(ValueError) as missing_ctx:
            _check_path(missing)
        self.assertEqual(
            str(existing_ctx.exception).replace(self.outside, ''),
            str(missing_ctx.exception).replace(missing, ''),
        )


class TestRequestMediaPath(unittest.TestCase):
    """A request inlines a local media path before the template runs, so it must honour the allowlist too."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.allowed_dir = os.path.join(self.tmp_dir, 'allowed')
        os.makedirs(self.allowed_dir)
        self.inside = os.path.join(self.allowed_dir, 'img.png')
        with open(self.inside, 'wb') as f:
            f.write(b'inside')
        self.outside = os.path.join(self.tmp_dir, 'secret.png')
        with open(self.outside, 'wb') as f:
            f.write(b'outside')
        os.environ[ENV_KEY] = self.allowed_dir

    def tearDown(self):
        os.environ.pop(ENV_KEY, None)

    @staticmethod
    def _request(url):
        from swift.infer_engine.protocol import ChatCompletionRequest
        return ChatCompletionRequest(
            model='m', messages=[{
                'role': 'user',
                'content': [{
                    'type': 'image_url',
                    'image_url': {
                        'url': url
                    }
                }]
            }])

    def test_path_outside_allowed_dir_is_not_inlined(self):
        with self.assertRaises(ValueError):
            self._request(self.outside)

    def test_missing_path_outside_allowed_dir_is_refused_alike(self):
        missing = os.path.join(self.tmp_dir, 'no-such-file.png')
        with self.assertRaises(ValueError) as existing_ctx:
            self._request(self.outside)
        with self.assertRaises(ValueError) as missing_ctx:
            self._request(missing)
        self.assertEqual(
            str(existing_ctx.exception).replace(self.outside, ''),
            str(missing_ctx.exception).replace(missing, ''),
        )

    def test_top_level_media_field_outside_allowed_dir_is_refused(self):
        from swift.infer_engine.protocol import ChatCompletionRequest
        with self.assertRaises(ValueError):
            ChatCompletionRequest(model='m', messages=[{'role': 'user', 'content': 'hi'}], images=[self.outside])

    def test_path_inside_allowed_dir_is_inlined(self):
        request = self._request(self.inside)
        self.assertTrue(request.messages[0]['content'][0]['image_url']['url'].startswith('data:'))


class TestSafeMediaInput(unittest.TestCase):
    """Templates delegate loading to third-party fetchers, so `_safe_media_input` must guard the value first."""

    def setUp(self):
        os.environ.pop(ENV_KEY, None)
        self.tmp_dir = tempfile.mkdtemp()
        self.allowed_dir = os.path.join(self.tmp_dir, 'allowed')
        os.makedirs(self.allowed_dir)
        self.inside = os.path.join(self.allowed_dir, 'img.png')
        with open(self.inside, 'wb') as f:
            f.write(b'inside')
        self.outside = os.path.join(self.tmp_dir, 'secret.png')
        with open(self.outside, 'wb') as f:
            f.write(b'outside')

    def tearDown(self):
        os.environ.pop(ENV_KEY, None)

    def test_base64_and_data_uri_pass_through_unchanged(self):
        from swift.template.vision_utils import _safe_media_input
        for value in ('aGVsbG8gd29ybGQ=', 'data:image/png;base64,aGVsbG8='):
            self.assertEqual(_safe_media_input(value), value)

    def test_non_string_values_pass_through_unchanged(self):
        """An already-loaded image / a frame list must be handed to the fetcher untouched."""
        from swift.template.vision_utils import _safe_media_input
        frame_list = [self.inside, self.inside]
        self.assertIs(_safe_media_input(frame_list), frame_list)
        sentinel = object()
        self.assertIs(_safe_media_input(sentinel), sentinel)

    def test_local_path_passes_through_but_honours_allowlist(self):
        from swift.template.vision_utils import _safe_media_input
        self.assertEqual(_safe_media_input(self.inside), self.inside)  # no allowlist: unchanged
        os.environ[ENV_KEY] = self.allowed_dir
        self.assertEqual(_safe_media_input(self.inside), self.inside)
        with self.assertRaises(ValueError):
            _safe_media_input(self.outside)

    def test_url_is_routed_through_the_ssrf_guard(self):
        """A metadata-endpoint URL must be refused instead of fetched by the downstream library."""
        from swift.template.vision_utils import _safe_media_input
        with self.assertRaises(ValueError):
            _safe_media_input('http://169.254.169.254/latest/meta-data/')

    def test_uppercase_scheme_is_still_guarded(self):
        """A mixed-case scheme must not skip the URL branch and reach the decoder as a raw path."""
        from swift.template.vision_utils import _safe_media_input
        with self.assertRaises(ValueError):
            _safe_media_input('HTTP://169.254.169.254/latest/meta-data/')

    def test_video_local_path_and_base64_pass_through(self):
        from swift.template.vision_utils import _safe_media_input
        os.environ[ENV_KEY] = self.allowed_dir
        self.assertEqual(_safe_media_input(self.inside, is_video=True), self.inside)
        self.assertEqual(_safe_media_input('aGVsbG8=', is_video=True), 'aGVsbG8=')
        with self.assertRaises(ValueError):
            _safe_media_input(self.outside, is_video=True)


class TestSafeVideoInput(unittest.TestCase):
    """`fetch_video` loads a frame list with its own unpatched `fetch_image`, so the list must be guarded."""

    def setUp(self):
        os.environ.pop(ENV_KEY, None)
        self.tmp_dir = tempfile.mkdtemp()
        self.allowed_dir = os.path.join(self.tmp_dir, 'allowed')
        os.makedirs(self.allowed_dir)
        self.inside = os.path.join(self.allowed_dir, 'img.png')
        with open(self.inside, 'wb') as f:
            f.write(b'inside')
        self.outside = os.path.join(self.tmp_dir, 'secret.png')
        with open(self.outside, 'wb') as f:
            f.write(b'outside')

    def tearDown(self):
        os.environ.pop(ENV_KEY, None)

    def test_single_source_string_passes_through_unchanged(self):
        """A single video source is handled by the patched reader backend, so it must be left untouched."""
        from swift.template.vision_utils import _safe_video_input
        self.assertEqual(_safe_video_input('http://example.com/v.mp4'), 'http://example.com/v.mp4')
        self.assertEqual(_safe_video_input('/some/where/v.mp4'), '/some/where/v.mp4')
        self.assertIsNone(_safe_video_input(None))

    def test_frame_list_base64_and_pil_pass_through(self):
        from PIL import Image

        from swift.template.vision_utils import _safe_video_input
        b64 = 'data:image/png;base64,aGVsbG8='
        self.assertEqual(_safe_video_input([b64, b64]), [b64, b64])
        img = Image.new('RGB', (2, 2))
        self.assertIs(_safe_video_input([img])[0], img)

    def test_frame_list_url_is_routed_through_the_guarded_loader(self):
        from unittest.mock import patch

        from swift.template import vision_utils
        with patch.object(vision_utils, 'load_image', return_value='LOADED') as m:
            out = vision_utils._safe_video_input(['http://127.0.0.1:9/a.png', 'HTTP://127.0.0.1:9/b.png'])
        self.assertEqual(out, ['LOADED', 'LOADED'])
        self.assertEqual(m.call_count, 2)

    def test_frame_list_local_path_honours_allowlist(self):
        from swift.template.vision_utils import _safe_video_input
        os.environ[ENV_KEY] = self.allowed_dir
        self.assertEqual(_safe_video_input([self.inside]), [self.inside])
        with self.assertRaises(ValueError):
            _safe_video_input([self.outside])


if __name__ == '__main__':
    unittest.main()
