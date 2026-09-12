import os
import tempfile
import unittest

from swift.utils import SafeMediaPath

ENV_KEY = 'SWIFT_MEDIA_ALLOWED_DIRS'


class TestSafeMediaPath(unittest.TestCase):

    def setUp(self):
        os.environ.pop(ENV_KEY, None)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self.temp_dir.name
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
        self.temp_dir.cleanup()

    def test_unset_allowlist_keeps_current_behavior(self):
        self.assertEqual(SafeMediaPath.check(self.outside), self.outside)

    def test_path_inside_allowed_dir_is_returned(self):
        os.environ[ENV_KEY] = self.allowed_dir
        self.assertEqual(SafeMediaPath.check(self.inside), self.inside)

    def test_path_outside_allowed_dir_is_refused(self):
        os.environ[ENV_KEY] = self.allowed_dir
        with self.assertRaises(ValueError):
            SafeMediaPath.check(self.outside)

    def test_traversal_and_symlink_out_of_allowed_dir_are_refused(self):
        os.environ[ENV_KEY] = self.allowed_dir
        link = os.path.join(self.allowed_dir, 'link.png')
        os.symlink(self.outside, link)
        for path in (os.path.join(self.allowed_dir, '..', 'secret.png'), link):
            with self.subTest(path=path), self.assertRaises(ValueError):
                SafeMediaPath.check(path)

    def test_multiple_allowed_dirs(self):
        other_dir = os.path.join(self.tmp_dir, 'other')
        os.makedirs(other_dir)
        other = os.path.join(other_dir, 'img.png')
        with open(other, 'wb') as f:
            f.write(b'other')
        os.environ[ENV_KEY] = f'{self.allowed_dir}, {other_dir}'
        self.assertEqual(SafeMediaPath.check(self.inside), self.inside)
        self.assertEqual(SafeMediaPath.check(other), other)
        with self.assertRaises(ValueError):
            SafeMediaPath.check(self.outside)

    def test_sibling_dir_with_shared_prefix_is_refused(self):
        sibling = self.allowed_dir + '-evil'
        os.makedirs(sibling)
        path = os.path.join(sibling, 'img.png')
        with open(path, 'wb') as f:
            f.write(b'sibling')
        os.environ[ENV_KEY] = self.allowed_dir
        with self.assertRaises(ValueError):
            SafeMediaPath.check(path)

    def test_missing_path_outside_allowed_dir_is_refused_alike(self):
        os.environ[ENV_KEY] = self.allowed_dir
        missing = os.path.join(self.tmp_dir, 'no-such-file.png')
        with self.assertRaises(ValueError) as existing_ctx:
            SafeMediaPath.check(self.outside)
        with self.assertRaises(ValueError) as missing_ctx:
            SafeMediaPath.check(missing)
        self.assertEqual(
            str(existing_ctx.exception).replace(self.outside, ''),
            str(missing_ctx.exception).replace(missing, ''),
        )


class TestRequestMediaPath(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self.temp_dir.name
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
        self.temp_dir.cleanup()

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


if __name__ == '__main__':
    unittest.main()
