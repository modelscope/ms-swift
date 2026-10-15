import os
import unittest

from swift.utils import get_dist_setting, get_node_setting

_KEYS = ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'LOCAL_SIZE', '_PATCH_WORLD_SIZE', 'NODE_RANK',
         'NNODES')


class TestDistEnvValues(unittest.TestCase):

    def setUp(self):
        self._saved = {key: os.environ.get(key) for key in _KEYS}
        for key in _KEYS:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_unset_falls_back_to_defaults(self):
        self.assertEqual(get_dist_setting(), (-1, -1, 1, 1))
        self.assertEqual(get_node_setting(), (0, 1))

    def test_present_values_keep_their_meaning(self):
        os.environ['RANK'] = '2'
        os.environ['NNODES'] = '4'
        self.assertEqual(get_dist_setting()[0], 2)
        self.assertEqual(get_node_setting(), (0, 4))

    def test_blank_rank_falls_back_to_unset(self):
        # `RANK=` is what a launcher script or a Dockerfile leaves behind when the variable is
        # exported before the value is known; get_dist_setting() must not raise from it.
        os.environ['RANK'] = ''
        self.assertEqual(get_dist_setting(), (-1, -1, 1, 1))

    def test_blank_node_env_falls_back_to_defaults(self):
        os.environ['NODE_RANK'] = ''
        os.environ['NNODES'] = ''
        self.assertEqual(get_node_setting(), (0, 1))

    def test_blank_world_size_already_falls_back(self):
        # the control: WORLD_SIZE and LOCAL_WORLD_SIZE read through `or`, which already tolerates
        # a blank value. The plain int() reads must behave the same way.
        os.environ['WORLD_SIZE'] = ''
        os.environ['LOCAL_WORLD_SIZE'] = ''
        self.assertEqual(get_dist_setting(), (-1, -1, 1, 1))


if __name__ == '__main__':
    unittest.main()
