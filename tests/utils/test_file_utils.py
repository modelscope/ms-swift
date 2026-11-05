import os
import shutil
import sys
import tempfile
import unittest

from swift.utils import copy_files_by_pattern, import_external_file


class TestFileUtils(unittest.TestCase):

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_copy_files(self):
        os.makedirs(os.path.join(self.tmp_dir, 'source'))
        os.makedirs(os.path.join(self.tmp_dir, 'source', 'subfolder'))
        with open(os.path.join(self.tmp_dir, 'source', '1.txt'), 'w') as f:
            f.write('')
        with open(os.path.join(self.tmp_dir, 'source', 'subfolder', '2.txt'), 'w') as f:
            f.write('')
        copy_files_by_pattern(
            os.path.join(self.tmp_dir, 'source'), os.path.join(self.tmp_dir, 'target'), ['*.txt', 'subfolder/*.txt'])
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, 'target', '1.txt')))
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, 'target', 'subfolder', '2.txt')))

    def test_import_external_files_with_same_name(self):
        plugin_dirs = [os.path.join(self.tmp_dir, name) for name in ('first', 'second')]
        plugin_paths = []
        for plugin_dir, value in zip(plugin_dirs, ('first', 'second')):
            os.makedirs(plugin_dir)
            plugin_path = os.path.join(plugin_dir, 'plugin.py')
            with open(plugin_path, 'w') as f:
                f.write(f'VALUE = {value!r}\n')
            plugin_paths.append(plugin_path)

        modules = []
        try:
            modules = [import_external_file(plugin_path) for plugin_path in plugin_paths]
            self.assertEqual([module.VALUE for module in modules], ['first', 'second'])
        finally:
            for module in modules:
                sys.modules.pop(module.__name__, None)
            for plugin_dir in plugin_dirs:
                while plugin_dir in sys.path:
                    sys.path.remove(plugin_dir)


if __name__ == '__main__':
    unittest.main()
