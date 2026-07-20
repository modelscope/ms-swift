import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from swift.utils.import_utils import patch_trl_package_check


class TestImportUtils(unittest.TestCase):

    @staticmethod
    def make_trl_modules(package_check):
        trl = ModuleType('trl')
        trl.__path__ = []
        trl_import_utils = ModuleType('trl.import_utils')
        trl_import_utils._is_package_available = package_check

        def is_vllm_ascend_available():
            return trl_import_utils._is_package_available('vllm_ascend')

        def is_weave_available():
            return trl_import_utils._is_package_available('weave')

        trl_import_utils.is_vllm_ascend_available = is_vllm_ascend_available
        trl_import_utils.is_weave_available = is_weave_available
        trl.import_utils = trl_import_utils
        return {'trl': trl, 'trl.import_utils': trl_import_utils}

    def test_patch_trl_package_check(self):

        def package_check(package, return_version=False):
            if return_version:
                return True, '1.0.0'
            return package not in {'vllm_ascend', 'weave'}, None

        modules = self.make_trl_modules(package_check)
        trl_import_utils = modules['trl.import_utils']
        with patch.dict(sys.modules, modules):
            patch_trl_package_check()

            self.assertIs(trl_import_utils.is_vllm_ascend_available(), False)
            self.assertIs(trl_import_utils.is_weave_available(), False)
            self.assertEqual(trl_import_utils._is_package_available('vllm', return_version=True), (True, '1.0.0'))

    def test_patch_trl_package_check_is_noop_for_bool_return(self):

        def package_check(package, return_version=False):
            if return_version:
                return True, '1.0.0'
            return True

        modules = self.make_trl_modules(package_check)
        trl_import_utils = modules['trl.import_utils']
        with patch.dict(sys.modules, modules):
            patch_trl_package_check()

            self.assertIs(trl_import_utils._is_package_available, package_check)

    def test_patch_trl_package_check_is_noop_without_trl(self):
        with patch.dict(sys.modules, {'trl': None, 'trl.import_utils': None}):
            patch_trl_package_check()

    def test_patch_trl_package_check_is_noop_without_package_check(self):
        modules = self.make_trl_modules(lambda package: True)
        del modules['trl.import_utils']._is_package_available
        with patch.dict(sys.modules, modules):
            patch_trl_package_check()


if __name__ == '__main__':
    unittest.main()
