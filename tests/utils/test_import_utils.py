import trl.import_utils as trl_import_utils
import unittest
from unittest.mock import patch

from swift.utils.import_utils import patch_trl_package_check


class TestImportUtils(unittest.TestCase):

    def test_patch_trl_package_check(self):

        def package_check(package, return_version=False):
            if return_version:
                return True, '1.0.0'
            return package not in {'vllm_ascend', 'weave'}, None

        with patch.object(trl_import_utils, '_is_package_available', package_check):
            patch_trl_package_check()

            self.assertIs(trl_import_utils.is_vllm_ascend_available(), False)
            self.assertIs(trl_import_utils.is_weave_available(), False)
            self.assertEqual(trl_import_utils._is_package_available('vllm', return_version=True), (True, '1.0.0'))

    def test_patch_trl_package_check_is_noop_for_bool_return(self):

        def package_check(package, return_version=False):
            if return_version:
                return True, '1.0.0'
            return True

        with patch.object(trl_import_utils, '_is_package_available', package_check):
            patch_trl_package_check()

            self.assertIs(trl_import_utils._is_package_available, package_check)


if __name__ == '__main__':
    unittest.main()
