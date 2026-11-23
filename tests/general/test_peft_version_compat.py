# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest
from unittest.mock import patch

from swift.utils.import_utils import check_peft_version, get_peft_version_warning


class TestPeftVersionCompat(unittest.TestCase):

    def _warning(self, transformers_version, peft_version):
        with patch('transformers.__version__', transformers_version), patch('peft.__version__', peft_version):
            return get_peft_version_warning()

    def test_warns_about_old_peft_on_transformers_524(self):
        message = self._warning('5.24.0', '0.21.3')
        self.assertIn('requires peft>=0.22.0', message)
        self.assertIn('peft 0.21.3', message)

    def test_no_warning_with_required_peft(self):
        self.assertIsNone(self._warning('5.24.0', '0.22.0'))

    def test_old_transformers_keeps_old_peft_supported(self):
        self.assertIsNone(self._warning('5.23.0', '0.19.0'))

    def test_check_only_warns(self):
        with patch('transformers.__version__', '5.24.0'), patch('peft.__version__', '0.21.3'), \
                patch('swift.utils.logger.get_logger') as get_logger:
            check_peft_version()
        get_logger.return_value.warning_once.assert_called_once()


if __name__ == '__main__':
    unittest.main()
