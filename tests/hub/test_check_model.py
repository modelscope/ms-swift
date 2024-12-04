import os
import shutil
import tempfile
import unittest

from modelscope import Model, check_local_model_is_latest


class TestCheckModel(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        import peft
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def test_check_model(self):
        model = Model.from_pretrained('damo/nlp_corom_sentence-embedding_chinese-base', revision='v1.0.0')
        self.assertFalse(check_local_model_is_latest(model.model_dir))
