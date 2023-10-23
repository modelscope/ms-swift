import os
import shutil
import tempfile
import unittest

from modelscope import Model, AutoTokenizer

from swift import Swift
from swift.tuners.rome import RomeConfig


class TestRome(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def test_rome(self):
        model = Model.from_pretrained('modelscope/Llama-2-7b-ms', device_map='auto', trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained('modelscope/Llama-2-7b-ms', trust_remote_code=True)
        request = [
            {
                "prompt": "{} was the founder of",
                "subject": "Steve Jobs",
                "target": "Microsoft",
            }
        ]
        config = RomeConfig(
            model_type='llama-7b',
            knowledge=request,
            tokenizer=tokenizer,
        )

        model = Swift.prepare_model(model, config)
        model.generate()
