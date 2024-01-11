import os
import shutil
import tempfile
import unittest

import torch
from modelscope import AutoTokenizer, Model, Preprocessor

from swift import Swift
from swift.tuners import NEFTuneConfig


class TestNEFT(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def test_neft(self):
        model = Model.from_pretrained(
            'damo/nlp_structbert_sentence-similarity_chinese-base')
        preprocessor = Preprocessor.from_pretrained(
            'damo/nlp_structbert_sentence-similarity_chinese-base')
        inputs = preprocessor('how are you')
        config = NEFTuneConfig()

        t1 = model.encoder.embeddings.word_embeddings(inputs['input_ids'])
        model = Swift.prepare_model(model, config)
        model.train()
        t2 = model.encoder.embeddings.word_embeddings(inputs['input_ids'])
        model.deactivate_adapter('default')
        t3 = model.encoder.embeddings.word_embeddings(inputs['input_ids'])
        self.assertTrue(torch.allclose(t1, t3))
        self.assertFalse(torch.allclose(t1, t2))
