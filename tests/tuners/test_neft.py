import os
import shutil
import tempfile
import unittest

import torch
from modelscope import AutoModel, Preprocessor
from peft.utils import WEIGHTS_NAME
from transformers import PreTrainedModel

from swift import LoRAConfig, Swift
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
        model = AutoModel.from_pretrained('AI-ModelScope/bert-base-uncased')
        preprocessor = Preprocessor.from_pretrained('damo/nlp_structbert_sentence-similarity_chinese-base')
        inputs = preprocessor('how are you')
        config = NEFTuneConfig()

        t1 = model.embeddings.word_embeddings(inputs['input_ids'])
        model = Swift.prepare_model(model, config)
        model.train()
        t2 = model.embeddings.word_embeddings(inputs['input_ids'])
        model.deactivate_adapter('default')
        t3 = model.embeddings.word_embeddings(inputs['input_ids'])
        self.assertTrue(torch.allclose(t1, t3))
        self.assertFalse(torch.allclose(t1, t2))
        model.save_pretrained(self.tmp_dir)
        bin_file = os.path.join(self.tmp_dir, 'pytorch_model.bin')
        self.assertTrue(os.path.isfile(bin_file))
        model2 = AutoModel.from_pretrained(self.tmp_dir)

        state_dict = model.state_dict()
        state_dict2 = model2.state_dict()
        self.assertTrue(len(state_dict) > 0)
        for key in state_dict:
            self.assertTrue(key in state_dict2)
            self.assertTrue(all(torch.isclose(state_dict[key], state_dict2[key]).flatten().detach().cpu()))

        shutil.rmtree(self.tmp_dir)
        PreTrainedModel.origin_save_pretrained = PreTrainedModel.save_pretrained
        delattr(PreTrainedModel, 'save_pretrained')
        model.save_pretrained(self.tmp_dir)
        bin_file = os.path.join(self.tmp_dir, WEIGHTS_NAME)
        self.assertTrue(os.path.isfile(bin_file))
        model_new = AutoModel.from_pretrained('AI-ModelScope/bert-base-uncased')
        model_new_2 = Swift.from_pretrained(model_new, self.tmp_dir)

        state_dict = model.state_dict()
        state_dict2 = model_new_2.state_dict()
        self.assertTrue(len(state_dict) > 0)
        for key in state_dict:
            self.assertTrue(key in state_dict2)
            self.assertTrue(all(torch.isclose(state_dict[key], state_dict2[key]).flatten().detach().cpu()))
        PreTrainedModel.save_pretrained = PreTrainedModel.origin_save_pretrained

    def test_neft_lora(self):
        model = AutoModel.from_pretrained('AI-ModelScope/bert-base-uncased')
        preprocessor = Preprocessor.from_pretrained('damo/nlp_structbert_sentence-similarity_chinese-base')
        inputs = preprocessor('how are you')
        config = NEFTuneConfig()
        config2 = LoRAConfig(target_modules=['query', 'key', 'value'])

        t1 = model.embeddings.word_embeddings(inputs['input_ids'])
        model = Swift.prepare_model(model, {'c1': config, 'c2': config2})
        model.train()
        t2 = model.embeddings.word_embeddings(inputs['input_ids'])
        model.deactivate_adapter('c1')
        t3 = model.embeddings.word_embeddings(inputs['input_ids'])
        self.assertTrue(torch.allclose(t1, t3))
        self.assertFalse(torch.allclose(t1, t2))
        model.save_pretrained(self.tmp_dir)
        bin_file = os.path.join(self.tmp_dir, 'c2', WEIGHTS_NAME)
        self.assertTrue(os.path.isfile(bin_file))
        bin_file = os.path.join(self.tmp_dir, 'c1', WEIGHTS_NAME)
        self.assertTrue(not os.path.isfile(bin_file))
        model_new = AutoModel.from_pretrained('AI-ModelScope/bert-base-uncased')
        t1 = model_new.embeddings.word_embeddings(inputs['input_ids'])
        model_new = Swift.from_pretrained(model_new, self.tmp_dir)
        model_new.train()
        t2 = model_new.embeddings.word_embeddings(inputs['input_ids'])
        model_new.eval()
        t4 = model_new.embeddings.word_embeddings(inputs['input_ids'])
        model_new.train()
        model_new.deactivate_adapter('c1')
        t3 = model_new.embeddings.word_embeddings(inputs['input_ids'])
        self.assertTrue(torch.allclose(t1, t3))
        self.assertTrue(torch.allclose(t1, t4))
        self.assertFalse(torch.allclose(t1, t2))

        state_dict = model.state_dict()
        state_dict2 = model_new.state_dict()
        self.assertTrue(len(state_dict) > 0 and all(['lora' in key for key in state_dict.keys()]))
        for key in state_dict:
            self.assertTrue(key in state_dict2)
            self.assertTrue(all(torch.isclose(state_dict[key], state_dict2[key]).flatten().detach().cpu()))
