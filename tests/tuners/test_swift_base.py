import copy
import os
import shutil
import tempfile
import unittest
from time import time

import torch
from modelscope.models.nlp.structbert import (SbertConfig,
                                              SbertForSequenceClassification)
from peft.utils import WEIGHTS_NAME

from swift import AdapterConfig, LoRAConfig, Swift, SwiftModel, push_to_hub


class TestSwift(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def test_swift_lora_injection(self):
        model = SbertForSequenceClassification(SbertConfig())
        model2 = copy.deepcopy(model)
        lora_config = LoRAConfig(target_modules=['query', 'key', 'value'])
        model = Swift.prepare_model(model, config=lora_config)
        self.assertTrue(isinstance(model, SwiftModel))
        model.save_pretrained(self.tmp_dir)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, 'default')))
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir, 'default', WEIGHTS_NAME)))

        model2 = Swift.from_pretrained(model2, self.tmp_dir)

        state_dict = model.state_dict()
        state_dict2 = model2.state_dict()
        for key in state_dict:
            self.assertTrue(key in state_dict2)
            self.assertTrue(
                all(
                    torch.isclose(state_dict[key],
                                  state_dict2[key]).flatten().detach().cpu()))

    def test_swift_multiple_adapters(self):
        model = SbertForSequenceClassification(SbertConfig())
        model2 = copy.deepcopy(model)
        lora_config = LoRAConfig(target_modules=['query', 'key', 'value'])
        adapter_config = AdapterConfig(
            dim=model.config.hidden_size,
            target_modules=r'.*layer\.\d+$',
            method_name='feed_forward_chunk',
            hidden_pos=0)
        model = Swift.prepare_model(
            model, config={
                'lora': lora_config,
                'adapter': adapter_config
            })
        self.assertTrue(isinstance(model, SwiftModel))
        model.save_pretrained(self.tmp_dir, adapter_name=['lora', 'adapter'])
        with open(os.path.join(self.tmp_dir, 'configuration.json'), 'w') as f:
            f.write('{}')
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, 'lora')))
        self.assertTrue(
            os.path.exists(os.path.join(self.tmp_dir, 'lora', WEIGHTS_NAME)))
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, 'adapter')))
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir, 'adapter', WEIGHTS_NAME)))

        revision = str(int(time()))
        push_to_hub(
            'damo/test_swift_multiple_model',
            output_dir=self.tmp_dir,
            tag=revision)
        model2 = Swift.from_pretrained(
            model2,
            'damo/test_swift_multiple_model',
            adapter_name=['lora', 'adapter'],
            revision=revision)
        state_dict = model.state_dict()
        state_dict2 = model2.state_dict()
        for key in state_dict:
            self.assertTrue(key in state_dict2)
            self.assertTrue(
                all(
                    torch.isclose(state_dict[key],
                                  state_dict2[key]).flatten().detach().cpu()))
