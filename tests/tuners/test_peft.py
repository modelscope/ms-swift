import copy
import os
import shutil
import tempfile
import unittest

import peft
import torch
from modelscope import Preprocessor
from modelscope.models.nlp.structbert import SbertConfig, SbertForSequenceClassification
from peft import PeftModel, inject_adapter_in_model
from peft.config import PeftConfigMixin
from peft.tuners.lora import Linear
from peft.utils import WEIGHTS_NAME
from torch import nn

from swift import AdaLoraConfig, LoraConfig, LoRAConfig, Swift, get_peft_model


class TestPeft(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def test_peft_lora_injection(self):
        model = SbertForSequenceClassification(SbertConfig())
        model2 = copy.deepcopy(model)
        lora_config = LoraConfig(target_modules=['query', 'key', 'value'])
        model = Swift.prepare_model(model, lora_config)
        model.save_pretrained(self.tmp_dir, safe_serialization=False)
        with open(os.path.join(self.tmp_dir, 'configuration.json'), 'w') as f:
            f.write('{}')
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, WEIGHTS_NAME)))
        model2 = Swift.from_pretrained(model2, self.tmp_dir)
        state_dict = model.state_dict()
        state_dict2 = model2.state_dict()
        for key in state_dict:
            self.assertTrue(key in state_dict2)
            self.assertTrue(all(torch.isclose(state_dict[key], state_dict2[key]).flatten().detach().cpu()))

    @unittest.skip
    def test_lora_merge(self):

        def reset_lora_parameters(self, adapter_name, init_lora_weights):
            if init_lora_weights is False:
                return

            if adapter_name == 'default':
                ratio = 1.0
            elif adapter_name == 'second':
                ratio = 2.0
            else:
                ratio = 3.0

            if adapter_name in self.lora_A.keys():
                nn.init.ones_(self.lora_A[adapter_name].weight)
                self.lora_A[adapter_name].weight.data = self.lora_A[adapter_name].weight.data * ratio
                nn.init.ones_(self.lora_B[adapter_name].weight)

        Linear.reset_lora_parameters = reset_lora_parameters

        model = SbertForSequenceClassification(SbertConfig())
        lora_config = LoRAConfig(target_modules=['query', 'key', 'value'])
        model = Swift.prepare_model(model, lora_config)
        lora_config2 = LoRAConfig(target_modules=['query', 'key', 'value'])
        model = Swift.prepare_model(model, {'second': lora_config2})
        model.add_weighted_adapter(['default', 'second'],
                                   weights=[0.7, 0.3],
                                   adapter_name='test',
                                   combination_type='cat')
        self.assertTrue(model.base_model.bert.encoder.layer[0].attention.self.key.active_adapter == ['test'])

        model2 = SbertForSequenceClassification(SbertConfig())
        lora_config = LoraConfig(target_modules=['query', 'key', 'value'])
        model2 = get_peft_model(model2, lora_config)
        lora_config2 = LoraConfig(target_modules=['query', 'key', 'value'])
        inject_adapter_in_model(lora_config2, model2, adapter_name='second')
        model2.add_weighted_adapter(['default', 'second'],
                                    weights=[0.7, 0.3],
                                    adapter_name='test',
                                    combination_type='cat')
        state_dict = model.state_dict()
        state_dict2 = model2.state_dict()
        state_dict2 = {key[len('base_model.model.'):]: value for key, value in state_dict2.items() if 'lora' in key}
        for key in state_dict:
            self.assertTrue(key in state_dict2)
            self.assertTrue(all(torch.isclose(state_dict[key], state_dict2[key]).flatten().detach().cpu()))

        preprocessor = Preprocessor.from_pretrained('damo/nlp_structbert_sentence-similarity_chinese-base')
        inputs = preprocessor('how are you')
        print(model(**inputs))
        model.save_pretrained(self.tmp_dir)
        model3 = SbertForSequenceClassification(SbertConfig())
        model3 = Swift.from_pretrained(model3, self.tmp_dir)
        state_dict3 = model3.state_dict()
        for key in state_dict:
            self.assertTrue(key in state_dict3)
            self.assertTrue(all(torch.isclose(state_dict[key], state_dict3[key]).flatten().detach().cpu()))

    def test_lora_reload_by_peft(self):
        lora_config = LoRAConfig(target_modules=['query', 'key', 'value'])
        model = SbertForSequenceClassification(SbertConfig())
        model2 = copy.deepcopy(model)
        model = Swift.prepare_model(model, lora_config)
        model.save_pretrained(self.tmp_dir, peft_format=True)
        model2 = PeftModel.from_pretrained(model2, self.tmp_dir)
        state_dict = model.state_dict()
        state_dict2 = model2.state_dict()
        state_dict2 = {key[len('base_model.model.'):]: value for key, value in state_dict2.items() if 'lora' in key}
        for key in state_dict:
            self.assertTrue(key in state_dict2)
            self.assertTrue(all(torch.isclose(state_dict[key], state_dict2[key]).flatten().detach().cpu()))

    def test_peft_adalora_injection(self):
        model = SbertForSequenceClassification(SbertConfig())
        model2 = copy.deepcopy(model)
        adalora_config = AdaLoraConfig(target_modules=['query', 'key', 'value'])
        model = Swift.prepare_model(model, adalora_config)
        model.save_pretrained(self.tmp_dir, safe_serialization=False)
        with open(os.path.join(self.tmp_dir, 'configuration.json'), 'w') as f:
            f.write('{}')
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, WEIGHTS_NAME)))
        model2 = Swift.from_pretrained(model2, self.tmp_dir)
        state_dict = model.state_dict()
        state_dict2 = model2.state_dict()
        for key in state_dict:
            self.assertTrue(key in state_dict2)
            self.assertTrue(all(torch.isclose(state_dict[key], state_dict2[key]).flatten().detach().cpu()))

    @unittest.skip
    def test_peft_lora_dtype(self):
        model = SbertForSequenceClassification(SbertConfig())
        model2 = copy.deepcopy(model)
        model3 = copy.deepcopy(model)
        lora_config = LoraConfig(target_modules=['query', 'key', 'value'], lora_dtype='float16')
        model = Swift.prepare_model(model, lora_config)
        model.save_pretrained(self.tmp_dir, safe_serialization=False)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, 'additional_config.json')))
        model2 = Swift.from_pretrained(model2, self.tmp_dir)
        self.assertTrue(model2.base_model.model.bert.encoder.layer[0].attention.self.key.lora_A.default.weight.dtype ==
                        torch.float16)
        self.assertTrue(model2.peft_config['default'].lora_dtype == 'float16')
        state_dict = model.state_dict()
        state_dict2 = model2.state_dict()
        for key in state_dict:
            self.assertTrue(key in state_dict2)
            self.assertTrue(all(torch.isclose(state_dict[key], state_dict2[key]).flatten().detach().cpu()))

        PeftConfigMixin.from_pretrained = PeftConfigMixin.from_pretrained_origin
        model3 = Swift.from_pretrained(model3, self.tmp_dir)
        self.assertTrue(model3.base_model.model.bert.encoder.layer[0].attention.self.key.lora_A.default.weight.dtype ==
                        torch.float32)
        self.assertTrue(isinstance(model3.peft_config['default'], peft.LoraConfig))
