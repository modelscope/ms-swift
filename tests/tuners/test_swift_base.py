import copy
import os
import shutil
import tempfile
import unittest
from time import time

import torch
from modelscope import Model, Preprocessor
from modelscope.models.nlp.structbert import (SbertConfig,
                                              SbertForSequenceClassification)
from peft.utils import WEIGHTS_NAME
from torch import nn
import math
from swift import AdapterConfig, LoRAConfig, Swift, SwiftModel, push_to_hub, SideConfig, PromptConfig, ResTuningConfig


class TestSwift(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def test_swift_lora_forward(self):

        from swift.tuners.lora import Linear
        def reset_parameters(self):
            nn.Linear.reset_parameters(self)
            if hasattr(self, 'lora_A'):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.ones_(self.lora_B)

        Linear.reset_parameters = reset_parameters

        model = Model.from_pretrained(
            'damo/nlp_structbert_sentence-similarity_chinese-base')
        preprocessor = Preprocessor.from_pretrained(
            'damo/nlp_structbert_sentence-similarity_chinese-base')
        inputs = preprocessor('how are you')
        lora_config = LoRAConfig(target_modules=['query', 'key', 'value'])
        outputs = model(**inputs)
        model = Swift.prepare_model(model, config=lora_config)
        outputs_lora = model(**inputs)
        model.deactivate_adapter('default')
        outputs_deactivate = model(**inputs)
        model.activate_adapter('default')
        outputs_reactivate = model(**inputs)
        self.assertTrue(torch.allclose(outputs.logits, outputs_deactivate.logits))
        self.assertTrue(not torch.allclose(outputs.logits, outputs_lora.logits))
        self.assertTrue(torch.allclose(outputs_lora.logits, outputs_reactivate.logits))

    def test_swift_adapter_forward(self):
        model = Model.from_pretrained(
            'damo/nlp_structbert_sentence-similarity_chinese-base')
        preprocessor = Preprocessor.from_pretrained(
            'damo/nlp_structbert_sentence-similarity_chinese-base')
        inputs = preprocessor('how are you')
        adapter_config = AdapterConfig(
            dim=model.config.hidden_size,
            target_modules=r'.*layer\.\d+$',
            method_name='feed_forward_chunk',
            hidden_pos=0)
        outputs = model(**inputs)
        model = Swift.prepare_model(model, config=adapter_config)
        outputs_lora = model(**inputs)
        model.deactivate_adapter('default')
        outputs_deactivate = model(**inputs)
        model.activate_adapter('default')
        outputs_reactivate = model(**inputs)
        self.assertTrue(torch.allclose(outputs.logits, outputs_deactivate.logits))
        self.assertTrue(not torch.allclose(outputs.logits, outputs_lora.logits))
        self.assertTrue(torch.allclose(outputs_lora.logits, outputs_reactivate.logits))

    def test_swift_prompt_forward(self):
        model = Model.from_pretrained(
            'damo/nlp_structbert_sentence-similarity_chinese-base')
        preprocessor = Preprocessor.from_pretrained(
            'damo/nlp_structbert_sentence-similarity_chinese-base')
        inputs = preprocessor('how are you')
        prompt_config = PromptConfig(
            dim=model.config.hidden_size,
            target_modules=r'.*layer\.\d+$',
            embedding_pos=0,
            attention_mask_pos=1)
        outputs = model(**inputs)
        model = Swift.prepare_model(model, config=prompt_config)
        outputs_lora = model(**inputs)
        model.deactivate_adapter('default')
        outputs_deactivate = model(**inputs)
        model.activate_adapter('default')
        outputs_reactivate = model(**inputs)
        self.assertTrue(torch.allclose(outputs.logits, outputs_deactivate.logits))
        self.assertTrue(not torch.allclose(outputs.logits, outputs_lora.logits))
        self.assertTrue(torch.allclose(outputs_lora.logits, outputs_reactivate.logits))

    def test_swift_restuner_forward(self):
        model = Model.from_pretrained(
            'damo/nlp_structbert_sentence-similarity_chinese-base')
        preprocessor = Preprocessor.from_pretrained(
            'damo/nlp_structbert_sentence-similarity_chinese-base')
        inputs = preprocessor('how are you')
        restuner_config = ResTuningConfig(
            dims=model.config.hidden_size,
            root_modules=r'.*layer.0$',
            stem_modules=r'.*layer\.\d+$',
            target_modules=r'.*pooler',
            target_modules_hook='input',
            tuner_cfg="res_adapter",
        )
        outputs = model(**inputs)
        model = Swift.prepare_model(model, config=restuner_config)
        outputs_lora = model(**inputs)
        model.deactivate_adapter('default')
        outputs_deactivate = model(**inputs)
        model.activate_adapter('default')
        outputs_reactivate = model(**inputs)
        self.assertTrue(torch.allclose(outputs.logits, outputs_deactivate.logits))
        self.assertTrue(not torch.allclose(outputs.logits, outputs_lora.logits))
        self.assertTrue(torch.allclose(outputs_lora.logits, outputs_reactivate.logits))

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

    def test_swift_side_bert(self):
        model = Model.from_pretrained(
            'damo/nlp_structbert_sentence-similarity_chinese-base')
        preprocessor = Preprocessor.from_pretrained(
            'damo/nlp_structbert_sentence-similarity_chinese-base')
        inputs = preprocessor('how are you')
        model2 = copy.deepcopy(model)
        result_origin = model(**inputs).logits
        print(
            f'test_swift_side_bert result_origin shape: {result_origin.shape}, result_origin sum: {torch.sum(result_origin)}'
        )

        side_config = SideConfig(
            dim=model.config.hidden_size,
            target_modules=r'.*encoder.encoder',
            side_module_name='mlp',
            hidden_pos='last_hidden_state'
        )

        model = Swift.prepare_model(model, config=side_config)
        result_activate = model(**inputs).logits
        model.deactivate_adapter('default')
        result_deactivate = model(**inputs).logits
        model.activate_adapter('default')
        result_reactivate = model(**inputs).logits
        self.assertTrue(torch.allclose(result_origin, result_deactivate))
        self.assertTrue(not torch.allclose(result_origin, result_activate))
        self.assertTrue(torch.allclose(result_activate, result_reactivate))
        print(
            f'test_swift_side_bert result shape: {result_origin.shape}, result sum: {torch.sum(result_origin)}'
        )

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


if __name__ == '__main__':
    unittest.main()
