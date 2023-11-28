import unittest

import torch
from modelscope import Model, Preprocessor

from swift import (LoRAConfig, Swift)


class TestMergedLinear(unittest.TestCase):

    def test_swift_lora_forward(self):
        model = Model.from_pretrained(
            'damo/nlp_structbert_sentence-similarity_chinese-base')
        preprocessor = Preprocessor.from_pretrained(
            'damo/nlp_structbert_sentence-similarity_chinese-base')
        inputs = preprocessor('how are you')
        lora_config = LoRAConfig(target_modules=['query', 'key', 'value'], use_merged_linear=True, enable_lora=[True, True, True])
        outputs = model(**inputs)
        model = Swift.prepare_model(model, config=lora_config)
        outputs_lora = model(**inputs)
        model.deactivate_adapter('default')
        outputs_deactivate = model(**inputs)
        model.activate_adapter('default')
        outputs_reactivate = model(**inputs)
        self.assertTrue(
            torch.allclose(outputs.logits, outputs_deactivate.logits))
        self.assertTrue(
            not torch.allclose(outputs.logits, outputs_lora.logits))
        self.assertTrue(
            torch.allclose(outputs_lora.logits, outputs_reactivate.logits))