import os.path
import shutil
import tempfile
import unittest

import torch
from modelscope import Model
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

from swift import LoRAConfig, Swift
from swift.tuners.utils import ModulesToSaveWrapper


class TestExtraStateDict(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def test_swift_extra_state_dict(self):
        model = Model.from_pretrained('damo/nlp_structbert_sentence-similarity_chinese-base')
        lora_config = LoRAConfig(target_modules=['query', 'key', 'value'])
        model = Swift.prepare_model(model, lora_config, extra_state_keys=['classifier.*'])
        model.save_pretrained(self.tmp_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp_dir, 'extra_states', 'adapter_model.safetensors')))
        state_dict = safe_load_file(os.path.join(self.tmp_dir, 'extra_states', 'adapter_model.safetensors'))
        self.assertTrue(any('classifier' in key for key in state_dict))
        state_dict['classifier.weight'] = torch.ones_like(state_dict['classifier.weight']) * 2.0
        safe_save_file(state_dict, os.path.join(self.tmp_dir, 'extra_states', 'adapter_model.safetensors'))
        model = Model.from_pretrained('damo/nlp_structbert_sentence-similarity_chinese-base')
        model = Swift.from_pretrained(model, self.tmp_dir, inference_mode=False)
        names = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertTrue(any('classifier' in name for name in names))
        self.assertTrue(torch.allclose(state_dict['classifier.weight'], model.base_model.classifier.weight))

    def test_swift_modules_to_save(self):
        model = Model.from_pretrained('damo/nlp_structbert_sentence-similarity_chinese-base')
        lora_config = LoRAConfig(target_modules=['query', 'key', 'value'], modules_to_save=['classifier'])
        lora_config2 = LoRAConfig(target_modules=['query', 'key', 'value'], modules_to_save=['classifier'])
        model = Swift.prepare_model(model, {'lora1': lora_config, 'lora2': lora_config2})
        model.set_active_adapters('lora1')
        model.set_active_adapters('lora2')
        self.assertTrue(isinstance(model.classifier, ModulesToSaveWrapper))
        self.assertTrue(model.classifier.active_adapter == 'lora2')
        model.save_pretrained(self.tmp_dir)
        state_dict = safe_load_file(os.path.join(self.tmp_dir, 'lora2', 'adapter_model.safetensors'))
        self.assertTrue(any('classifier' in key for key in state_dict))
        state_dict['classifier.weight'] = torch.ones_like(state_dict['classifier.weight']) * 2.0
        safe_save_file(state_dict, os.path.join(self.tmp_dir, 'lora2', 'adapter_model.safetensors'))
        model = Model.from_pretrained('damo/nlp_structbert_sentence-similarity_chinese-base')
        model = Swift.from_pretrained(model, self.tmp_dir, adapter_name='lora2')
        names = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertTrue(any('classifier' in name for name in names))
        self.assertTrue(
            torch.allclose(state_dict['classifier.weight'],
                           model.base_model.classifier.modules_to_save['lora2'].weight))
