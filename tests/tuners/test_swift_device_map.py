import os
import shutil
import tempfile
import unittest

import torch
from modelscope import Model
from peft.utils import WEIGHTS_NAME

from swift import LoRAConfig, SwiftModel


@unittest.skip
class TestSwift(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def test_swift_multiple_adapters(self):
        model = Model.from_pretrained('modelscope/Llama-2-7b-ms', device_map='auto')
        lora_config = LoRAConfig(target_modules=['q_proj', 'k_proj', 'v_proj'])
        model: SwiftModel = SwiftModel(model, config={'lora': lora_config})
        self.assertTrue(isinstance(model, SwiftModel))
        model.save_pretrained(self.tmp_dir, adapter_name=['lora'])
        state_dict = model.state_dict()
        with open(os.path.join(self.tmp_dir, 'configuration.json'), 'w') as f:
            f.write('{}')
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, 'lora')))
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, 'lora', WEIGHTS_NAME)))
        model = Model.from_pretrained('modelscope/Llama-2-7b-ms', device_map='auto')
        model = SwiftModel.from_pretrained(model, self.tmp_dir, adapter_name=['lora'], device_map='auto')

        state_dict2 = model.state_dict()
        for key in state_dict:
            self.assertTrue(key in state_dict2)
            self.assertTrue(all(torch.isclose(state_dict[key], state_dict2[key]).flatten().detach().cpu()))

        self.assertTrue(len(set(model.hf_device_map.values())) == torch.cuda.device_count())
