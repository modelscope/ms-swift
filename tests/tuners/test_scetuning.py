import copy
import os
import shutil
import tempfile
import unittest

import torch
from modelscope import snapshot_download

from swift import SCETuningConfig, Swift
from swift.tuners.part import PartConfig


class TestSCETuning(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def model_comparison(self, model, model2):
        model_key = list(model.state_dict().keys())
        model2_key = list(model2.state_dict().keys())
        self.assertTrue(model_key == model2_key)
        model_val = torch.sum(torch.stack([torch.sum(val) for val in model.state_dict().values()]))
        model2_val = torch.sum(torch.stack([torch.sum(val) for val in model2.state_dict().values()]))
        self.assertTrue(torch.isclose(model_val, model2_val))

    def test_scetuning_on_diffusers_v1(self):
        model_dir = snapshot_download('AI-ModelScope/stable-diffusion-v1-5')
        from diffusers import UNet2DConditionModel
        model = UNet2DConditionModel.from_pretrained(model_dir, subfolder='unet')
        model.requires_grad_(False)
        model_check = copy.deepcopy(model)
        # module_keys = [key for key, _ in model.named_modules()]
        scetuning_config = SCETuningConfig(
            dims=[320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280],
            tuner_mode='encoder',
            target_modules=[
                'conv_in', 'down_blocks.0.attentions.0', 'down_blocks.0.attentions.1', 'down_blocks.0.downsamplers',
                'down_blocks.1.attentions.0', 'down_blocks.1.attentions.1', 'down_blocks.1.downsamplers',
                'down_blocks.2.attentions.0', 'down_blocks.2.attentions.1', 'down_blocks.2.downsamplers',
                'down_blocks.3.resnets.0', 'down_blocks.3.resnets.1'
            ])
        model = Swift.prepare_model(model, config=scetuning_config)
        print(model.get_trainable_parameters())
        input_data = {
            'sample': torch.ones((1, 4, 64, 64)),
            'timestep': 10,
            'encoder_hidden_states': torch.ones((1, 77, 768))
        }
        result = model(**input_data).sample
        print(result.shape)
        model.save_pretrained(self.tmp_dir)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, 'default')))
        model_check = Swift.from_pretrained(model_check, self.tmp_dir)
        self.model_comparison(model, model_check)

    def test_scetuning_part_mixin(self):
        model_dir = snapshot_download('AI-ModelScope/stable-diffusion-v1-5')
        from diffusers import UNet2DConditionModel
        model = UNet2DConditionModel.from_pretrained(model_dir, subfolder='unet')
        model.requires_grad_(False)
        model_check = copy.deepcopy(model)
        # module_keys = [key for key, _ in model.named_modules()]
        scetuning_config = SCETuningConfig(
            dims=[320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280],
            tuner_mode='encoder',
            target_modules=[
                'conv_in', 'down_blocks.0.attentions.0', 'down_blocks.0.attentions.1', 'down_blocks.0.downsamplers',
                'down_blocks.1.attentions.0', 'down_blocks.1.attentions.1', 'down_blocks.1.downsamplers',
                'down_blocks.2.attentions.0', 'down_blocks.2.attentions.1', 'down_blocks.2.downsamplers',
                'down_blocks.3.resnets.0', 'down_blocks.3.resnets.1'
            ])
        targets = r'.*(to_k|to_v).*'
        part_config = PartConfig(target_modules=targets)
        model = Swift.prepare_model(model, config=scetuning_config)
        model = Swift.prepare_model(model, config={'part': part_config})
        print(model.get_trainable_parameters())
        input_data = {
            'sample': torch.ones((1, 4, 64, 64)),
            'timestep': 10,
            'encoder_hidden_states': torch.ones((1, 77, 768))
        }
        model.set_active_adapters('default')
        model.set_active_adapters('part')
        model.set_active_adapters('default')
        result = model(**input_data).sample
        print(result.shape)
        model.save_pretrained(self.tmp_dir)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, 'default')))
        model_check = Swift.from_pretrained(model_check, self.tmp_dir)
        self.model_comparison(model, model_check)

    def test_scetuning_on_diffusers_v2(self):
        model_dir = snapshot_download('AI-ModelScope/stable-diffusion-v1-5')
        from diffusers import UNet2DConditionModel
        model = UNet2DConditionModel.from_pretrained(model_dir, subfolder='unet')
        model.requires_grad_(False)
        model_check = copy.deepcopy(model)
        # module_keys = [key for key, _ in model.named_modules()]
        scetuning_config = SCETuningConfig(
            dims=[1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320, 320],
            tuner_mode='decoder',
            target_modules=[
                'up_blocks.0.resnets.0', 'up_blocks.0.resnets.1', 'up_blocks.0.resnets.2', 'up_blocks.1.resnets.0',
                'up_blocks.1.resnets.1', 'up_blocks.1.resnets.2', 'up_blocks.2.resnets.0', 'up_blocks.2.resnets.1',
                'up_blocks.2.resnets.2', 'up_blocks.3.resnets.0', 'up_blocks.3.resnets.1', 'up_blocks.3.resnets.2'
            ])
        model = Swift.prepare_model(model, config=scetuning_config)
        print(model.get_trainable_parameters())
        input_data = {
            'sample': torch.ones((1, 4, 64, 64)),
            'timestep': 10,
            'encoder_hidden_states': torch.ones((1, 77, 768))
        }
        result = model(**input_data).sample
        print(result.shape)
        model.save_pretrained(self.tmp_dir)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, 'default')))
        model_check = Swift.from_pretrained(model_check, self.tmp_dir)
        self.model_comparison(model, model_check)


if __name__ == '__main__':
    unittest.main()
