import copy
import os
import shutil
import tempfile
import unittest

import torch
from modelscope import snapshot_download

from swift import ResTuningConfig, Swift, SwiftModel


class TestSwiftResTuning(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def set_random_seed(self, seed=123):
        """Set random seed manually to get deterministic results"""
        import random
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def model_comparison(self, model, model2):
        model_key = list(model.state_dict().keys())
        model2_key = list(model2.state_dict().keys())
        self.assertTrue(model_key == model2_key)
        model_val = torch.sum(torch.stack([torch.sum(val) for val in model.state_dict().values()]))
        model2_val = torch.sum(torch.stack([torch.sum(val) for val in model2.state_dict().values()]))
        self.assertTrue(torch.isclose(model_val, model2_val))

    def test_swift_restuning_vit(self):
        model_dir = snapshot_download('AI-ModelScope/vit-base-patch16-224')
        from transformers import AutoModelForImageClassification
        model = AutoModelForImageClassification.from_pretrained(model_dir)
        model_swift_1 = copy.deepcopy(model)
        model_swift_2 = copy.deepcopy(model)
        result_origin = model(torch.ones((1, 3, 224, 224))).logits
        print(f'test_swift_restuning_vit result_origin shape: {result_origin.shape}, '
              f'result_origin sum: {torch.sum(result_origin)}')

        # load type - 1
        self.set_random_seed()
        restuning_config_1 = ResTuningConfig(
            dims=768,
            root_modules=r'.*vit.encoder.layer.0$',
            stem_modules=r'.*vit.encoder.layer\.\d+$',
            target_modules=r'.*vit.layernorm',
            target_modules_hook='input',
            tuner_cfg='res_adapter',
        )
        model_swift_1 = Swift.prepare_model(model_swift_1, config=restuning_config_1)
        self.assertTrue(isinstance(model_swift_1, SwiftModel))
        print(model_swift_1.get_trainable_parameters())
        result_swift_1 = model_swift_1(torch.ones((1, 3, 224, 224))).logits
        print(f'test_swift_restuning_vit result_swift_1 shape: {result_swift_1.shape}, '
              f'result_swift_1 sum: {torch.sum(result_swift_1)}')

        # load type - 2
        self.set_random_seed()
        restuning_config_2 = ResTuningConfig(
            dims=768,
            root_modules=r'.*vit.encoder.layer.0$',
            stem_modules=r'.*vit.encoder.layer\.\d+$',
            target_modules=r'.*vit.encoder',
            target_modules_hook='output',
            target_hidden_pos='last_hidden_state',
            tuner_cfg='res_adapter',
        )
        model_swift_2 = Swift.prepare_model(model_swift_2, config=restuning_config_2)
        self.assertTrue(isinstance(model_swift_2, SwiftModel))
        print(model_swift_2.get_trainable_parameters())
        result_swift_2 = model_swift_2(torch.ones((1, 3, 224, 224))).logits
        print(f'test_swift_restuning_vit result_swift_2 shape: {result_swift_2.shape}, '
              f'result_swift_2 sum: {torch.sum(result_swift_2)}')

        self.assertTrue(all(torch.isclose(result_swift_1, result_swift_2).flatten()))

        model_swift_1.save_pretrained(self.tmp_dir)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, 'default')))
        model_loaded = Swift.from_pretrained(model, self.tmp_dir)
        self.model_comparison(model_swift_1, model_loaded)

    def test_swift_restuning_diffusers_sd(self):
        model_dir = snapshot_download('AI-ModelScope/stable-diffusion-v1-5')
        from diffusers import UNet2DConditionModel
        model = UNet2DConditionModel.from_pretrained(model_dir, subfolder='unet')
        model.requires_grad_(False)
        model2 = copy.deepcopy(model)
        self.set_random_seed()
        input_data = {
            'sample': torch.ones((1, 4, 64, 64)),
            'timestep': 10,
            'encoder_hidden_states': torch.ones((1, 77, 768))
        }
        result_origin = model(**input_data).sample
        print(f'test_swift_restuning_diffusers_sd result_origin shape: {result_origin.shape}, '
              f'result_origin sum: {torch.sum(result_origin)}')

        self.set_random_seed()
        restuning_config = ResTuningConfig(
            dims=[1280, 1280, 1280, 640, 320],
            root_modules='mid_block',
            stem_modules=['mid_block', 'up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3'],
            target_modules='conv_norm_out',
            tuner_cfg='res_group_adapter',
            use_upsample=True,
            upsample_out_channels=[1280, 1280, 640, 320, None],
            zero_init_last=True)

        model = Swift.prepare_model(model, config=restuning_config)
        self.assertTrue(isinstance(model, SwiftModel))
        print(model.get_trainable_parameters())

        result = model(**input_data).sample
        print(f'test_swift_restuning_diffusers_sd result shape: {result.shape}, result sum: {torch.sum(result)}')
        model.save_pretrained(self.tmp_dir)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, 'default')))
        model2 = Swift.from_pretrained(model2, self.tmp_dir)
        self.model_comparison(model, model2)


if __name__ == '__main__':
    unittest.main()
