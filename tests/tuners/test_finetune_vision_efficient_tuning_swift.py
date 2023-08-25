# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import os
import shutil
import tempfile
import unittest

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer

from swift import AdapterConfig, LoRAConfig, PromptConfig


class TestVisionEfficientTuningSwiftTrainer(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

        self.train_dataset = MsDataset.load(
            'foundation_model_evaluation_benchmark',
            namespace='damo',
            subset_name='OxfordFlowers',
            split='train')

        self.eval_dataset = MsDataset.load(
            'foundation_model_evaluation_benchmark',
            namespace='damo',
            subset_name='OxfordFlowers',
            split='eval')

        self.max_epochs = 1
        self.num_classes = 102
        self.tune_length = 10

        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def test_vision_efficient_tuning_swift_lora_train(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-lora'

        def cfg_modify_fn(cfg):
            cfg.model.head.num_classes = self.num_classes
            cfg.model.finetune = True
            cfg.train.max_epochs = self.max_epochs
            cfg.train.lr_scheduler.T_max = self.max_epochs
            cfg.model.backbone.lora_length = 0
            return cfg

        lora_config = LoRAConfig(
            r=self.tune_length,
            target_modules=['qkv'],
            merge_weights=False,
            use_merged_linear=True,
            enable_lora=[True])

        kwargs = dict(
            model=model_id,
            work_dir=self.tmp_dir,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            cfg_modify_fn=cfg_modify_fn,
            efficient_tuners=[lora_config])

        trainer = build_trainer(
            name=Trainers.vision_efficient_tuning, default_args=kwargs)
        trainer.train()
        result = trainer.evaluate()
        print(f'Vision-efficient-tuning-lora train output: {result}.')

        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    def test_vision_efficient_tuning_swift_adapter_train(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-adapter'

        def cfg_modify_fn(cfg):
            cfg.model.head.num_classes = self.num_classes
            cfg.model.finetune = True
            cfg.train.max_epochs = self.max_epochs
            cfg.train.lr_scheduler.T_max = self.max_epochs
            cfg.model.backbone.adapter_length = 0
            return cfg

        adapter_config = AdapterConfig(
            dim=768,
            hidden_pos=0,
            target_modules=r'.*blocks\.\d+\.mlp$',
            adapter_length=self.tune_length)

        kwargs = dict(
            model=model_id,
            work_dir=self.tmp_dir,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            cfg_modify_fn=cfg_modify_fn,
            efficient_tuners=[adapter_config])

        trainer = build_trainer(
            name=Trainers.vision_efficient_tuning, default_args=kwargs)
        trainer.train()
        result = trainer.evaluate()
        print(f'Vision-efficient-tuning-adapter train output: {result}.')

        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)

    def test_vision_efficient_tuning_swift_prompt_train(self):
        model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-prompt'

        def cfg_modify_fn(cfg):
            cfg.model.head.num_classes = self.num_classes
            cfg.model.finetune = True
            cfg.train.max_epochs = self.max_epochs
            cfg.train.lr_scheduler.T_max = self.max_epochs
            cfg.model.backbone.prompt_length = 0
            return cfg

        prompt_config = PromptConfig(
            dim=768,
            target_modules=r'.*blocks\.\d+$',
            embedding_pos=0,
            prompt_length=self.tune_length,
            attach_front=False)

        kwargs = dict(
            model=model_id,
            work_dir=self.tmp_dir,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            cfg_modify_fn=cfg_modify_fn,
            efficient_tuners=[prompt_config])

        trainer = build_trainer(
            name=Trainers.vision_efficient_tuning, default_args=kwargs)
        trainer.train()
        result = trainer.evaluate()
        print(f'Vision-efficient-tuning-prompt train output: {result}.')

        results_files = os.listdir(self.tmp_dir)
        self.assertIn(f'{trainer.timestamp}.log.json', results_files)
        for i in range(self.max_epochs):
            self.assertIn(f'epoch_{i+1}.pth', results_files)


if __name__ == '__main__':
    unittest.main()
