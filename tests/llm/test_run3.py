if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import os
import shutil
import json
import tempfile
import unittest

import torch
from swift.llm import get_model_tokenizer
from swift.llm import EncodePreprocessor, load_dataset, MODEL_MAPPING, get_template


class TestRun3(unittest.TestCase):

    def setUp(self):
        print(f'Testing {type(self).__name__}.{self._testMethodName}')
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def load_ds(self, ds):
        train_dataset, val_dataset = load_dataset(
                ds + ':all',
                split_dataset_ratio=0.0,
                strict=False,
                num_proc=1,
                model_name=['小黄', 'Xiao Huang'],
                model_author=['魔搭', 'ModelScope'])
        return train_dataset.select(range(min(50, len(train_dataset))))

    def test_model_load(self):
        if os.path.exists('./models.txt'):
            with open('./models.txt', 'r') as f:
                models = json.load(f)
        else:
            models = []
        for model_name, model_meta in MODEL_MAPPING.items():
            model_type = model_meta.model_type
            template = model_meta.template
            for group in model_meta.model_groups:
                model = group.models[0]
                if 'skip_test' in (group.tags or []) or model.ms_model_id in models:
                    break
                model_ins = None
                requires = group.requires
                for req in (requires or []):
                    os.system(f'pip install "{req}"')
                if not any(['transformers' in req for req in (requires or [])]):
                    os.system(f'pip install transformers==4.45.1')
                if not any(['accelerate' in req for req in (requires or [])]):
                    os.system(f'pip install accelerate==1.1.1')
                try:
                    print(f'Test model: {model.ms_model_id}')
                    model_ins, tokenizer = get_model_tokenizer(model.ms_model_id)
                except Exception as e:
                    import traceback
                    print(traceback.format_exc())
                    passed = False
                else:
                    passed = True
                    models.append(model.ms_model_id)
                finally:
                    if model_ins is not None:
                        del model_ins
                    if passed:
                        with open('./models.txt', 'w') as f:
                            json.dump(models, f)
                    

    # def test_template_load(self):
    #     self.llm_ds = self.load_ds('AI-ModelScope/sharegpt_gpt4')
    #     self.img_ds = self.load_ds('swift/OK-VQA_train')
    #     self.audio_ds = self.load_ds('speech_asr/speech_asr_aishell1_trainsets')
    #     for model_name, model_meta in MODEL_MAPPING.items():
    #         model_type = model_meta.model_type
    #         template = model_meta.template
    #         requires = model_meta.requires
    #         for req in (requires or []):
    #             os.system(f'pip install {req}')
    #         for group in model_meta.model_groups:
    #             model = group.models[0]
    #             model_ins = None
    #             try:
    #                 model_ins, tokenizer = get_model_tokenizer(model.ms_model_id, load_model=False)
    #                 template = get_template(template, tokenizer)
    #                 if 'audio' in template.__class__.__name__.lower():
    #                     EncodePreprocessor(template)(self.audio_ds)
    #                 elif 'vl' in template.__class__.__name__.lower():
    #                     EncodePreprocessor(template)(self.img_ds)
    #                 else:
    #                     EncodePreprocessor(template)(self.llm_ds)
                    
    #             except Exception as e:
    #                 import traceback
    #                 print(traceback.format_exc())


if __name__ == '__main__':
    unittest.main()
