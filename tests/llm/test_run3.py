import os
import shutil
import json
import tempfile
import unittest
from multiprocessing import Process
from functools import partial
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
                ds,
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
            meta_requires = model_meta.requires or []
            for group in model_meta.model_groups:
                model = group.models[0]
                if 'skip_test' in (group.tags or []) or model.ms_model_id in models:
                    break
                requires = meta_requires + (group.requires or [])
                for req in requires:
                    os.system(f'pip install "{req}"')
                if not any(['transformers' in req for req in requires]):
                    os.system(f'pip install transformers -U')
                if not any(['accelerate' in req for req in requires]):
                    os.system(f'pip install accelerate -U')
                try:
                    cmd = f'PYTHONPATH=. python tests/llm/load_model.py --ms_model_id {model.ms_model_id}'
                    if os.system(cmd) != 0:
                        raise RuntimeError()
                except Exception as e:
                    passed = False
                else:
                    passed = True
                    models.append(model.ms_model_id)
                finally:
                    if passed:
                        with open('./models.txt', 'w') as f:
                            json.dump(models, f)
                    

    def test_template_load(self):
        if os.path.exists('./templates.txt'):
            with open('./templates.txt', 'r') as f:
                templates = json.load(f)
        else:
            templates = []
        self.llm_ds = self.load_ds('AI-ModelScope/sharegpt_gpt4:default')
        self.img_ds = self.load_ds('swift/OK-VQA_train')
        self.audio_ds = self.load_ds('speech_asr/speech_asr_aishell1_trainsets:validation')
        for model_name, model_meta in MODEL_MAPPING.items():
            model_type = model_meta.model_type
            template = model_meta.template
            requires = model_meta.requires
            # for req in (requires or []):
            #     os.system(f'pip install {req}')
            for group in model_meta.model_groups:
                model = group.models[0]
                if template in templates:
                    break
                try:
                    _, tokenizer = get_model_tokenizer(model.ms_model_id, load_model=False)
                    template_ins = get_template(template, tokenizer)
                    if 'audio' in template_ins.__class__.__name__.lower():
                        EncodePreprocessor(template_ins)(self.audio_ds)
                    elif 'vl' in template_ins.__class__.__name__.lower():
                        EncodePreprocessor(template_ins)(self.img_ds)
                    else:
                        EncodePreprocessor(template_ins)(self.llm_ds)
                    
                except Exception as e:
                    import traceback
                    print(traceback.format_exc())
                    passed = False
                else:
                    passed = True
                    templates.append(template)
                finally:
                    if passed:
                        with open('./templates.txt', 'w') as f:
                            json.dump(templates, f)


if __name__ == '__main__':
    unittest.main()
