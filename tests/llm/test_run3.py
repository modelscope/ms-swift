if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import os
import shutil
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
        self.llm_ds = self.load_ds('AI-ModelScope/alpaca-gpt4-data-zh')
        self.img_ds = self.load_ds('swift/OK-VQA_train')
        self.audio_ds = self.load_ds('speech_asr/speech_asr_aishell1_trainsets')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def load_ds(self, ds):
        train_dataset, val_dataset = load_dataset(
                ds + ':all',
                split_dataset_ratio=0.0,
                strict=False,
                num_proc=12,
                model_name=['小黄', 'Xiao Huang'],
                model_author=['魔搭', 'ModelScope'])
        return train_dataset.select(range(min(50, len(train_dataset))))

    def test_model_load(self):
        for model_name, model_meta in MODEL_MAPPING.items():
            model_type = model_meta.model_type
            template = model_meta.template
            requires = model_meta.requires
            for req in (requires or []):
                os.system(f'pip install {req}')
            for group in model_meta.model_groups:
                model = group.models[0]
                model_ins = None
                try:
                    model_ins, tokenizer = get_model_tokenizer(model.ms_model_id)
                    template = get_template(template, tokenizer)
                    if 'audio' in template.__class__.__name__.lower():
                        EncodePreprocessor(template)(self.audio_ds)
                    elif 'vl' in template.__class__.__name__.lower():
                        EncodePreprocessor(template)(self.img_ds)
                    else:
                        EncodePreprocessor(template)(self.llm_ds)
                    
                except Exception as e:
                    import traceback
                    print(traceback.format_exc())
                finally:
                    if model_ins is not None:
                        del model_ins


if __name__ == '__main__':
    unittest.main()
