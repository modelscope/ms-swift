if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import shutil
import tempfile
import unittest

from swift.llm import DatasetName, ModelType
from swift.llm.run import infer_main, sft_main


class TestRun(unittest.TestCase):

    def setUp(self):
        print(f'Testing {type(self).__name__}.{self._testMethodName}')
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_run(self):
        for predict_with_generate in ['false', 'true']:
            ckpt_dir = sft_main([
                '--model_type', ModelType.qwen_7b_chat_int4, '--eval_steps',
                '10', '--train_dataset_sample', '400',
                '--predict_with_generate', predict_with_generate, '--dataset',
                DatasetName.jd_sentiment_zh, '--output', self.tmp_dir,
                '--use_flash_attn', 'false', '--gradient_checkpointing', 'true'
            ])
            print(ckpt_dir)
        infer_main([
            '--model_type', ModelType.qwen_7b_chat_int4, '--ckpt_dir',
            ckpt_dir, '--dataset', DatasetName.jd_sentiment_zh
        ])


if __name__ == '__main__':
    unittest.main()
