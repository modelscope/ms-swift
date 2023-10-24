if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import shutil
import tempfile
import unittest

from swift.llm import DatasetName, InferArguments, ModelType, SftArguments
from swift.llm.run import infer_main, sft_main


class TestRun(unittest.TestCase):

    def setUp(self):
        print(f'Testing {type(self).__name__}.{self._testMethodName}')
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_run(self):
        output_dir = self.tmp_dir
        # output_dir = 'output'
        # test predict_with_generate=True
        sft_args = SftArguments(
            model_type=ModelType.qwen_7b_chat_int4,
            eval_steps=10,
            train_dataset_sample=400,
            predict_with_generate=True,
            dataset=[DatasetName.jd_sentiment_zh],
            output_dir=output_dir,
            use_flash_attn=False,
            gradient_checkpointing=True)
        ckpt_dir = sft_main(sft_args)
        # test predict_with_generate=False
        ckpt_dir = sft_main([
            '--model_type', ModelType.qwen_7b_chat_int4, '--eval_steps', '10',
            '--train_dataset_sample', '400', '--predict_with_generate',
            'false', '--dataset', DatasetName.jd_sentiment_zh, '--output_dir',
            output_dir, '--use_flash_attn', 'false',
            '--gradient_checkpointing', 'true'
        ])
        print(ckpt_dir)
        # test stream=False
        infer_args = InferArguments(
            model_type=ModelType.qwen_7b_chat_int4,
            ckpt_dir=ckpt_dir,
            dataset=[DatasetName.jd_sentiment_zh],
            stream=False,
            show_dataset_sample=5)
        infer_main(infer_args)
        # test stream=True
        infer_main([
            '--model_type', ModelType.qwen_7b_chat_int4, '--ckpt_dir',
            ckpt_dir, '--dataset', DatasetName.jd_sentiment_zh,
            '--show_dataset_sample', '5'
        ])


if __name__ == '__main__':
    unittest.main()
