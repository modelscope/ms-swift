if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import os
import shutil
import tempfile
import unittest

import torch

from swift import get_logger
from swift.llm import DatasetName, InferArguments, ModelType, SftArguments, infer_main, sft_main

NO_EVAL_HUMAN = True

logger = get_logger()


class TestRun2(unittest.TestCase):

    def setUp(self):
        print(f'Testing {type(self).__name__}.{self._testMethodName}')
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_loss_matching(self):
        output_dir = 'output'
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        losses = []
        for tuner_backend in ['swift', 'peft']:
            if tuner_backend == 'swift':
                bool_var = True
            else:
                bool_var = False
            torch.cuda.empty_cache()
            output = sft_main([
                '--model_type', ModelType.qwen_7b_chat, '--eval_steps', '5', '--tuner_backend', tuner_backend,
                '--dataset', f'{DatasetName.leetcode_python_en}#200', '--output_dir', output_dir,
                '--gradient_checkpointing', 'true', '--max_new_tokens', '100', '--use_flash_attn', 'true',
                '--lora_target_modules', 'ALL', '--seed', '0', '--lora_bias_trainable', 'all', '--lora_modules_to_save',
                'EMBEDDING', 'LN', 'lm_head'
            ])
            best_model_checkpoint = output['best_model_checkpoint']
            print(f'best_model_checkpoint: {best_model_checkpoint}')
            load_dataset_config = str(bool_var or NO_EVAL_HUMAN)
            if load_dataset_config:
                val_dataset_sample = 2
            else:
                val_dataset_sample = -1
            torch.cuda.empty_cache()
            infer_main([
                '--ckpt_dir', best_model_checkpoint, '--val_dataset_sample',
                str(val_dataset_sample), '--max_new_tokens', '100', '--use_flash_attn', 'false', '--verbose',
                str(not bool_var), '--merge_lora',
                str(bool_var), '--load_dataset_config',
                str(load_dataset_config)
            ])
            loss = output['log_history'][-1]['train_loss']
            losses.append(loss)
        self.assertTrue(abs(losses[0] - losses[1]) < 5e-4)
        print(f'swift_loss: {losses[0]}')
        print(f'peft_loss: {losses[1]}')
        self.assertTrue(0.95 <= losses[0] <= 1)

    def test_yi_vl_6b_chat(self):
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        folder = os.path.join(os.path.dirname(__file__), 'data')
        torch.cuda.empty_cache()
        output = sft_main(
            SftArguments(
                model_type=ModelType.yi_vl_6b_chat,
                #   dataset=DatasetName.capcha_images,
                lora_target_modules='ALL',
                train_dataset_sample=100,
                eval_steps=5,
                custom_train_dataset_path=[os.path.join(folder, 'multi_modal_2.jsonl')],
                lazy_tokenize=False))
        best_model_checkpoint = output['best_model_checkpoint']
        torch.cuda.empty_cache()
        infer_main(InferArguments(ckpt_dir=best_model_checkpoint, load_dataset_config=True, val_dataset_sample=2))

    def test_glm4v_9b_chat(self):
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        folder = os.path.join(os.path.dirname(__file__), 'data')
        torch.cuda.empty_cache()
        output = sft_main(
            SftArguments(
                model_type=ModelType.glm4v_9b_chat,
                #   dataset=DatasetName.capcha_images,
                lora_target_modules='ALL',
                train_dataset_sample=100,
                eval_steps=5,
                custom_train_dataset_path=[os.path.join(folder, 'multi_modal_3.jsonl')],
                lazy_tokenize=False))
        best_model_checkpoint = output['best_model_checkpoint']
        torch.cuda.empty_cache()
        infer_main(
            InferArguments(
                ckpt_dir=best_model_checkpoint, load_dataset_config=True, val_dataset_sample=2, merge_lora=True))

    def test_baichuan2_chat_int4(self):
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        from swift.llm import sft_main, infer_main, SftArguments, InferArguments, ModelType, DatasetName
        output = sft_main(
            SftArguments(
                model_type=ModelType.baichuan2_7b_chat_int4,
                dataset=['alpaca-zh'],
                lora_target_modules=['DEFAULT', 'EMBEDDING'],
                train_dataset_sample=20))
        best_model_checkpoint = output['best_model_checkpoint']
        infer_main(InferArguments(ckpt_dir=best_model_checkpoint, load_dataset_config=True, val_dataset_sample=1))

    def test_self_cognition(self):
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        for dataset in [[], [DatasetName.alpaca_zh, DatasetName.alpaca_en]]:
            sft_args = SftArguments(
                model_type=ModelType.qwen1half_1_8b_chat_int4,
                dataset=dataset,  # no dataset
                train_dataset_sample=100,
                dtype='fp16',
                eval_steps=5,
                output_dir='output',
                lora_target_modules=['ALL', 'EMBEDDING'],
                lazy_tokenize=True,
                max_length=512,
                self_cognition_sample=100,
                model_name=['小黄', 'Xiao Huang'],
                model_author=['魔搭', 'ModelScope'],
                use_flash_attn=True)
            torch.cuda.empty_cache()
            output = sft_main(sft_args)
            last_model_checkpoint = output['last_model_checkpoint']
            best_model_checkpoint = output['best_model_checkpoint']
            print(f'last_model_checkpoint: {last_model_checkpoint}')
            print(f'best_model_checkpoint: {best_model_checkpoint}')
            ckpt_dir = best_model_checkpoint or last_model_checkpoint
            if len(dataset) == 0:
                continue
            infer_args = InferArguments(
                ckpt_dir=ckpt_dir, val_dataset_sample=2, verbose=False, load_dataset_config=True)
            # merge_lora_main(infer_args)
            torch.cuda.empty_cache()
            result = infer_main(infer_args)
            print(result)


if __name__ == '__main__':
    unittest.main()
