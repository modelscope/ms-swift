if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import os
import shutil
import tempfile
import unittest

import torch

from swift.llm import DatasetName, InferArguments, ModelType, SftArguments
from swift.llm.run import infer_main, merge_lora_main, sft_main


class TestRun(unittest.TestCase):

    def setUp(self):
        print(f'Testing {type(self).__name__}.{self._testMethodName}')
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_basic(self):
        output_dir = 'output'
        if not __name__ == '__main__':
            output_dir = self.tmp_dir
        model_type = ModelType.chatglm3_6b
        sft_args = SftArguments(
            model_type=model_type,
            template_type='AUTO',
            quantization_bit=4,
            batch_size=2,
            eval_steps=5,
            check_dataset_strategy='warning',
            train_dataset_sample=200,
            predict_with_generate=False,
            dataset=[DatasetName.jd_sentiment_zh],
            output_dir=output_dir,
            gradient_checkpointing=True)
        self.assertTrue(sft_args.gradient_accumulation_steps == 8)
        output = sft_main(sft_args)
        print(output)
        best_model_checkpoint = output['best_model_checkpoint']
        print(f'best_model_checkpoint: {best_model_checkpoint}')
        torch.cuda.empty_cache()
        if __name__ == '__main__':
            infer_args = InferArguments(
                ckpt_dir=best_model_checkpoint,
                stream=False,
                show_dataset_sample=5)
            merge_lora_main(infer_args)
            result = infer_main(infer_args)
            print(result)
            torch.cuda.empty_cache()
        # if __name__ == '__main__':
        #     web_ui_main(infer_args)

    def test_loss_matching(self):
        output_dir = 'output'
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        losses = []
        for tuner_backend in ['swift', 'peft']:
            output = sft_main([
                '--model_type', ModelType.qwen_7b_chat_int4, '--eval_steps',
                '5', '--tuner_backend', tuner_backend,
                '--train_dataset_sample', '200', '--predict_with_generate',
                'true', '--dataset', DatasetName.leetcode_python_en,
                '--output_dir', output_dir, '--gradient_checkpointing', 'true',
                '--max_new_tokens', '100', '--use_flash_attn', 'true',
                '--lora_target_modules', 'ALL'
            ])
            best_model_checkpoint = output['best_model_checkpoint']
            print(f'best_model_checkpoint: {best_model_checkpoint}')
            torch.cuda.empty_cache()
            infer_main([
                '--ckpt_dir', best_model_checkpoint, '--show_dataset_sample',
                '2', '--max_new_tokens', '100', '--use_flash_attn', 'true'
            ])
            loss = output['log_history'][-1]['train_loss']
            losses.append(loss)
            torch.cuda.empty_cache()
        self.assertTrue(abs(losses[0] - losses[1]) < 1e-4)

    def test_vl_audio(self):
        output_dir = 'output'
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        model_type_list = [ModelType.qwen_vl_chat, ModelType.qwen_audio_chat]
        dataset_list = [DatasetName.coco_mini_en, DatasetName.aishell1_mini_zh]
        for model_type, dataset in zip(model_type_list, dataset_list):
            sft_args = SftArguments(
                model_type=model_type,
                template_type='AUTO',
                eval_steps=5,
                check_dataset_strategy='warning',
                train_dataset_sample=200,
                dataset=[dataset],
                output_dir=output_dir,
                gradient_checkpointing=True)
            output = sft_main(sft_args)
            print(output)
            best_model_checkpoint = output['best_model_checkpoint']
            print(f'best_model_checkpoint: {best_model_checkpoint}')
            torch.cuda.empty_cache()
            infer_args = InferArguments(
                ckpt_dir=best_model_checkpoint,
                stream=False,
                show_dataset_sample=5)
            # merge_lora_main(infer_args)  # TODO: ERROR FIX
            result = infer_main(infer_args)
            print(result)
            torch.cuda.empty_cache()

    def test_custom_dataset(self):
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        train_dataset_fnames = [
            'alpaca.csv', 'chatml.jsonl', 'swift_pre.jsonl',
            'swift_single.csv', 'swift_multi.jsonl', 'swift_multi.json'
        ]
        val_dataset_fnames = [
            'alpaca.jsonl', 'alpaca2.csv', 'conversations.jsonl',
            'swift_pre.csv', 'swift_single.jsonl'
        ]
        folder = os.path.join(os.path.dirname(__file__), 'data')
        sft_args = SftArguments(
            model_type='qwen-7b-chat',
            custom_train_dataset_path=[
                os.path.join(folder, fname) for fname in train_dataset_fnames
            ],
            check_dataset_strategy='warning')
        best_model_checkpoint = sft_main(sft_args)['best_model_checkpoint']
        torch.cuda.empty_cache()
        for load_args_from_ckpt_dir in [True, False]:
            kwargs = {}
            if load_args_from_ckpt_dir is False:
                kwargs = {'model_type': 'qwen-7b-chat'}
            infer_args = InferArguments(
                ckpt_dir=best_model_checkpoint,
                load_args_from_ckpt_dir=load_args_from_ckpt_dir,
                val_dataset_sample=-1,
                custom_val_dataset_path=[
                    os.path.join(folder, fname) for fname in val_dataset_fnames
                ],
                **kwargs)
            infer_main(infer_args)
            torch.cuda.empty_cache()

    def test_self_cognition(self):
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        for dataset in [None, [DatasetName.alpaca_zh, DatasetName.alpaca_en]]:
            sft_args = SftArguments(
                model_type=ModelType.qwen_7b_chat,
                dataset=dataset,  # no dataset
                train_dataset_sample=100,
                eval_steps=5,
                output_dir='output',
                lora_target_modules='ALL',
                self_cognition_sample=100,
                model_name=['小黄', 'Xiao Huang'],
                model_author=['魔搭', 'ModelScope'],
                use_flash_attn=False)
            output = sft_main(sft_args)
            torch.cuda.empty_cache()
            last_model_checkpoint = output['last_model_checkpoint']
            best_model_checkpoint = output['best_model_checkpoint']
            print(f'last_model_checkpoint: {last_model_checkpoint}')
            print(f'best_model_checkpoint: {best_model_checkpoint}')
            ckpt_dir = best_model_checkpoint or last_model_checkpoint
            if dataset is None:
                continue
            infer_args = InferArguments(
                ckpt_dir=ckpt_dir, show_dataset_sample=2)
            # merge_lora_main(infer_args)
            result = infer_main(infer_args)
            print(result)
            torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
