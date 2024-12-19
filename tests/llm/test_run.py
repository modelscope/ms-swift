if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import os
import shutil
import tempfile
import unittest
from functools import partial
from typing import Any, Dict, List

import torch
from datasets import Dataset as HfDataset
from modelscope import Model, MsDataset, snapshot_download
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from swift import Trainer, TrainingArguments, get_logger
from swift.llm import (InferArguments, ModelType, RLHFArguments, TrainArguments, infer_main, merge_lora, rlhf_main,
                       sft_main)

NO_EVAL_HUMAN = True

logger = get_logger()

kwargs = {
    'per_device_train_batch_size': 2,
    'per_device_eval_batch_size': 2,
    'save_steps': 10,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
}


class TestRun(unittest.TestCase):

    def setUp(self):
        print(f'Testing {type(self).__name__}.{self._testMethodName}')
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_template(self):
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        torch.cuda.empty_cache()
        output = sft_main(
            TrainArguments(
                model='Qwen/Qwen1.5-0.5B',
                train_type='full',
                dataset='DAMO_NLP/jd',
                val_dataset='DAMO_NLP/jd#20',
                streaming=True,
                max_steps=12,
                **kwargs))
        last_model_checkpoint = output['last_model_checkpoint']
        torch.cuda.empty_cache()
        result = infer_main(InferArguments(model=last_model_checkpoint, load_data_args=True, val_dataset_sample=2))
        assert len(result[0]['response']) < 20

    def test_hf_hub(self):
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        torch.cuda.empty_cache()
        train_dataset_fnames = [
            'alpaca.csv', 'chatml.jsonl', 'swift_pre.jsonl', 'swift_single.csv', 'swift_multi.jsonl',
            'swift_multi.json#2'
        ]
        folder = os.path.join(os.path.dirname(__file__), 'data')
        dataset = [
            'llm-wizard/alpaca-gpt4-data-zh#20',
            'shibing624/alpaca-zh#20',
        ] + [os.path.join(folder, fname) for fname in train_dataset_fnames]
        output = sft_main(
            TrainArguments(
                model='Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4', train_type='lora', dataset=dataset, use_hf=True, **kwargs))
        last_model_checkpoint = output['last_model_checkpoint']
        torch.cuda.empty_cache()
        infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, val_dataset_sample=2))

    @unittest.skip('avoid ci error')
    def test_basic(self):
        output_dir = 'output'
        quant_bits_list = [0, 4]
        train_dataset_fnames = [
            'alpaca.csv', 'chatml.jsonl', 'swift_pre.jsonl', 'swift_single.csv', 'swift_multi.jsonl',
            'swift_multi.json#2'
        ]
        folder = os.path.join(os.path.dirname(__file__), 'data')
        dataset = [
            'AI-ModelScope/alpaca-gpt4-data-zh#20',
            'hurner/alpaca-gpt4-data-zh#20',
        ] + [os.path.join(folder, fname) for fname in train_dataset_fnames]
        if not __name__ == '__main__':
            output_dir = self.tmp_dir
            quant_bits_list = [4]
            dataset = dataset[:2]
        for quant_bits in quant_bits_list:
            if quant_bits == 0:
                predict_with_generate = False
                quant_method = None
            else:
                predict_with_generate = True
                quant_method = 'bnb'
            sft_args = TrainArguments(
                model='Qwen/Qwen2-0.5B-Instruct',
                quant_bits=quant_bits,
                eval_steps=5,
                adam_beta2=0.95,
                quant_method=quant_method,
                predict_with_generate=predict_with_generate,
                dataset=dataset,
                val_dataset='DAMO_NLP/jd#20',
                output_dir=output_dir,
                download_mode='force_redownload',
                include_num_input_tokens_seen=True,
                gradient_checkpointing=True,
                **kwargs)
            torch.cuda.empty_cache()
            output = sft_main(sft_args)
            print(output)
            best_model_checkpoint = output['best_model_checkpoint']
            print(f'best_model_checkpoint: {best_model_checkpoint}')
            if __name__ == '__main__':
                infer_args = InferArguments(
                    adapters=best_model_checkpoint,
                    merge_lora={
                        0: True,
                        4: False
                    }[quant_bits],
                    load_data_args=NO_EVAL_HUMAN,
                    val_dataset_sample=5)
                torch.cuda.empty_cache()
                result = infer_main(infer_args)
                print(result)
        # if __name__ == '__main__':
        #     app_ui_main(infer_args)

    def test_vl_audio(self):
        output_dir = 'output'
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        model_type_list = ['Qwen/Qwen-VL-Chat', 'Qwen/Qwen-Audio-Chat']
        dataset_list = [
            'modelscope/coco_2014_caption:validation#100', 'speech_asr/speech_asr_aishell1_trainsets:validation#100'
        ]
        for model, dataset in zip(model_type_list, dataset_list):
            sft_args = TrainArguments(
                model=model,
                eval_steps=5,
                dataset=[dataset],
                output_dir=output_dir,
                gradient_checkpointing=True,
                lazy_tokenize=True,
                disable_tqdm=True,
                **kwargs)
            torch.cuda.empty_cache()
            output = sft_main(sft_args)
            print(output)
            best_model_checkpoint = output['best_model_checkpoint']
            print(f'best_model_checkpoint: {best_model_checkpoint}')
            infer_args = InferArguments(
                adapters=best_model_checkpoint,
                load_data_args=True,
                stream={
                    'Qwen/Qwen-VL-Chat': True,
                    'Qwen/Qwen-Audio-Chat': False
                }[model],
                val_dataset_sample=5)
            torch.cuda.empty_cache()
            result = infer_main(infer_args)
            print(result)

    def test_custom_dataset(self):
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        train_dataset_fnames = [
            'alpaca.csv', 'chatml.jsonl', 'swift_pre.jsonl', 'swift_single.csv', 'swift_multi.jsonl',
            'swift_multi.json', 'sharegpt.jsonl'
        ]
        val_dataset_fnames = [
            'alpaca.jsonl',
            'alpaca2.csv',
            'conversations.jsonl',
            'swift_pre.csv',
            'swift_single.jsonl',
            # 'swift_#:#.jsonl#3'
        ]
        folder = os.path.join(os.path.dirname(__file__), 'data')
        resume_from_checkpoint = None
        train_kwargs = kwargs.copy()
        train_kwargs.pop('num_train_epochs')
        for num_train_epochs in [1, 2]:
            sft_args = TrainArguments(
                model='Qwen/Qwen-7B-Chat',
                dataset=['swift/self-cognition#20'] + [os.path.join(folder, fname) for fname in train_dataset_fnames],
                val_dataset=[os.path.join(folder, fname) for fname in val_dataset_fnames],
                resume_from_checkpoint=resume_from_checkpoint,
                num_train_epochs=num_train_epochs,
                model_name='小黄',
                model_author='魔搭',
                **train_kwargs)

            torch.cuda.empty_cache()
            result = sft_main(sft_args)
            best_model_checkpoint = result['best_model_checkpoint']
            resume_from_checkpoint = result['last_model_checkpoint']

        for load_args in [True, False]:
            infer_kwargs = {}
            if load_args is False:
                args_json = os.path.join(best_model_checkpoint, 'args.json')
                assert os.path.exists(args_json)
                os.remove(args_json)
                infer_kwargs = {'model': 'Qwen/Qwen-7B-Chat'}
            infer_args = InferArguments(
                adapters=best_model_checkpoint,
                load_data_args=load_args and NO_EVAL_HUMAN,
                merge_lora=load_args,
                val_dataset=[os.path.join(folder, fname) for fname in val_dataset_fnames],
                **infer_kwargs)
            torch.cuda.empty_cache()
            infer_main(infer_args)

    def test_rlhf(self):
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        torch.cuda.empty_cache()
        # llm rlhf
        #
        rlhf_types = ['dpo', 'orpo', 'simpo', 'kto', 'cpo']  # , 'rm', 'ppo'
        for rlhf_type in rlhf_types:
            dataset = ('AI-ModelScope/hh_rlhf_cn:harmless_base_cn#100'
                       if rlhf_type != 'kto' else 'AI-ModelScope/ultrafeedback-binarized-preferences-cleaned-kto#100')
            train_kwargs = {}
            if rlhf_type == 'ppo':
                train_kwargs['reward_model_type'] = 'Qwen/Qwen2-1.5B-Instruct'
            output = rlhf_main(
                RLHFArguments(
                    rlhf_type=rlhf_type,
                    model='Qwen/Qwen2-1.5B-Instruct',
                    dataset=dataset,
                    eval_steps=5,
                    **train_kwargs,
                    **kwargs))
            if rlhf_type == 'ppo':
                model_checkpoint = output['last_model_checkpoint']
            else:
                model_checkpoint = output['best_model_checkpoint']

            torch.cuda.empty_cache()
            infer_main(InferArguments(adapters=model_checkpoint, load_data_args=True))

        # mllm rlhf
        visual_rlhf_types = ['dpo', 'orpo', 'simpo', 'cpo']  # 'rm'
        #  'florence-2-base-ft'
        # 'swift/llava-v1.6-mistral-7b-hf',
        test_model = ['OpenGVLab/InternVL2-2B', 'Qwen/Qwen2-VL-2B-Instruct']  # decoder only and encoder-decoder
        for rlhf_type in visual_rlhf_types:
            for model in test_model:
                dataset_name = 'swift/RLAIF-V-Dataset#100'
                output = rlhf_main(
                    RLHFArguments(
                        rlhf_type=rlhf_type,
                        model=model,
                        dataset=dataset_name,
                        eval_steps=5,
                        dataset_num_proc=16,
                        **kwargs))
                best_model_checkpoint = output['best_model_checkpoint']
                torch.cuda.empty_cache()
                infer_main(InferArguments(adapters=best_model_checkpoint, load_data_args=True, val_dataset_sample=2))

    def test_loss_matching(self):
        output_dir = 'output'
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        losses = []
        for use_swift_lora in [False, True]:
            bool_var = use_swift_lora
            torch.cuda.empty_cache()
            output = sft_main([
                '--model', 'Qwen/Qwen-7B-Chat', '--eval_steps', '5', '--dataset',
                'AI-ModelScope/leetcode-solutions-python#200', '--output_dir', output_dir, '--gradient_checkpointing',
                'true', '--max_new_tokens', '100', '--attn_impl', 'flash_attn', '--target_modules', 'all-linear',
                '--seed', '0', '--lora_bias', 'all', '--modules_to_save', 'lm_head', '--use_swift_lora',
                str(use_swift_lora), '--num_train_epochs', '1', '--gradient_accumulation_steps', '16'
            ])
            best_model_checkpoint = output['best_model_checkpoint']
            print(f'best_model_checkpoint: {best_model_checkpoint}')
            load_data_args = str(bool_var or NO_EVAL_HUMAN)
            if load_data_args:
                val_dataset_sample = 2
            else:
                val_dataset_sample = -1
            torch.cuda.empty_cache()
            infer_main([
                '--adapters', best_model_checkpoint, '--val_dataset_sample',
                str(val_dataset_sample), '--max_new_tokens', '100', '--attn_impl', 'eager', '--merge_lora',
                str(bool_var), '--load_data_args',
                str(load_data_args)
            ])
            loss = output['log_history'][-1]['train_loss']
            losses.append(loss)
        self.assertTrue(abs(losses[0] - losses[1]) < 5e-4)
        print(f'swift_loss: {losses[0]}')
        print(f'peft_loss: {losses[1]}')
        self.assertTrue(0.95 <= losses[0] <= 1)

    def test_pai_compat(self):
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        from swift.llm import sft_main, infer_main
        os.environ['PAI_TRAINING_JOB_ID'] = '123456'
        folder = os.path.join(os.path.dirname(__file__), 'config')
        tensorboard_dir = os.path.join('output/pai_test', 'pai_tensorboard')
        os.environ['PAI_OUTPUT_TENSORBOARD'] = tensorboard_dir
        sft_json = os.path.join(folder, 'sft.json')
        infer_json = os.path.join(folder, 'infer.json')
        torch.cuda.empty_cache()
        output = sft_main([sft_json])
        print()
        infer_args = {
            'adapters': output['best_model_checkpoint'],
            'val_dataset_sample': 2,
            'load_data_args': True,
        }
        import json
        with open(infer_json, 'w') as f:
            json.dump(infer_args, f, ensure_ascii=False, indent=4)
        torch.cuda.empty_cache()
        infer_main([infer_json])
        os.environ.pop('PAI_TRAINING_JOB_ID')


def data_collate_fn(batch: List[Dict[str, Any]], tokenizer) -> Dict[str, torch.Tensor]:
    # text-classification
    assert tokenizer.pad_token_id is not None
    input_ids = [torch.tensor(b['input_ids']) for b in batch]
    labels = torch.tensor([b['labels'] for b in batch])
    attention_mask = [torch.ones(len(input_ids[i]), dtype=torch.int64) for i in range(len(input_ids))]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


class BertTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        if loss is None:
            logits, loss = list(outputs.logits)
        return (loss, outputs) if return_outputs else loss


class TestTrainer(unittest.TestCase):

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name
        # self.tmp_dir = 'test'
        logger.info(f'self.tmp_dir: {self.tmp_dir}')

    def tearDown(self):
        if os.path.isdir(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        # api = HubApi()
        # api.delete_model(self.hub_model_id)
        # logger.info(f'delete model: {self.hub_model_id}')

    def test_trainer(self):
        self.hub_model_id = 'test_trainer2'
        logger.info(f'self.hub_model_id: {self.hub_model_id}')
        self.tmp_dir = 'output/damo/nlp_structbert_backbone_base_std'
        push_to_hub = True
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        model_id = 'damo/nlp_structbert_backbone_base_std'
        model_dir = snapshot_download(model_id, 'master')
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        dataset = MsDataset.load('clue', subset_name='tnews')
        num_labels = max(dataset['train']['label']) + 1
        model = Model.from_pretrained(model_dir, task='text-classification', num_labels=num_labels)
        train_dataset, val_dataset = dataset['train'].to_hf_dataset(), dataset['validation'].to_hf_dataset()
        train_dataset: HfDataset = train_dataset.select(range(100))
        val_dataset: HfDataset = val_dataset.select(range(20))

        #
        def tokenize_func(examples):
            data = tokenizer(examples['sentence'], return_attention_mask=False)
            examples['input_ids'] = data['input_ids']
            examples['labels'] = examples['label']
            del examples['sentence'], examples['label']
            return examples

        train_dataset = train_dataset.map(tokenize_func)
        val_dataset = val_dataset.map(tokenize_func)

        data_collator = partial(data_collate_fn, tokenizer=tokenizer)
        for save_only_model in [True, False]:
            trainer_args = TrainingArguments(
                self.tmp_dir,
                do_train=True,
                do_eval=True,
                num_train_epochs=1,
                evaluation_strategy='steps',
                save_strategy='steps',
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                push_to_hub=push_to_hub,
                hub_token=None,  # use env var
                hub_private_repo=True,
                hub_strategy='every_save',
                hub_model_id=self.hub_model_id,
                overwrite_output_dir=True,
                save_steps=10,
                save_total_limit=2,
                metric_for_best_model='loss',
                greater_is_better=False,
                report_to=['tensorboard'],
                gradient_accumulation_steps=1,
                logging_steps=5,
                eval_steps=10,
                save_safetensors=False,
                save_only_model=save_only_model)
        trainer_args._n_gpu = 1
        trainer = BertTrainer(model, trainer_args, data_collator, train_dataset, val_dataset, tokenizer)
        self.hub_model_id = trainer_args.hub_model_id
        trainer.train()
        if trainer_args.push_to_hub:
            trainer.push_to_hub()


if __name__ == '__main__':
    # TestRun().test_template()
    # TestRun().test_hf_hub()
    # TestRun().test_basic()
    # TestRun().test_custom_dataset()
    # TestRun().test_vl_audio()
    # TestRun().test_loss_matching()
    #
    # TestRun().test_rlhf()
    unittest.main()
