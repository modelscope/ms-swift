if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import os
import shutil
import tempfile
import time
import unittest
from functools import partial
from typing import Any, Dict, List

import torch
import transformers
from datasets import Dataset as HfDataset
from modelscope import Model, MsDataset, snapshot_download
from packaging import version
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoTokenizer

from swift import Trainer, TrainingArguments, get_logger
from swift.llm import (DatasetName, InferArguments, ModelType, RLHFArguments, SftArguments, infer_main, merge_lora_main,
                       rlhf_main, sft_main)

NO_EVAL_HUMAN = True

logger = get_logger()


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
            SftArguments(
                model_type=ModelType.qwen1half_1_8b,
                model_id_or_path='../models/Qwen1.5-1.8B',
                template_type='qwen',
                sft_type='full',
                dataset=f'{DatasetName.jd_sentiment_zh}#200',
                eval_steps=5))
        best_model_checkpoint = output['best_model_checkpoint']
        torch.cuda.empty_cache()
        result = infer_main(
            InferArguments(ckpt_dir=best_model_checkpoint, load_dataset_config=True, val_dataset_sample=2))
        assert len(result['result'][0]['response']) < 20

    def test_basic(self):
        output_dir = 'output'
        quantization_bit_list = [0, 4]
        train_dataset_fnames = [
            'alpaca.csv', 'chatml.jsonl', 'swift_pre.jsonl', 'swift_single.csv', 'swift_multi.jsonl',
            'swift_multi.json#2'
        ]
        folder = os.path.join(os.path.dirname(__file__), 'data')
        dataset = [
            f'MS::{DatasetName.alpaca_zh}#20',
            f'{DatasetName.jd_sentiment_zh}#20',
            'AI-ModelScope/alpaca-gpt4-data-zh#20',
            'HF::llm-wizard/alpaca-gpt4-data-zh#20',
            'hurner/alpaca-gpt4-data-zh#20',
            'HF::shibing624/alpaca-zh#20',
        ] + [os.path.join(folder, fname) for fname in train_dataset_fnames]
        if not __name__ == '__main__':
            output_dir = self.tmp_dir
            quantization_bit_list = [4]
            dataset = dataset[:2]
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) >= version.parse('4.42'):
            model_type = ModelType.qwen2_0_5b_instruct
        else:
            model_type = ModelType.chatglm3_6b
        for quantization_bit in quantization_bit_list:
            if quantization_bit == 4 and version.parse(transformers.__version__) >= version.parse('4.38'):
                continue
            predict_with_generate = True
            if quantization_bit == 0:
                predict_with_generate = False
            sft_args = SftArguments(
                model_type=model_type,
                template_type='AUTO',
                lora_target_modules=['AUTO', 'EMBEDDING'],
                quantization_bit=quantization_bit,
                batch_size=2,
                eval_steps=5,
                adam_beta2=0.95,
                check_dataset_strategy='warning',
                predict_with_generate=predict_with_generate,
                dataset=dataset,
                val_dataset=f'{DatasetName.jd_sentiment_zh}#20',
                output_dir=output_dir,
                include_num_input_tokens_seen=True,
                gradient_checkpointing=True)
            self.assertTrue(sft_args.gradient_accumulation_steps == 8)
            torch.cuda.empty_cache()
            output = sft_main(sft_args)
            print(output)
            best_model_checkpoint = output['best_model_checkpoint']
            print(f'best_model_checkpoint: {best_model_checkpoint}')
            if __name__ == '__main__':
                infer_args = InferArguments(
                    ckpt_dir=best_model_checkpoint,
                    merge_lora={
                        0: True,
                        4: False
                    }[quantization_bit],
                    merge_device_map='cpu',
                    load_dataset_config=NO_EVAL_HUMAN,
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
        model_type_list = [ModelType.qwen_vl_chat, ModelType.qwen_audio_chat]
        dataset_list = [DatasetName.coco_en_mini, DatasetName.aishell1_zh_mini]
        for model_type, dataset in zip(model_type_list, dataset_list):
            sft_args = SftArguments(
                model_type=model_type,
                template_type='AUTO',
                eval_steps=5,
                check_dataset_strategy='warning',
                lora_target_modules='ALL',
                train_dataset_sample=200,
                dataset=[dataset],
                output_dir=output_dir,
                gradient_checkpointing=True,
                lazy_tokenize=True,
                disable_tqdm=True)
            torch.cuda.empty_cache()
            output = sft_main(sft_args)
            print(output)
            best_model_checkpoint = output['best_model_checkpoint']
            print(f'best_model_checkpoint: {best_model_checkpoint}')
            infer_args = InferArguments(
                ckpt_dir=best_model_checkpoint,
                load_dataset_config=True,
                stream={
                    ModelType.qwen_vl_chat: True,
                    ModelType.qwen_audio_chat: False
                }[model_type],
                val_dataset_sample=5)
            # merge_lora_main(infer_args)  # TODO: ERROR FIX
            torch.cuda.empty_cache()
            result = infer_main(infer_args)
            print(result)

    def test_vqa(self):
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        train_dataset_fnames = ['science-qa#300', 'a-okvqa#300', 'alpaca-cleaned#300']
        val_dataset_fnames = ['okvqa']

        sft_args = SftArguments(
            model_type='yi-vl-6b-chat',
            dataset=train_dataset_fnames,
            lora_target_modules='ALL',
            num_train_epochs=1,
            check_dataset_strategy='warning')

        torch.cuda.empty_cache()
        result = sft_main(sft_args)
        best_model_checkpoint = result['best_model_checkpoint']

        infer_args = InferArguments(
            ckpt_dir=best_model_checkpoint,
            load_args_from_ckpt_dir=True,
            load_dataset_config=True,
            merge_lora=False,
            val_dataset_sample=10,
            dataset=val_dataset_fnames)
        torch.cuda.empty_cache()
        infer_main(infer_args)

    def test_gpt4o_image(self):
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        train_dataset_fnames = ['sharegpt-4o-image']

        sft_args = SftArguments(
            model_type='yi-vl-6b-chat',
            dataset=train_dataset_fnames,
            lora_target_modules='ALL',
            train_dataset_sample=200,
            num_train_epochs=1,
            eval_steps=10,
            save_steps=10,
            check_dataset_strategy='warning')

        torch.cuda.empty_cache()
        self.assertTrue(sft_main(sft_args)['best_model_checkpoint'])

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
        for num_train_epochs in [1, 2]:
            sft_args = SftArguments(
                model_type='qwen-7b-chat',
                dataset=['self-cognition#20'],
                custom_train_dataset_path=[os.path.join(folder, fname) for fname in train_dataset_fnames],
                custom_val_dataset_path=[os.path.join(folder, fname) for fname in val_dataset_fnames],
                lora_target_modules='ALL',
                resume_from_checkpoint=resume_from_checkpoint,
                num_train_epochs=num_train_epochs,
                model_name='小黄',
                model_author='魔搭',
                check_dataset_strategy='warning')

            torch.cuda.empty_cache()
            result = sft_main(sft_args)
            best_model_checkpoint = result['best_model_checkpoint']
            resume_from_checkpoint = result['last_model_checkpoint']

        for load_args_from_ckpt_dir in [True, False]:
            kwargs = {}
            if load_args_from_ckpt_dir is False:
                kwargs = {'model_type': 'qwen-7b-chat'}
            infer_args = InferArguments(
                ckpt_dir=best_model_checkpoint,
                load_args_from_ckpt_dir=load_args_from_ckpt_dir,
                load_dataset_config=load_args_from_ckpt_dir and NO_EVAL_HUMAN,
                merge_lora=load_args_from_ckpt_dir,
                val_dataset_sample=-1,
                custom_val_dataset_path=[os.path.join(folder, fname) for fname in val_dataset_fnames],
                **kwargs)
            torch.cuda.empty_cache()
            infer_main(infer_args)

    def test_cogagent_instruct(self):
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        quantization_bit = 4
        if version.parse(transformers.__version__) >= version.parse('4.38'):
            quantization_bit = 0
        torch.cuda.empty_cache()
        output = sft_main(
            SftArguments(
                model_type=ModelType.cogagent_18b_instruct,
                dataset=DatasetName.coco_en_2_mini,
                train_dataset_sample=100,
                lora_target_modules='ALL',
                eval_steps=5,
                quantization_bit=quantization_bit))
        best_model_checkpoint = output['best_model_checkpoint']
        torch.cuda.empty_cache()
        infer_main(InferArguments(ckpt_dir=best_model_checkpoint, load_dataset_config=True, val_dataset_sample=2))

    def test_xcomposer_chat(self):
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        torch.cuda.empty_cache()
        output = sft_main(
            SftArguments(
                model_type=ModelType.internlm_xcomposer2_7b_chat,
                dataset=DatasetName.coco_en_mini,
                lora_target_modules='DEFAULT',
                train_dataset_sample=100,
                eval_steps=5))
        best_model_checkpoint = output['best_model_checkpoint']
        torch.cuda.empty_cache()
        infer_main(InferArguments(ckpt_dir=best_model_checkpoint, load_dataset_config=True, val_dataset_sample=2))

    def test_rlhf(self):
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        torch.cuda.empty_cache()
        rlhf_types = ['dpo', 'orpo', 'simpo', 'kto', 'cpo']
        for rlhf_type in rlhf_types:
            dataset_name = 'hh-rlhf-cn-harmless-base-cn' if rlhf_type != 'kto' else 'ultrafeedback-kto'
            output = rlhf_main(
                RLHFArguments(
                    rlhf_type=rlhf_type,
                    model_type=ModelType.qwen_1_8b_chat,
                    sft_type='full',
                    dataset=dataset_name,
                    train_dataset_sample=100,
                    eval_steps=5))
            best_model_checkpoint = output['best_model_checkpoint']
            torch.cuda.empty_cache()
            infer_main(InferArguments(ckpt_dir=best_model_checkpoint, load_dataset_config=True, val_dataset_sample=2))

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
            'ckpt_dir': output['best_model_checkpoint'],
            'val_dataset_sample': 2,
            'load_dataset_config': True,
        }
        import json
        with open(infer_json, 'w') as f:
            json.dump(infer_args, f, ensure_ascii=False, indent=4)
        torch.cuda.empty_cache()
        infer_main([infer_json])
        os.environ.pop('PAI_TRAINING_JOB_ID')

    def test_deepseek_vl_chat(self):
        if not __name__ == '__main__':
            # ignore citest error in github
            return
        folder = os.path.join(os.path.dirname(__file__), 'data')
        torch.cuda.empty_cache()
        sft_main(
            SftArguments(
                model_type=ModelType.deepseek_vl_1_3b_chat,
                #   dataset=DatasetName.capcha_images,
                lora_target_modules='ALL',
                train_dataset_sample=100,
                eval_steps=5,
                custom_train_dataset_path=[os.path.join(folder, 'multi_modal_1.jsonl')],
                lazy_tokenize=False))


def data_collate_fn(batch: List[Dict[str, Any]], tokenizer) -> Dict[str, Tensor]:
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
                push_hub_strategy='push_best',
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
                save_only_model=save_only_model)
        trainer_args._n_gpu = 1
        trainer = BertTrainer(model, trainer_args, data_collator, train_dataset, val_dataset, tokenizer)
        self.hub_model_id = trainer_args.hub_model_id
        trainer.train()
        if trainer_args.push_to_hub:
            trainer.push_to_hub()


if __name__ == '__main__':
    unittest.main()
