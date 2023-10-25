import os
import shutil
import tempfile
import unittest
from functools import partial
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset as HfDataset
from modelscope import Model, MsDataset, snapshot_download
from numpy.random import RandomState
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoTokenizer

from swift import Trainer, TrainingArguments, get_logger

logger = get_logger()


def data_collate_fn(batch: List[Dict[str, Any]],
                    tokenizer) -> Dict[str, Tensor]:
    # text-classification
    assert tokenizer.pad_token_id is not None
    input_ids = [torch.tensor(b['input_ids']) for b in batch]
    labels = torch.tensor([b['labels'] for b in batch])
    attention_mask = [
        torch.ones(len(input_ids[i]), dtype=torch.int64)
        for i in range(len(input_ids))
    ]

    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(
        attention_mask, batch_first=True, padding_value=0)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


class BertTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Saving model checkpoint to {output_dir}')
        self.model.save_pretrained(output_dir, 'pytorch_model.bin')
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))


class TestTrainer(unittest.TestCase):

    def setUp(self):
        print(f'Testing {type(self).__name__}.{self._testMethodName}')
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name
        # self.tmp_dir = 'test'
        logger.info(f'self.tmp_dir: {self.tmp_dir}')
        self.hub_model_id = 'test_trainer2'
        logger.info(f'self.hub_model_id: {self.hub_model_id}')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        # api = HubApi()
        # api.delete_model(self.hub_model_id)
        # logger.info(f'delete model: {self.hub_model_id}')

    def test_trainer(self):
        push_to_hub = True
        if not __name__ == '__main__':
            # ignore citest error in github
            push_to_hub = False
        model_id = 'damo/nlp_structbert_backbone_base_std'
        model_dir = snapshot_download(model_id, 'master')
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        dataset = MsDataset.load('clue', subset_name='tnews')
        num_labels = max(dataset['train']['label']) + 1
        model = Model.from_pretrained(
            model_dir, task='text-classification', num_labels=num_labels)
        train_dataset, val_dataset = dataset['train'].to_hf_dataset(
        ), dataset['validation'].to_hf_dataset()
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
            eval_steps=10)
        trainer_args._n_gpu = 1
        trainer = BertTrainer(model, trainer_args, data_collator,
                              train_dataset, val_dataset, tokenizer)
        self.hub_model_id = trainer_args.hub_model_id
        trainer.train()
        if trainer_args.push_to_hub:
            trainer.push_to_hub()


if __name__ == '__main__':
    unittest.main()
