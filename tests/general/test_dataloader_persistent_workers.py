# Copyright (c) ModelScope Contributors. All rights reserved.
import tempfile
import torch
import unittest
from datasets import Dataset
from transformers import Trainer as HfTrainer
from transformers import default_data_collator
from types import SimpleNamespace
from unittest import mock

from swift.trainers import Seq2SeqTrainingArguments, TrainingArguments
from swift.trainers.mixin import DataLoaderMixin


class TestDataloaderPersistentWorkers(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)

    def make_args(self, args_cls, **kwargs):
        return args_cls(
            output_dir=self.tmp_dir.name, use_cpu=True, report_to=[], train_dataloader_shuffle=False, **kwargs)

    def check_training_loader(self, args, dataset):
        trainer = DataLoaderMixin()
        trainer.args = args
        trainer.template = SimpleNamespace(sequence_parallel_size=1)
        trainer.accelerator = SimpleNamespace(device=torch.device('cpu'))
        trainer.train_dataset = dataset
        trainer.data_collator = default_data_collator
        trainer._train_batch_size = 1
        loader = trainer.get_train_dataloader()
        self.assertEqual([batch['input_ids'].item() for batch in loader], [1, 2])

    def test_zero_workers_can_load_map_and_iterable_datasets(self):
        rows = Dataset.from_dict({'input_ids': [[1], [2]]})
        for args_cls in (TrainingArguments, Seq2SeqTrainingArguments):
            for dataset in (rows, rows.to_iterable_dataset()):
                with self.subTest(args_cls=args_cls.__name__, dataset=type(dataset).__name__):
                    args = self.make_args(args_cls, dataloader_num_workers=0)
                    self.check_training_loader(args, dataset)

    def test_zero_workers_can_load_eval_and_prediction_data(self):
        rows = Dataset.from_dict({'input_ids': [[1], [2]]})
        for args_cls in (TrainingArguments, Seq2SeqTrainingArguments):
            with self.subTest(args_cls=args_cls.__name__):
                args = self.make_args(args_cls, dataloader_num_workers=0, remove_unused_columns=False)
                trainer = HfTrainer(
                    model=torch.nn.Linear(1, 1), args=args, eval_dataset=rows, data_collator=default_data_collator)
                for loader in (trainer.get_eval_dataloader(), trainer.get_test_dataloader(rows)):
                    self.assertEqual(torch.cat([batch['input_ids'] for batch in loader]).flatten().tolist(), [1, 2])

    def test_explicit_persistence_settings_with_zero_workers(self):
        for persistent in (True, False):
            with self.subTest(persistent=persistent):
                args = self.make_args(
                    Seq2SeqTrainingArguments, dataloader_num_workers=0, dataloader_persistent_workers=persistent)
                self.check_training_loader(args, [{'input_ids': [1]}, {'input_ids': [2]}])

    def test_explicit_prefetch_with_zero_workers_still_raises(self):
        with self.assertRaises(ValueError):
            self.make_args(Seq2SeqTrainingArguments, dataloader_num_workers=0, dataloader_prefetch_factor=2)

    def test_windows_default_can_load_data(self):
        # Exercise Windows argument defaults without starting platform-specific worker processes.
        for args_cls in (TrainingArguments, Seq2SeqTrainingArguments):
            with self.subTest(args_cls=args_cls.__name__):
                with mock.patch('swift.trainers.arguments.platform.system', return_value='Windows'):
                    args = self.make_args(args_cls)
                self.assertEqual(args.dataloader_num_workers, 0)
                self.check_training_loader(args, [{'input_ids': [1]}, {'input_ids': [2]}])

    def test_positive_workers_keep_default_and_explicit_opt_out(self):
        for args_cls in (TrainingArguments, Seq2SeqTrainingArguments):
            with self.subTest(args_cls=args_cls.__name__):
                args = self.make_args(args_cls, dataloader_num_workers=1)
                self.assertTrue(args.dataloader_persistent_workers)
                args = self.make_args(args_cls, dataloader_num_workers=1, dataloader_persistent_workers=False)
                self.assertFalse(args.dataloader_persistent_workers)


if __name__ == '__main__':
    unittest.main()
