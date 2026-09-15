# Copyright (c) ModelScope Contributors. All rights reserved.
import tempfile
import torch
import unittest
from torch import nn
from transformers import Trainer
from types import SimpleNamespace

from swift.optimizers.galore.utils import GaloreOptimizerCallback
from swift.trainers import TrainingArguments


class TestGaloreSchedule(unittest.TestCase):

    def test_split_hooks_create_galore_optimizer(self):
        from swift.optimizers.galore import GaLoreAdamW
        with tempfile.TemporaryDirectory() as output_dir:
            args = TrainingArguments(
                output_dir=output_dir,
                use_cpu=True,
                report_to=[],
                optim='adamw_torch',
                galore_target_modules=['0'],
                galore_rank=2)
            trainer = Trainer(args=args, model=nn.Sequential(nn.Linear(4, 4)))
            callback = GaloreOptimizerCallback(args, trainer)
            optimizer = callback.create_optimizer()
            self.assertIsInstance(optimizer, GaLoreAdamW)
            self.assertTrue(any('rank' in group for group in optimizer.param_groups))
            scheduler = callback.create_scheduler(5, optimizer)
            self.assertIs(scheduler.optimizer, optimizer)

    def test_learning_rate_follows_training_steps_and_warmup(self):
        cases = ((5, 0.4, 0), (5, 0.0, 2), (5, 0.0, 0), (4, 0.0, 0))
        for per_parameter in (False, True):
            for total_steps, warmup_ratio, warmup_steps in cases:
                with self.subTest(
                        per_parameter=per_parameter,
                        total_steps=total_steps,
                        warmup_ratio=warmup_ratio,
                        warmup_steps=warmup_steps):
                    with tempfile.TemporaryDirectory() as output_dir:
                        args = TrainingArguments(
                            output_dir=output_dir,
                            use_cpu=True,
                            report_to=[],
                            optim='adamw_torch',
                            learning_rate=0.01,
                            weight_decay=0.0,
                            lr_scheduler_type='linear',
                            warmup_ratio=warmup_ratio,
                            warmup_steps=warmup_steps,
                            per_device_train_batch_size=2,
                            gradient_accumulation_steps=2,
                            num_train_epochs=1,
                            galore_target_modules=['0'],
                            galore_rank=2,
                            galore_optim_per_parameter=per_parameter)
                        model = nn.Sequential(nn.Linear(4, 4))
                        trainer = SimpleNamespace(args=args, model=model, train_dataset=list(range(18)))
                        GaloreOptimizerCallback(args, trainer).create_optimizer_and_scheduler(total_steps)
                        optimizers = list(
                            trainer.optimizer.optimizers.values()) if per_parameter else [trainer.optimizer]
                        resolved_warmup = args.get_warmup_steps(total_steps)
                        learning_rates = []
                        for step in range(total_steps + 1):
                            rates = [group['lr'] for optimizer in optimizers for group in optimizer.param_groups]
                            expected_factor = (
                                step / max(1, resolved_warmup) if step < resolved_warmup else max(
                                    0.0, (total_steps - step) / max(1, total_steps - resolved_warmup)))
                            learning_rates.append((rates, args.learning_rate * expected_factor))
                            if step < total_steps:
                                model(torch.ones(2, 4)).square().mean().backward()
                                trainer.optimizer.step()
                                trainer.lr_scheduler.step()
                                trainer.optimizer.zero_grad()
                        for rates, expected in learning_rates:
                            for actual in rates:
                                self.assertAlmostEqual(actual, expected, places=10)


if __name__ == '__main__':
    unittest.main()
