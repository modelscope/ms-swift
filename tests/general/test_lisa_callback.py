# Copyright (c) ModelScope Contributors. All rights reserved.
import tempfile
import torch
import unittest
from torch import nn
from transformers import Trainer, TrainerCallback, TrainingArguments
from types import SimpleNamespace
from unittest.mock import patch

from swift.callbacks.lisa import LISACallback


class TinyLayerModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.head = nn.Linear(2, 1)

    def forward(self, input_ids, labels=None):
        hidden = input_ids.float()
        for layer in self.layers:
            hidden = layer(hidden).tanh()
        logits = self.head(hidden).squeeze(-1)
        loss = None if labels is None else (logits - labels).square().mean()
        return {'loss': loss, 'logits': logits}


def make_args(tmp_path):
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        max_steps=3,
        per_device_train_batch_size=1,
        learning_rate=0.05,
        lr_scheduler_type='constant',
        optim='adamw_torch',
        save_strategy='no',
        report_to=[],
        disable_tqdm=True)
    args.tuner_type = 'full'
    args.lisa_activated_layers = 1
    args.lisa_step_interval = 1
    return args


class TestLISACallback(unittest.TestCase):

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = directory.name

    def test_lisa_can_be_created_before_trainer_initialization(self):
        args = make_args(self.tmp_path)
        trainer = Trainer.__new__(Trainer)
        # Swift constructs its callbacks before calling the HF Trainer constructor.
        callback = LISACallback(args, trainer)
        model = TinyLayerModel()
        Trainer.__init__(trainer, model=model, args=args, callbacks=[callback])
        trainer.create_optimizer()
        optimized = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
        assert all(id(p) in optimized for p in model.parameters())
        trainer.callback_handler.on_train_begin(args, trainer.state, trainer.control)
        assert sum(any(p.requires_grad for p in layer.parameters()) for layer in model.layers) == 1

    def test_lisa_does_not_freeze_layers_before_optimizer_creation(self):
        model = TinyLayerModel()
        args = make_args(self.tmp_path)
        LISACallback(args, SimpleNamespace(model=model))
        assert all(p.requires_grad for p in model.parameters())

    def test_newly_activated_lisa_layers_receive_optimizer_updates(self):
        torch.manual_seed(42)
        model = TinyLayerModel()
        args = make_args(self.tmp_path)
        active = [0]
        choose = patch('swift.callbacks.lisa.np.random.choice', side_effect=lambda *args, **kwargs: [active[0]])
        choose.start()
        self.addCleanup(choose.stop)
        callback = LISACallback(args, SimpleNamespace(model=model))
        updated_layers = []

        class CheckUpdates(TrainerCallback):

            def on_step_begin(self, args, state, control, **kwargs):
                self.before = [[p.detach().clone() for p in layer.parameters()] for layer in model.layers]

            def on_step_end(self, args, state, control, **kwargs):
                for index, layer in enumerate(model.layers):
                    changed = any(not torch.equal(before, after)
                                  for before, after in zip(self.before[index], layer.parameters()))
                    assert changed == (index == active[0])
                updated_layers.append(active[0])
                active[0] = 1 - active[0]

        trainer = Trainer(
            model=model,
            args=args,
            callbacks=[callback, CheckUpdates()],
            train_dataset=[{
                'input_ids': [0.5, 1.0],
                'labels': 2.0
            }] * 3)
        trainer.train()
        assert updated_layers == [0, 1, 0]

    def test_lisa_rejects_adapter_training(self):
        args = make_args(self.tmp_path)
        args.tuner_type = 'lora'
        with self.assertRaisesRegex(AssertionError, 'full parameter training'):
            LISACallback(args, SimpleNamespace())


if __name__ == '__main__':
    unittest.main()
