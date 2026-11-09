# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
import unittest
from torch import nn
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

from swift.optimizers.muon import MuonOptimizerCallback


class TestMuonOptimizerArgs(unittest.TestCase):

    def make_optimizer(self, optim_args):
        model = nn.Linear(4, 4)
        model.model_meta = SimpleNamespace(model_arch=None)
        args = SimpleNamespace(
            local_repo_path='/unused/moonlight',
            optim_args=optim_args,
            learning_rate=0.01,
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-8)
        constructor = Mock(return_value=sentinel.optimizer)
        # Only test the callback boundary here; the optional Moonlight checkout is not required.
        with patch.dict(sys.modules,
                        {'toy_train': SimpleNamespace(Muon=constructor)}), patch.object(sys, 'path', sys.path[:]):
            optimizer = MuonOptimizerCallback(args, SimpleNamespace(model=model)).create_optimizer()
        self.assertIs(optimizer, sentinel.optimizer)
        return constructor.call_args.kwargs

    def test_typed_options(self):
        cases = (
            ('momentum=0.8', {
                'momentum': 0.8
            }),
            ('ns_steps=3', {
                'ns_steps': 3
            }),
            ('nesterov=false', {
                'nesterov': False
            }),
            ('nesterov=True', {
                'nesterov': True
            }),
            ('momentum=0.85, nesterov=false, ns_steps=4', {
                'momentum': 0.85,
                'nesterov': False,
                'ns_steps': 4
            }),
        )
        for text, expected in cases:
            with self.subTest(optim_args=text):
                actual = self.make_optimizer(text)
                for name, value in expected.items():
                    self.assertEqual(actual[name], value)
                    self.assertIs(type(actual[name]), type(value))
                self.assertEqual(actual['lr'], 0.01)
                self.assertEqual(actual['adamw_betas'], (0.9, 0.95))

    def test_backend_defaults_are_preserved(self):
        for text in (None, ''):
            with self.subTest(optim_args=text):
                actual = self.make_optimizer(text)
                self.assertTrue({'momentum', 'nesterov', 'ns_steps'}.isdisjoint(actual))


if __name__ == '__main__':
    unittest.main()
