# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import unittest
from types import SimpleNamespace

from swift.arguments import RLHFArguments
from swift.rlhf_trainers.arguments import GKDConfig
from swift.rlhf_trainers.gkd_loss import TeacherOutput, gkd_loss, mopd_gkd_loss
from swift.rlhf_trainers.gkd_trainer import GKDTrainer


class TestMOPDTrainer(unittest.TestCase):

    def _trainer(self, loss_type):
        trainer = object.__new__(GKDTrainer)
        trainer.args = SimpleNamespace(gkd_loss_type=loss_type)
        trainer.gkd_logits_topk = 3
        trainer.beta = .5
        trainer.temperature = 2.
        return trainer

    def test_mopd_dispatch_and_causal_mask(self):
        torch.manual_seed(31)
        student = torch.randn(1, 5, 9, dtype=torch.float64, requires_grad=True)
        teacher = torch.randn_like(student)
        labels = torch.tensor([[-100, -100, 1, 2, 3]])
        trainer = self._trainer('mopd')
        actual = trainer._compute_jsd_loss(student, TeacherOutput(full_logits=teacher, labels=labels), labels)
        targets = TeacherOutput(full_logits=teacher, labels=labels.roll(-1, 1)).to_topk(3, normalize=True)
        expected, count = mopd_gkd_loss(student, targets, labels.roll(-1, 1))
        torch.testing.assert_close(actual, expected / count)
        actual_grad, = torch.autograd.grad(actual, student, retain_graph=True)
        expected_grad, = torch.autograd.grad(expected / count, student)
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_default_jsd_dispatch_unchanged(self):
        torch.manual_seed(37)
        student = torch.randn(1, 4, 7, dtype=torch.float64, requires_grad=True)
        teacher = torch.randn_like(student)
        labels = torch.tensor([[-100, 1, 2, 3]])
        trainer = self._trainer('jsd')
        actual = trainer._compute_jsd_loss(student, TeacherOutput(full_logits=teacher, labels=labels), labels)
        targets = TeacherOutput(full_logits=teacher, labels=labels.roll(-1, 1)).to_topk(3)
        expected, count = gkd_loss(student, targets, labels.roll(-1, 1), trainer.beta, trainer.temperature)
        torch.testing.assert_close(actual, expected / count, atol=0., rtol=0.)
        self.assertEqual(GKDConfig.__dataclass_fields__['gkd_loss_type'].default, 'jsd')

    def test_unsupported_configurations_fail_before_training(self):
        for topk, liger, sequence_parallel in [(None, False, 1), (3, True, 1), (3, False, 2)]:
            with self.subTest(topk=topk, liger=liger, sequence_parallel=sequence_parallel):
                args = SimpleNamespace(
                    rlhf_type='gkd',
                    gkd_loss_type='mopd',
                    gkd_logits_topk=topk,
                    use_liger_kernel=liger,
                    sequence_parallel_size=sequence_parallel)
                with self.assertRaises(ValueError):
                    RLHFArguments._check_gkd(args)


if __name__ == '__main__':
    unittest.main()
