# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU contracts for native Megatron GKD token normalization.

Load the actual argument class and loss methods through AST to avoid importing
CUDA, vLLM and model-loading infrastructure. The loss math is the real shared
gkd_loss module. Collectives are mocked; these are not distributed scheduler tests.

Run: python -m pytest tests/megatron/test_gkd_global_token_mean.py -q
"""
import ast
import importlib.util
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Literal, Optional
from unittest import mock

try:
    import torch
    TORCH_UNAVAILABLE = None
except (ImportError, OSError) as error:
    torch = None
    TORCH_UNAVAILABLE = str(error)

ROOT = Path(__file__).resolve().parents[2]


def _load_node(relative_path, name, namespace, owner=None):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    nodes = tree.body
    if owner is not None:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == owner).body
    node = next(node for node in nodes if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


@dataclass
class _SftArguments:
    use_ray: bool = False
    context_parallel_size: int = 1

    def __post_init__(self):
        self.parent_post_init_called = True


class TestGKDTokenArguments(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.args_cls = _load_node(
            'swift/megatron/arguments/rlhf_args.py', 'MegatronRLHFArguments', {
                '__name__': __name__,
                'dataclass': dataclass,
                'Literal': Literal,
                'Optional': Optional,
                'MegatronSftArguments': _SftArguments,
            })

    def test_native_gkd_default(self):
        args = self.args_cls(rlhf_type='gkd')
        self.assertIs(args.calculate_per_token_loss, True)
        self.assertTrue(args.parent_post_init_called)

    def test_other_objective_defaults_unchanged(self):
        for objective in ('dpo', 'kto', 'grpo', 'rm'):
            with self.subTest(objective=objective):
                args = self.args_cls(rlhf_type=objective)
                self.assertIs(args.calculate_per_token_loss, False)

    def test_ray_default_unchanged(self):
        args = self.args_cls(rlhf_type='gkd', use_ray=True, context_parallel_size=2)
        self.assertIs(args.calculate_per_token_loss, False)

    def test_explicit_legacy_opt_out(self):
        args = self.args_cls(rlhf_type='gkd', calculate_per_token_loss=False, context_parallel_size=2)
        self.assertIs(args.calculate_per_token_loss, False)

    def test_explicit_values_preserved_for_other_backends(self):
        for kwargs in ({'rlhf_type': 'grpo'}, {'rlhf_type': 'gkd', 'use_ray': True}):
            for value in (False, True):
                with self.subTest(kwargs=kwargs, value=value):
                    self.assertIs(
                        self.args_cls(**kwargs, calculate_per_token_loss=value).calculate_per_token_loss, value)

    def test_cp_requires_explicit_scope_decision(self):
        for value in (None, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(NotImplementedError, 'context_parallel_size=1'):
                    self.args_cls(rlhf_type='gkd', context_parallel_size=2, calculate_per_token_loss=value)


@unittest.skipIf(TORCH_UNAVAILABLE is not None, TORCH_UNAVAILABLE or '')
class TestGKDGlobalTokenMean(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path = ROOT / 'swift/rlhf_trainers/gkd_loss.py'
        spec = importlib.util.spec_from_file_location('_gkd_token_test_math', path)
        cls.math = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {spec.name: cls.math}):
            spec.loader.exec_module(cls.math)
        cls.mpu = SimpleNamespace(get_context_parallel_world_size=lambda: 1, get_data_parallel_group=lambda: 'dp')
        namespace = {'torch': torch, 'mpu': cls.mpu, 'Dict': Dict}
        cp_reduce = _load_node('swift/megatron/trainers/gkd_utils.py', 'cp_reduce', namespace)
        cls.reduce_metric = staticmethod(
            _load_node('swift/megatron/trainers/base.py', '_all_reduce_metric', namespace, 'BaseMegatronTrainer'))
        cls.namespace = {
            'torch': torch,
            'mpu': cls.mpu,
            'DataSource': cls.math.DataSource,
            'TeacherOutput': cls.math.TeacherOutput,
            'gkd_loss': cls.math.gkd_loss,
            'cp_reduce': cp_reduce,
            'tp_gather_topk': cls.math.default_gather,
            'vocab_parallel_log_softmax': cls.math.default_log_softmax,
            'vocab_parallel_kl_div': cls.math.default_kl_div,
        }
        cls.loss_func = staticmethod(
            _load_node('swift/megatron/trainers/gkd_trainer.py', 'loss_func', cls.namespace, 'MegatronGKDTrainer'))

    def _trainer(self, beta=1.0, alpha=0.0, per_token=True):

        def language_model_loss(labels, logits_sbv):
            logits = logits_sbv.transpose(0, 1)
            return torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction='none',
                ignore_index=-100).reshape_as(labels)

        trainer = SimpleNamespace(
            args=SimpleNamespace(calculate_per_token_loss=per_token, context_parallel_size=1),
            beta=beta,
            temperature=1.0,
            sft_alpha=alpha,
            unwrapped_models=[SimpleNamespace(compute_language_model_loss=language_model_loss)],
            _step=1,
            steps_per_generation=2,
            _flush_log_completions=mock.Mock(),
        )
        trainer._all_reduce_metric = lambda metric, **kwargs: self.reduce_metric(trainer, metric, **kwargs)
        return trainer

    def _teacher(self, logits, labels, mode):
        if mode == 'full':
            return self.math.TeacherOutput(full_logits=logits, labels=labels)
        values, indices = logits.topk(3, dim=-1)
        if mode == 'partial':
            values = values.clone()
            values[:, -1] = -torch.inf
        return self.math.TeacherOutput(topk_logprobs=values, topk_indices=indices, labels=labels)

    def _reference(self, logits, labels, teacher, beta, alpha=0.0):
        mask = labels != -100
        selected = logits[mask]
        if teacher.is_topk_mode:
            teacher_logits = teacher.topk_logprobs[mask]
            covered = ~torch.isinf(teacher_logits).all(-1)
            selected = selected.gather(-1, teacher.topk_indices[mask])[covered]
            teacher_logits = teacher_logits[covered]
        else:
            teacher_logits = teacher.full_logits[mask]
        student_log = selected.log_softmax(-1)
        teacher_log = teacher_logits.log_softmax(-1)
        if beta == 0:
            tokens = (teacher_log.exp() * (teacher_log - student_log)).sum(-1)
        elif beta == 1:
            tokens = (student_log.exp() * (student_log - teacher_log)).sum(-1)
        else:
            mixed = torch.logaddexp(student_log + student_log.new_tensor(1 - beta).log(),
                                    teacher_log + teacher_log.new_tensor(beta).log())
            tokens = (
                beta * (teacher_log.exp() * (teacher_log - mixed)).sum(-1) + (1 - beta) *
                (student_log.exp() * (student_log - mixed)).sum(-1))
        result = tokens.sum() / tokens.numel()
        if alpha:
            result = result + alpha * torch.nn.functional.cross_entropy(logits[mask], labels[mask])
        return result

    def _check_partition(self, mode, beta, alpha=0.0, source=None):
        generator = torch.Generator().manual_seed(19)
        inputs = torch.randn(4, 4, 7, generator=generator, dtype=torch.float64)
        teacher_logits = torch.randn(4, 4, 7, generator=generator, dtype=torch.float64)
        labels = torch.tensor([[1, -100, -100, -100], [0, 2, 1, 4], [3, 5, -100, -100], [-100] * 4])
        reference_weight = torch.zeros(7, dtype=torch.float64, requires_grad=True)
        teacher = self._teacher(teacher_logits, labels, mode)
        active_alpha = alpha if source != self.math.DataSource.STUDENT else 0.0
        expected = self._reference(inputs + reference_weight, labels, teacher, beta, active_alpha)
        expected.backward()
        for groups in (([0], [1], [2], [3]), ([0, 1], [2, 3]), ([0, 1, 2, 3], )):
            with self.subTest(groups=groups):
                weight = torch.zeros(7, dtype=torch.float64, requires_grad=True)
                trainer = self._trainer(beta, alpha)
                numerator = torch.zeros((), dtype=torch.float64)
                denominator = torch.zeros((), dtype=torch.int)
                stats = torch.zeros(2, dtype=torch.float64)
                with mock.patch.object(torch.distributed, 'all_reduce') as reduce:
                    for group in groups:
                        batch_labels = labels[group]
                        result = self.loss_func(
                            trainer,
                            inputs[group] + weight,
                            labels=batch_labels,
                            teacher_output=self._teacher(teacher_logits[group], batch_labels, mode),
                            data_source=source or self.math.DataSource.DATASET)
                        self.assertEqual(len(result), 3)
                        loss_sum, count, metric = result
                        self.assertEqual(count.dtype, torch.int)
                        self.assertTrue(loss_sum.requires_grad)
                        loss_sum.backward()
                        numerator += loss_sum.detach()
                        denominator += count
                        stats += metric['loss']
                    for call in reduce.call_args_list:
                        operation = call.kwargs.get('op', call.args[1] if len(call.args) > 1 else None)
                        self.assertIn(operation, (torch.distributed.ReduceOp.SUM, torch.distributed.ReduceOp.MAX))
                torch.testing.assert_close(numerator / denominator, expected.detach(), rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(weight.grad / denominator, reference_weight.grad, rtol=1e-11, atol=1e-12)
                torch.testing.assert_close(stats[0] / stats[1], expected.detach(), rtol=1e-12, atol=1e-12)

    def test_full_vocab_loss_and_gradients(self):
        for beta in (0.0, 0.5, 1.0):
            with self.subTest(beta=beta):
                self._check_partition('full', beta)

    def test_topk_loss_and_gradients(self):
        for beta in (0.0, 0.5, 1.0):
            with self.subTest(beta=beta):
                self._check_partition('topk', beta)

    def test_partial_teacher_coverage_uses_scored_token_count(self):
        self._check_partition('partial', 1.0)

    def test_sft_mixture_uses_token_sums(self):
        for mode in ('full', 'topk'):
            with self.subTest(mode=mode):
                self._check_partition(mode, 1.0, alpha=0.3)

    def test_student_rollouts_skip_sft(self):
        self._check_partition('partial', 1.0, alpha=0.3, source=self.math.DataSource.STUDENT)

    def test_partial_teacher_coverage_rejects_sft_mixture(self):
        labels = torch.tensor([[0, 1]])
        logits = torch.zeros(1, 2, 7, dtype=torch.float64, requires_grad=True)
        teacher = self._teacher(logits.detach(), labels, 'partial')
        with mock.patch.object(torch.distributed, 'all_reduce'):
            with self.assertRaisesRegex(ValueError, 'teacher scores for every'):
                self.loss_func(self._trainer(alpha=0.3), logits, labels=labels, teacher_output=teacher)

    def test_coverage_error_is_shared_with_other_dp_ranks(self):
        labels = torch.tensor([[0, 1]])
        logits = torch.zeros(1, 2, 7, dtype=torch.float64, requires_grad=True)

        def remote_mismatch(value, op, group):
            self.assertEqual(op, torch.distributed.ReduceOp.MAX)
            self.assertEqual(group, 'dp')
            value.fill_(1)

        with mock.patch.object(torch.distributed, 'all_reduce', side_effect=remote_mismatch):
            with self.assertRaisesRegex(ValueError, 'teacher scores for every'):
                self.loss_func(
                    self._trainer(alpha=0.3),
                    logits,
                    labels=labels,
                    teacher_output=self._teacher(logits.detach(), labels, 'full'))

    def test_dp_metric_sum_keeps_local_gradient_normalizer(self):
        trainer = self._trainer()
        logits = torch.tensor([[[1.0]]], requires_grad=True)
        labels = torch.tensor([[0]])

        def token_values(student, *_args, **_kwargs):
            return student.sum(), labels.new_tensor(1)

        def sum_remote(value, op, group):
            self.assertEqual(op, torch.distributed.ReduceOp.SUM)
            self.assertEqual(group, 'dp')
            value.add_(value.new_tensor([[9.0, 3.0]]))

        with mock.patch.dict(self.namespace, {'gkd_loss': token_values}):
            with mock.patch.object(torch.distributed, 'all_reduce', side_effect=sum_remote):
                loss, count, metric = self.loss_func(trainer, logits, labels=labels, teacher_output=None)
        self.assertEqual(count.item(), 1)
        self.assertEqual(loss.item(), 1)
        torch.testing.assert_close(metric['loss'], torch.tensor([10.0, 4.0]))
        self.assertEqual((metric['loss'][0] / metric['loss'][1]).item(), 2.5)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.ones_like(logits))

    def test_empty_mask_keeps_zero_gradient_and_zero_count(self):
        logits = torch.randn(1, 2, 7, dtype=torch.float64, requires_grad=True)
        labels = torch.full((1, 2), -100)
        with mock.patch.object(torch.distributed, 'all_reduce'):
            loss, count, metric = self.loss_func(
                self._trainer(), logits, labels=labels, teacher_output=self._teacher(logits.detach(), labels, 'full'))
        self.assertEqual(count.item(), 0)
        self.assertEqual(loss.item(), 0)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
        torch.testing.assert_close(metric['loss'], torch.zeros(2, dtype=torch.float64))

    def test_low_precision_logits_do_not_round_metric_counts(self):
        logits = torch.zeros(1, 1, 1, dtype=torch.bfloat16, requires_grad=True)
        labels = torch.tensor([[0]])

        def token_values(student, *_args, **_kwargs):
            return student.sum(), labels.new_tensor(257)

        with mock.patch.dict(self.namespace, {'gkd_loss': token_values}):
            with mock.patch.object(torch.distributed, 'all_reduce'):
                _, count, metric = self.loss_func(self._trainer(), logits, labels=labels, teacher_output=None)
        self.assertEqual(count.item(), 257)
        self.assertEqual(metric['loss'].dtype, torch.float32)
        self.assertEqual(metric['loss'][1].item(), 257)

    def test_legacy_mode_retains_two_value_interface(self):
        trainer = self._trainer(per_token=False)
        labels = torch.tensor([[0, 1]])
        logits = torch.tensor([[[2., 1., 0.], [0., 1., 3.]]], dtype=torch.float64, requires_grad=True)
        teacher = self._teacher(torch.zeros_like(logits), labels, 'full')
        with mock.patch.object(torch.distributed, 'all_reduce') as reduce:
            loss, metric = self.loss_func(trainer, logits, labels=labels, teacher_output=teacher)
        expected = self._reference(logits, labels, teacher, 1.0)
        torch.testing.assert_close(loss, expected)
        self.assertEqual(metric['loss'].numel(), 1)
        self.assertEqual(reduce.call_args.args[1], torch.distributed.ReduceOp.AVG)


if __name__ == '__main__':
    unittest.main()
