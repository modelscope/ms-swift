# Copyright (c) ModelScope Contributors. All rights reserved.
"""Gradient correctness of the vocab-parallel utilities under tensor parallelism.

Every helper in ``swift/megatron/trainers/vocab_parallel_utils.py`` shards the vocab
dimension across TP ranks and reduces with ``torch.distributed.all_reduce``. Raw
all_reduce is invisible to autograd, so a forward pass can be numerically perfect while
the backward pass silently drops the contributions of the other shards -- which makes the
bug undetectable from loss/reward curves alone. These tests pin the gradients against a
single-rank full-vocabulary reference.

Requires more than one GPU and must be launched with torchrun from the repository root
(torchrun puts the script's directory on sys.path, so ``swift`` needs PYTHONPATH)::

    PYTHONPATH=. torchrun --nproc_per_node=2 tests/megatron/test_vocab_parallel_grad.py
    PYTHONPATH=. torchrun --nproc_per_node=4 tests/megatron/test_vocab_parallel_grad.py

Running it directly (no torchrun) exercises the tp_size == 1 path only.
"""
import os
import unittest

import torch

BATCH, SEQ, VOCAB = 2, 6, 16
# -100 marks masked positions; the remaining targets are spread over every shard so that
# each rank sees both "owns the target" and "does not own the target" positions.
LABELS = [[0, 5, 11, -100, 15, 3], [7, 1, -100, 2, 9, 14]]
BETA = 0.5  # GKD interpolation: exercises the JSD branch that mixes both log-prob sets.


def _tp_world_size() -> int:
    return int(os.environ.get('WORLD_SIZE', '1'))


class TestVocabParallelGrad(unittest.TestCase):
    """Compare TP-sharded forward and backward against a full-vocab reference."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('requires CUDA')
        import torch.distributed as dist
        from megatron.core import mpu

        cls.rank = int(os.environ.get('RANK', '0'))
        cls.world = _tp_world_size()
        if cls.world > torch.cuda.device_count():
            raise unittest.SkipTest(f'requires {cls.world} GPUs')
        torch.cuda.set_device(cls.rank)
        if not dist.is_initialized():
            os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
            os.environ.setdefault('MASTER_PORT', '29591')
            dist.init_process_group('nccl', rank=cls.rank, world_size=cls.world)
        mpu.initialize_model_parallel(tensor_model_parallel_size=cls.world)

        cls.shard = VOCAB // cls.world
        cls.lo = cls.rank * cls.shard
        cls.hi = cls.lo + cls.shard
        # Identical on every rank, so each one can compute the reference locally.
        gen = torch.Generator(device='cuda').manual_seed(1234)
        randn = lambda *s: torch.randn(*s, generator=gen, device='cuda')  # noqa: E731
        cls.full = randn(BATCH, SEQ, VOCAB)
        cls.teacher = randn(BATCH, SEQ, VOCAB)
        cls.logit_weight = randn(BATCH, SEQ, VOCAB)
        cls.pos_weight = randn(BATCH, SEQ)
        cls.labels = torch.tensor(LABELS, device='cuda')

    @classmethod
    def tearDownClass(cls):
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()

    def _shard(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[..., self.lo:self.hi]

    def _reference(self, build):
        """Full-vocab single-rank ground truth for ``build(log_probs, teacher_log_probs)``."""
        logits = self.full.clone().requires_grad_(True)
        log_probs = torch.log_softmax(logits, dim=-1)
        teacher_log_probs = torch.log_softmax(self.teacher, dim=-1)
        out, loss = build(log_probs, teacher_log_probs)
        loss.backward()
        return out.detach(), logits.grad.clone()

    def _sharded(self, build):
        """Same quantity computed from this rank's vocab shard."""
        logits = self._shard(self.full).clone().requires_grad_(True)
        out, loss = build(logits)
        loss.backward()
        return out.detach(), logits.grad

    def _assert_matches(self, name, build_ref, build_sharded, shard_output: bool):
        want_out, want_grad = self._reference(build_ref)
        got_out, got_grad = self._sharded(build_sharded)
        if shard_output:
            want_out = self._shard(want_out)
        self.assertLess((got_out - want_out).abs().max().item(), 1e-5, f'{name}: forward mismatch')
        # The gradient is what a raw all_reduce silently gets wrong.
        self.assertLess((got_grad - self._shard(want_grad)).abs().max().item(), 1e-5, f'{name}: backward mismatch')

    def test_log_softmax(self):
        from swift.megatron.trainers.vocab_parallel_utils import vocab_parallel_log_softmax

        def ref(log_probs, _):
            return log_probs, (log_probs * self.logit_weight).sum()

        def sharded(logits):
            out = vocab_parallel_log_softmax(logits)
            return out, (out * self._shard(self.logit_weight)).sum()

        self._assert_matches('log_softmax', ref, sharded, shard_output=True)

    def test_entropy(self):
        from swift.megatron.trainers.vocab_parallel_utils import vocab_parallel_entropy, vocab_parallel_log_softmax

        def ref(log_probs, _):
            entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1)
            return entropy, (entropy * self.pos_weight).sum()

        def sharded(logits):
            entropy = vocab_parallel_entropy(vocab_parallel_log_softmax(logits))
            return entropy, (entropy * self.pos_weight).sum()

        self._assert_matches('entropy', ref, sharded, shard_output=False)

    def test_kl_div(self):
        from swift.megatron.trainers.vocab_parallel_utils import vocab_parallel_kl_div, vocab_parallel_log_softmax

        def ref(log_probs, teacher_log_probs):
            kl = (torch.exp(teacher_log_probs) * (teacher_log_probs - log_probs)).sum(dim=-1)
            return kl, (kl * self.pos_weight).sum()

        def sharded(logits):
            teacher = vocab_parallel_log_softmax(self._shard(self.teacher)).detach()
            kl = vocab_parallel_kl_div(vocab_parallel_log_softmax(logits), teacher)
            return kl, (kl * self.pos_weight).sum()

        self._assert_matches('kl_div', ref, sharded, shard_output=False)

    def test_gather_logps(self):
        from swift.megatron.trainers.vocab_parallel_utils import vocab_parallel_gather_logps

        def ref(log_probs, _):
            logps = self._target_logps(log_probs)
            return logps, (logps * self.pos_weight).sum()

        def sharded(logits):
            logps = vocab_parallel_gather_logps(logits, self.labels)
            return logps, (logps * self.pos_weight).sum()

        self._assert_matches('gather_logps', ref, sharded, shard_output=False)

    def test_logps_and_entropy_together(self):
        from swift.megatron.trainers.vocab_parallel_utils import compute_logps_and_entropy_from_logits

        def ref(log_probs, _):
            logps = self._target_logps(log_probs)
            entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1)
            return logps, (logps * self.pos_weight).sum() + (entropy * self.pos_weight).sum()

        def sharded(logits):
            logps, entropy = compute_logps_and_entropy_from_logits(logits, self.labels, compute_entropy=True)
            return logps, (logps * self.pos_weight).sum() + (entropy * self.pos_weight).sum()

        self._assert_matches('logps+entropy', ref, sharded, shard_output=False)

    def test_jsd_matches_reference(self):
        """The GKD consumer: log_softmax and kl_div composed as swift's jsd_loss does."""
        from swift.megatron.trainers.vocab_parallel_utils import vocab_parallel_kl_div, vocab_parallel_log_softmax

        log_beta = torch.log(torch.tensor(BETA, device='cuda'))
        log_1_minus_beta = torch.log1p(-torch.tensor(BETA, device='cuda'))

        def mixture(student_log, teacher_log):
            return torch.logsumexp(torch.stack([student_log + log_1_minus_beta, teacher_log + log_beta]), dim=0)

        def ref(log_probs, teacher_log_probs):
            mixed = mixture(log_probs, teacher_log_probs)
            jsd = (
                BETA * (torch.exp(teacher_log_probs) * (teacher_log_probs - mixed)).sum(dim=-1) + (1 - BETA) *
                (torch.exp(log_probs) * (log_probs - mixed)).sum(dim=-1))
            return jsd, jsd.sum()

        def sharded(logits):
            student_log = vocab_parallel_log_softmax(logits)
            teacher_log = vocab_parallel_log_softmax(self._shard(self.teacher)).detach()
            mixed = mixture(student_log, teacher_log)
            jsd = (
                BETA * vocab_parallel_kl_div(mixed, teacher_log) +
                (1 - BETA) * vocab_parallel_kl_div(mixed, student_log))
            return jsd, jsd.sum()

        self._assert_matches('jsd', ref, sharded, shard_output=False)

    def test_no_grad_builds_no_graph(self):
        """Reference-model and teacher paths run under no_grad; autograd.Function must not leak."""
        from swift.megatron.trainers.vocab_parallel_utils import (vocab_parallel_gather_logps,
                                                                  vocab_parallel_log_softmax)
        with torch.no_grad():
            log_probs = vocab_parallel_log_softmax(self._shard(self.full))
            self.assertFalse(log_probs.requires_grad)
            reference = torch.log_softmax(self.full, dim=-1)
            self.assertLess((log_probs - self._shard(reference)).abs().max().item(), 1e-5)
            # gather_logps consumes logits in place, so hand it a copy.
            logps = vocab_parallel_gather_logps(self._shard(self.full).clone(), self.labels)
            self.assertFalse(logps.requires_grad)

    def _target_logps(self, log_probs: torch.Tensor) -> torch.Tensor:
        mask = self.labels != -100
        targets = self.labels.masked_fill(~mask, 0)
        return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1) * mask


if __name__ == '__main__':
    unittest.main()
