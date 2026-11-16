# Copyright (c) ModelScope Contributors. All rights reserved.
import torch.nn as nn
import unittest

from swift.model.decision_head import ScoringHead


class _StubHead(ScoringHead):
    """Minimal head used to exercise the base-class save helpers."""

    def forward(self, hidden_states, lm_logits=None, meta=None):
        raise NotImplementedError


class TestScoringHeadOutputEmbeddings(unittest.TestCase):

    def test_output_embeddings_are_not_registered_as_child_modules(self):
        head = _StubHead()
        lm_head = nn.Linear(8, 16, bias=False)

        head.set_output_embeddings(lm_head)

        # The head still sees the live reference ...
        self.assertIs(head._output_embeddings, lm_head)
        # ... but it must not show up in the head's own state. A registered child would add
        # `_output_embeddings.weight`, which aliases the backbone lm_head and is not listed in the
        # backbone `_tied_weights_keys`, so `save_pretrained` would fail the tied-weight check.
        self.assertFalse(any(key.startswith('_output_embeddings') for key in head.state_dict()))
        self.assertFalse(any(key.startswith('_output_embeddings') for key in head.head_state_dict()))

    def test_attached_head_keeps_model_state_dict_clean(self):

        class _Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(8, 16, bias=False)
                self.scoring_head = _StubHead()

        model = _Model()
        model.scoring_head.set_output_embeddings(model.lm_head)

        # Full fine-tuning saves the head through the model state dict; no synthetic key may leak in.
        self.assertFalse(any('_output_embeddings' in key for key in model.state_dict()))


class TestClefLinearInputRanks(unittest.TestCase):
    """The legacy per-record Clef path must never feed a 1-D tensor into ``nn.Linear``.

    DeepSpeed ZeRO-3 replaces ``F.linear`` with an implementation whose backward only reshapes
    inputs with ``dim() > 2``; a 1-D input degrades into a dot product and raises
    ``inconsistent tensor size`` on the first backward pass.
    """

    def _build(self):
        import torch

        from swift.model.decision_head import ClefJointSchemaHead

        head = ClefJointSchemaHead(hidden_size=32, width=8, routing_layers=1, layers=1, heads=2, feedforward=16)
        sequence_hidden = torch.randn(12, 32)
        input_ids_row = torch.randint(0, 64, (12, ))
        output_embedding_weight = torch.randn(64, 32)
        return torch, head, sequence_hidden, input_ids_row, output_embedding_weight

    def test_per_record_path_keeps_linear_inputs_two_dimensional(self):
        torch, head, sequence_hidden, input_ids_row, output_embedding_weight = self._build()

        seen_dims = []
        original_linear = torch.nn.functional.linear

        def _spy(input, weight, bias=None):
            seen_dims.append(input.dim())
            return original_linear(input, weight, bias)

        torch.nn.functional.linear = _spy
        try:
            logits = head._score_record(sequence_hidden, input_ids_row, 0, [0, 1], [(0, 2), (3, 5)],
                                        [[(5, 6), (6, 7)], [(7, 8)]], [0, 2], output_embedding_weight)
        finally:
            torch.nn.functional.linear = original_linear

        self.assertEqual(len(logits), 2)
        self.assertTrue(seen_dims)
        self.assertGreaterEqual(
            min(seen_dims), 2,
            'nn.Linear received a 1-D input; ZeRO-3 backward cannot handle it (dims seen: %s)' % seen_dims)
