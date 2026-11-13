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
