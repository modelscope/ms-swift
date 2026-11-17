"""Rank guard for the Clef head's legacy per-record scoring path.

DeepSpeed ZeRO-3 replaces ``torch.nn.functional.linear`` with its ``memory_efficient_linear``
wrapper (on by default). Its backward only reshapes inputs with ``dim() > 2`` and degrades
anything flatter into ``grad_output.t().matmul(input)``, so a 1-D input turns the
weight-gradient product into a dot product between a ``[width]`` and a ``[hidden]`` vector
and the first backward aborts with ``inconsistent tensor size``. The per-record
``ClefJointSchemaHead._score_record`` path historically fed ``nn.Linear`` exactly such 1-D
vectors: the ``unsqueeze(0)`` sat on the projection OUTPUT, one step too late for the 1-D
input that had already gone through. The fix keeps the result shape bit-identical by
unsqueezing the input instead; this guard pins that invariant so the regression cannot
silently return.

Pure spy check -- no weights on disk, no GPU, no DeepSpeed install needed.
"""
import torch
import torch.nn.functional as F

from swift.dev.model.decision_head import ClefJointSchemaHead


def _run_per_record_path():
    head = ClefJointSchemaHead(hidden_size=32, width=8, routing_layers=1, layers=1, heads=2, feedforward=16)
    head.eval()
    sequence_hidden = torch.randn(12, 32)
    input_ids_row = torch.randint(0, 64, (12, ))
    output_embedding_weight = torch.randn(64, 32)
    return head._score_record(
        sequence_hidden,
        input_ids_row,
        0,
        [0, 1],
        [(0, 2), (3, 5)],
        [[(5, 6), (6, 7)], [(7, 8)]],
        [0, 2],
        output_embedding_weight,
    )


def test_per_record_path_keeps_linear_inputs_at_least_two_dimensional(monkeypatch):
    seen_dims = []
    original_linear = F.linear

    def _spy(input, weight, bias=None):
        seen_dims.append(input.dim())
        return original_linear(input, weight, bias)

    monkeypatch.setattr(F, 'linear', _spy)
    logits = _run_per_record_path()

    assert len(logits) == 2
    assert seen_dims, 'the spy never fired: the per-record path made no F.linear calls at all'
    assert min(seen_dims) >= 2, (
        f'nn.Linear received a {min(seen_dims)}-D input; DeepSpeed ZeRO-3 backward cannot handle anything '
        f'flatter than 2-D (input dims seen: {seen_dims})')
