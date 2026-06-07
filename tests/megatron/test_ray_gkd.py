# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for Ray-based Megatron GKD fixes (no GPU / no distributed init).

Run:
    python tests/megatron/test_ray_gkd.py
or with pytest:
    pytest tests/megatron/test_ray_gkd.py

Covers the standalone-teacher / padding_free correctness fixes:
- _collate_teacher_outputs: padding_free seq-dim concat, off-by-one pad to the
  student's collated (SP-padded) length, and empty-placeholder dropping.
- _logp_gap: metric-key consistency across DP ranks (always a tensor in top-k mode,
  None only in full-vocab mode) — prevents the DP all_reduce shape-mismatch deadlock.
- build_opsd_teacher_data: OPSD teacher-prompt substitution.
"""
import torch

from swift.ray.megatron.loss.gkd import GKDLoss
from swift.ray.megatron.megatron_worker import MegatronWorker
from swift.rlhf_trainers.gkd_loss import TeacherOutput, build_opsd_teacher_data, extract_active

_collate = MegatronWorker._collate_teacher_outputs


def _topk_to(seq_len, k, fill):
    """A per-sample teacher topk tensor shaped [1, seq_len, k]."""
    return TeacherOutput(
        topk_logprobs=torch.full((1, seq_len, k), float(fill)),
        topk_indices=torch.zeros((1, seq_len, k), dtype=torch.long),
    )


def test_collate_padding_free_concat_and_offbyone_pad():
    """padding_free: concat per-sample along seq dim, then pad to target_seq_len.

    This is the off-by-one fix: the student collation pads the concatenated
    sequence to a multiple via get_padding_to (SP), so the teacher (built from raw
    per-sample lengths) can be a few tokens short and must be padded to match.
    """
    k = 4
    samples = [_topk_to(3, k, -1.0), _topk_to(5, k, -2.0)]  # raw total = 8
    target = 10  # student SP-padded length (8 -> 10)
    out = _collate(samples, device='cpu', padding_free=True, target_seq_len=target)
    assert out.topk_logprobs.shape == (1, target, k), out.topk_logprobs.shape
    assert out.topk_indices.shape == (1, target, k)
    # real tokens 0..7 keep their values; padded tail 8..9 is -inf / 0
    assert torch.isinf(out.topk_logprobs[0, 8:, :]).all()
    assert (out.topk_indices[0, 8:, :] == 0).all()
    assert not torch.isinf(out.topk_logprobs[0, :8, :]).any()


def test_collate_padding_free_offbyone_single_token():
    """The exact failure mode that deadlocked T3: total length odd, padded +1."""
    k = 2
    samples = [_topk_to(8277, k, -1.0)]  # one micro-batch sample, raw len 8277 (odd)
    out = _collate(samples, device='cpu', padding_free=True, target_seq_len=8278)
    assert out.topk_logprobs.shape == (1, 8278, k)
    assert torch.isinf(out.topk_logprobs[0, 8277:, :]).all()


def test_collate_padding_free_drops_empty_placeholder():
    """colocated path emits [1, full, k] for sample 0 and empty [0, ...] for the
    rest of a micro-batch; empties must be dropped before the seq-dim concat."""
    k = 3
    full = _topk_to(6, k, -1.0)
    empty = TeacherOutput(
        topk_logprobs=torch.full((0, 6, k), float('-inf')),
        topk_indices=torch.zeros((0, 6, k), dtype=torch.long),
    )
    out = _collate([full, empty], device='cpu', padding_free=True, target_seq_len=6)
    assert out.topk_logprobs.shape == (1, 6, k)


def test_collate_non_padding_free_stacks_on_batch_dim():
    """non padding_free: per-sample tensors padded to target then stacked on dim 0."""
    k = 4
    samples = [_topk_to(3, k, -1.0), _topk_to(5, k, -2.0)]
    out = _collate(samples, device='cpu', padding_free=False, target_seq_len=5)
    assert out.topk_logprobs.shape == (2, 5, k), out.topk_logprobs.shape
    # sample 0 padded from 3 -> 5
    assert torch.isinf(out.topk_logprobs[0, 3:, :]).all()
    assert not torch.isinf(out.topk_logprobs[1]).any()


def test_collate_opsd_keeps_teacher_length_not_student_target():
    """OPSD: teacher scores a different prompt, so its length differs from the
    student. The collation must KEEP the teacher length (ignore target_seq_len)
    and concat opsd_teacher_labels (extract_active aligns by mask, not position)."""
    k = 3
    t_total = 7  # teacher (opsd) sequence length
    full = TeacherOutput(
        topk_logprobs=torch.full((1, t_total, k), -1.0),
        topk_indices=torch.zeros((1, t_total, k), dtype=torch.long),
        opsd_teacher_labels=torch.full((1, t_total), 5, dtype=torch.long),
    )
    empty = TeacherOutput()  # padding_free placeholder for the rest of the micro-batch
    # target_seq_len is the *student* length (e.g. 12) — must be ignored for OPSD.
    out = _collate([full, empty], device='cpu', padding_free=True, target_seq_len=12)
    assert out.topk_logprobs.shape == (1, t_total, k), out.topk_logprobs.shape
    assert out.opsd_teacher_labels.shape == (1, t_total)
    assert (out.opsd_teacher_labels == 5).all()


def test_logp_gap_full_vocab_returns_none():
    """Full-vocab mode -> None on every rank (consistent: mode is global)."""
    s = torch.randn(1, 4, 8)
    labels = torch.tensor([[0, 1, 2, -100]])
    t = TeacherOutput(full_logits=torch.randn(1, 4, 8))
    assert GKDLoss._logp_gap(s, t, labels, temperature=1.0) is None


def test_logp_gap_topk_all_uncovered_returns_tensor_not_none():
    """top-k mode with no covered positions must return a tensor (0.0), NOT None,
    so the metric dict keeps the same keys on every DP rank (no all_reduce desync)."""
    k = 4
    s = torch.randn(1, 4, 8)
    labels = torch.tensor([[0, 1, 2, 3]])
    # all teacher topk -inf -> every position 'uncovered' -> mask.sum() == 0
    t = TeacherOutput(
        topk_logprobs=torch.full((1, 4, k), float('-inf')),
        topk_indices=torch.zeros((1, 4, k), dtype=torch.long),
    )
    gap = GKDLoss._logp_gap(s, t, labels, temperature=1.0)
    assert gap is not None and torch.is_tensor(gap)
    assert float(gap) == 0.0


def test_logp_gap_opsd_returns_none():
    """OPSD: teacher/student have different lengths, so the positional logp_gap is
    undefined and must be skipped (None) consistently on all DP ranks."""
    s = torch.randn(1, 4, 8)
    labels = torch.tensor([[0, 1, 2, 3]])
    t = TeacherOutput(  # teacher length 6 != student length 4
        topk_logprobs=torch.randn(1, 6, 3),
        topk_indices=torch.zeros((1, 6, 3), dtype=torch.long),
        opsd_teacher_labels=torch.tensor([[-100, -100, -100, -100, 1, 2]]),
    )
    assert GKDLoss._logp_gap(s, t, labels, temperature=1.0) is None


def test_build_opsd_teacher_data_substitutes_teacher_prompt():
    inputs = [{
        'messages': [{
            'role': 'user',
            'content': 'student-prompt'
        }, {
            'role': 'assistant',
            'content': 'resp'
        }],
        'teacher_prompt': 'teacher-prompt',
    }]
    out = build_opsd_teacher_data(inputs, strip_assistant=False)
    assert out is not None
    msgs = out[0]['messages']
    assert msgs[0]['content'] == 'teacher-prompt'  # last user msg replaced
    assert msgs[-1] == {'role': 'assistant', 'content': 'resp'}  # response kept
    assert 'teacher_prompt' not in out[0]  # field stripped from item


def test_build_opsd_teacher_data_strip_assistant():
    inputs = [{
        'messages': [{
            'role': 'user',
            'content': 'sp'
        }, {
            'role': 'assistant',
            'content': 'resp'
        }],
        'teacher_prompt': 'tp',
    }]
    out = build_opsd_teacher_data(inputs, strip_assistant=True)
    assert out[0]['messages'][-1]['role'] == 'user'  # trailing assistant removed
    assert out[0]['messages'][-1]['content'] == 'tp'


def test_build_opsd_teacher_data_none_without_teacher_prompt():
    inputs = [{'messages': [{'role': 'user', 'content': 'x'}]}]  # no teacher_prompt
    assert build_opsd_teacher_data(inputs, strip_assistant=False) is None


def test_extract_active_opsd_aligns_by_mask_across_lengths():
    """OPSD: teacher and student have different sequence lengths; extract_active
    selects response positions by their own masks and requires equal counts."""
    V, k = 8, 3
    # student: length 5, 2 response positions (indices 3,4)
    s_logits = torch.randn(1, 5, V)
    s_labels = torch.tensor([[-100, -100, -100, 1, 2]])
    # teacher (opsd): length 7, 2 response positions (indices 5,6) — same count
    t = TeacherOutput(
        topk_logprobs=torch.randn(1, 7, k),
        topk_indices=torch.zeros((1, 7, k), dtype=torch.long),
        opsd_teacher_labels=torch.tensor([[-100, -100, -100, -100, -100, 1, 2]]),
    )
    s_act, t_act, n = extract_active(s_logits, t, s_labels)
    assert int(n) == 2
    assert s_act.shape == (2, V)
    assert t_act.topk_logprobs.shape == (2, k)


def test_extract_active_opsd_count_mismatch_raises():
    s_logits = torch.randn(1, 5, 8)
    s_labels = torch.tensor([[-100, -100, -100, 1, 2]])  # 2 response tokens
    t = TeacherOutput(
        topk_logprobs=torch.randn(1, 6, 3),
        topk_indices=torch.zeros((1, 6, 3), dtype=torch.long),
        opsd_teacher_labels=torch.tensor([[-100, -100, -100, -100, -100, 9]]),  # 1 token
    )
    try:
        extract_active(s_logits, t, s_labels)
        raise AssertionError('expected an assertion on OPSD count mismatch')
    except AssertionError as e:
        assert 'OPSD' in str(e) or 'mismatch' in str(e) or 'count' in str(e).lower()


def test_example_yaml_config_contracts():
    """Config-contract regression for the standardized example yamls.

    - teacher replicas (standalone) must declare vllm_engine_kwargs.max_logprobs
      >= gkd_logits_topk, else vLLM rejects the prompt_logprobs request.
    - the standalone teacher group serves a real teacher checkpoint (model override).
    """
    import os
    import yaml
    base = os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'ray', 'gkd')

    cfg = yaml.safe_load(open(os.path.join(base, 'rollout_colocate_teacher_standalone.yaml')))
    topk = cfg['gkd_logits_topk']
    teacher = cfg['teacher']
    max_logprobs = teacher['vllm_engine_kwargs']['max_logprobs']
    assert max_logprobs >= topk, f'teacher max_logprobs {max_logprobs} < gkd_logits_topk {topk}'
    assert teacher.get('model'), 'standalone teacher group must override `model`'

    # colocate / separate examples keep max_length & max_completion_length consistent
    for name in ('rollout_colocate_teacher_colocate.yaml', 'rollout_separate_teacher_colocate.yaml'):
        c = yaml.safe_load(open(os.path.join(base, name)))
        assert c['max_length'] > 0 and c['max_completion_length'] > 0


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f'PASS {fn.__name__}')
        except Exception as e:  # noqa
            failed += 1
            import traceback
            print(f'FAIL {fn.__name__}: {e}')
            traceback.print_exc()
    print(f'\n{len(fns) - failed}/{len(fns)} passed')
    raise SystemExit(1 if failed else 0)
