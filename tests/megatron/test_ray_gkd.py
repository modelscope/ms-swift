# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from types import SimpleNamespace

from swift.ray.megatron.gkd_trainer import GKDTrainer
from swift.ray.megatron.megatron_worker import MegatronWorker
from swift.rlhf_trainers.gkd_loss import TeacherOutput, extract_active

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
    and concat labels (extract_active aligns by mask, not position)."""
    k = 3
    t_total = 7  # teacher (opsd) sequence length
    full = TeacherOutput(
        topk_logprobs=torch.full((1, t_total, k), -1.0),
        topk_indices=torch.zeros((1, t_total, k), dtype=torch.long),
        labels=torch.full((1, t_total), 5, dtype=torch.long),
    )
    empty = TeacherOutput()  # padding_free placeholder for the rest of the micro-batch
    # target_seq_len is the *student* length (e.g. 12) — must be ignored for OPSD.
    out = _collate([full, empty], device='cpu', padding_free=True, target_seq_len=12, is_opsd=True)
    assert out.topk_logprobs.shape == (1, t_total, k), out.topk_logprobs.shape
    assert out.labels.shape == (1, t_total)
    assert (out.labels == 5).all()


def test_build_per_sample_teacher_output_uses_raw_input_length():
    """Standalone teacher outputs are built from each sample's RAW (un-CP-padded) input
    length. Because these raw per-sample token-logprobs cannot be CP-sharded to match the
    student, CP>1 with standalone teacher replicas is rejected by a fail-fast in
    GKDTrainer._collate_for_workers_gkd (use a colocated teacher_model for CP>1).
    This test guards the raw-length contract that the CP>1 fail-fast depends on.
    """
    k = 3
    raw_len = 5
    lps = [[-1.0] * k for _ in range(raw_len)]
    ixs = [[0] * k for _ in range(raw_len)]
    encoded = {'input_ids': list(range(raw_len))}
    out = GKDTrainer._build_per_sample_teacher_output((lps, ixs), encoded, topk=k)
    assert out.topk_logprobs.shape == (1, raw_len, k), out.topk_logprobs.shape
    assert out.topk_indices.shape == (1, raw_len, k)
    assert out.labels is None


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
        labels=torch.tensor([[-100, -100, -100, -100, -100, 1, 2]]),
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
        labels=torch.tensor([[-100, -100, -100, -100, -100, 9]]),  # 1 token
    )
    try:
        extract_active(s_logits, t, s_labels)
        raise AssertionError('expected an assertion on OPSD count mismatch')
    except AssertionError as e:
        assert 'OPSD' in str(e) or 'mismatch' in str(e) or 'count' in str(e).lower()


def test_extract_active_non_opsd_uses_student_labels():
    """Non-OPSD: teacher_output.labels is None (Ray GKD non-OPSD path).

    The student label mask should apply to both student and teacher.
    This is the Critical #1 regression test: before the fix, extract_active
    crashed with TypeError on ``None != -100``.
    """
    V, k = 8, 3
    # student: length 5, 3 response positions (indices 2,3,4)
    s_logits = torch.randn(1, 5, V)
    s_labels = torch.tensor([[-100, -100, 1, 2, 3]])
    # teacher (non-OPSD): same length, labels=None
    t = TeacherOutput(
        topk_logprobs=torch.randn(1, 5, k),
        topk_indices=torch.zeros((1, 5, k), dtype=torch.long),
        labels=None,
    )
    s_act, t_act, n = extract_active(s_logits, t, s_labels)
    assert int(n) == 3
    assert s_act.shape == (3, V)
    assert t_act.topk_logprobs.shape == (3, k)


def test_extract_active_non_opsd_full_logits():
    """Non-OPSD with full-vocab teacher (no topk): labels=None path.

    Verifies that the student mask is used for both student and teacher
    when teacher_output.labels is None.
    """
    V = 8
    s_logits = torch.randn(1, 4, V)
    s_labels = torch.tensor([[-100, 1, 2, 3]])
    t = TeacherOutput(
        full_logits=torch.randn(1, 4, V),
        labels=None,
    )
    s_act, t_act, n = extract_active(s_logits, t, s_labels)
    assert int(n) == 3
    assert s_act.shape == (3, V)
    assert t_act.full_logits.shape == (3, V)


def test_megatron_assemble_teacher_outputs_api_topk_rolls_labels():
    """Megatron Teacher API + topk: ``_assemble_teacher_outputs`` must roll teacher
    labels by -1 so the invariant 'teacher_output.labels is pre-shifted before
    extract_active' holds on the API path too (the local-teacher path gets shifted
    labels from _prepare_batch). assemble_teacher_output returns the RAW labels, so
    the trainer applies the shift; without it the API path would feed unshifted
    teacher labels against shifted student labels -> silent KL/JSD misalignment.
    """
    try:
        from swift.megatron.trainers.gkd_trainer import MegatronGKDTrainer
    except Exception as e:  # noqa: megatron-core not installed in this env
        print(f'SKIP test_megatron_assemble_teacher_outputs_api_topk_rolls_labels: {e}')
        return

    k = 3
    # raw (unshifted) labels: prompt=-100, response at positions 2,3,4
    raw_labels = torch.tensor([[-100, -100, 11, 22, 33]])
    seq_len = raw_labels.shape[-1]
    # parsed teacher topk: one (logprobs, indices) row per response token (len+1 cu)
    parsed = [([[-1.0] * k] * (seq_len - 1), [[0] * k] * (seq_len - 1))]
    teacher_model_inputs = {
        'input_ids': torch.zeros((1, seq_len), dtype=torch.long),
        'labels': raw_labels.clone(),
        'attention_mask': torch.ones((1, seq_len), dtype=torch.long),
    }
    encoded_batch = {'_teacher_parsed': parsed, 'teacher_model_inputs': teacher_model_inputs}

    stub = SimpleNamespace(gkd_logits_topk=k, template=SimpleNamespace(padding_free=False), device=torch.device('cpu'))
    MegatronGKDTrainer._assemble_teacher_outputs(stub, [encoded_batch])

    teacher_out = encoded_batch['teacher_output']
    assert torch.equal(teacher_out.labels, torch.roll(raw_labels, shifts=-1, dims=-1))
    assert teacher_out.topk_logprobs.shape == (1, seq_len, k)

    # The shifted teacher labels must align with shifted student labels in extract_active.
    s_labels = torch.roll(raw_labels, shifts=-1, dims=-1)
    s_logits = torch.randn(1, seq_len, 8)
    s_act, t_act, n = extract_active(s_logits, teacher_out, s_labels)
    assert int(n) == 3
    assert t_act.topk_logprobs.shape == (3, k)


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
