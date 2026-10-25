"""S8a — the preference-batch encoder: interleaving, prefix stripping, fail-loudly (L1).

``PreferenceBatchEncoder.__call__`` turns raw preference rows into ``forward_backward`` args. The
paired family (dpo/cpo/orpo/simpo) interleaves ``[chosen_1, rejected_1, chosen_2, rejected_2, ...]``
-- the flat layout the DPO family's chosen/rejected split relies on -- and the template's
``chosen_*``/``rejected_*`` keys are stripped back into standalone features. RM (seq_cls) routes
``task='seq_cls'`` and carries no labels (RewardLoss scores the head's logits, not logps). A stub
template stands in for the encoding.

Maps to ut_plan S8-04/05/06.
"""
import pytest


class _PairedTemplate:
    """Encodes a row into chosen_/rejected_ prefixed features (the template's rlhf-mode shape)."""

    def encode(self, row):
        # .get(): a row missing one side must reach the ENCODER's guard, not crash the stub first.
        encoded = {}
        if 'chosen_ids' in row:
            encoded['chosen_input_ids'] = row['chosen_ids']
            encoded['chosen_labels'] = list(row['chosen_ids'])
        if 'rejected_ids' in row:
            encoded['rejected_input_ids'] = row['rejected_ids']
            encoded['rejected_labels'] = list(row['rejected_ids'])
        return encoded


def _encoder(rl_type='dpo'):
    from swift.dev.recipe.rlhf.encoder import PreferenceBatchEncoder

    return PreferenceBatchEncoder(model=None, template=_PairedTemplate(), rl_type=rl_type,
                                  calculate_kl=False, gradient_accumulation_steps=1)


def test_paired_interleave_order_and_prefix_strip():
    """S8-04: rows encode to [chosen_1, rejected_1, chosen_2, rejected_2, ...] with the
    chosen_/rejected_ prefixes stripped into standalone input_ids/labels features."""
    encoder = _encoder('dpo')
    kwargs = encoder([
        {'chosen_ids': [1, 2], 'rejected_ids': [3, 4]},
        {'chosen_ids': [5], 'rejected_ids': [6, 7]},
    ], owns_ga=True)
    features = kwargs['inputs']
    assert [f['input_ids'] for f in features] == [[1, 2], [3, 4], [5], [6, 7]]
    assert [f['labels'] for f in features] == [[1, 2], [3, 4], [5], [6, 7]]
    assert 'chosen_input_ids' not in features[0]  # prefixes stripped, not merged


def test_rm_routes_seq_cls_without_labels():
    """S8-05: the rm family forwards the head -- task='seq_cls' and no per-token labels (a template
    in rm mode emits no labels; if it did, RewardLoss would ignore them -- the encoder never adds
    them back)."""
    class _RmTemplate:

        def encode(self, row):
            return {'chosen_input_ids': row['chosen_ids'], 'rejected_input_ids': row['rejected_ids']}

    from swift.dev.recipe.rlhf.encoder import PreferenceBatchEncoder

    encoder = PreferenceBatchEncoder(model=None, template=_RmTemplate(), rl_type='rm',
                                     calculate_kl=False, gradient_accumulation_steps=1)
    kwargs = encoder([{'chosen_ids': [1], 'rejected_ids': [2]}], owns_ga=True)
    assert kwargs['task'] == 'seq_cls'
    assert all('labels' not in f for f in kwargs['inputs'])
    assert [f['input_ids'] for f in kwargs['inputs']] == [[1], [2]]


def test_missing_side_fails_loudly():
    """S8-06: a row that encodes only one side is a hard error -- a silent half-row would shift the
    interleave and pair the wrong completions."""
    encoder = _encoder('dpo')
    with pytest.raises(ValueError, match='rejected'):
        encoder([{'chosen_ids': [1]}], owns_ga=True)  # stub omits rejected_* -> template got no side


def test_missing_chosen_side_fails_loudly():
    encoder = _encoder('cpo')
    with pytest.raises(ValueError, match='chosen'):
        encoder([{'rejected_ids': [2]}], owns_ga=True)


def test_ga_steps_forwarded_when_not_owned():
    """When the caller does not own gradient accumulation (transformers loop), the encoder forwards
    the ga steps the forward must simulate."""
    encoder = _encoder('orpo')
    kwargs = encoder([{'chosen_ids': [1], 'rejected_ids': [2]}], owns_ga=False)
    assert kwargs['gradient_accumulation_steps'] == 1
    kwargs_owned = encoder([{'chosen_ids': [1], 'rejected_ids': [2]}], owns_ga=True)
    assert 'gradient_accumulation_steps' not in kwargs_owned
