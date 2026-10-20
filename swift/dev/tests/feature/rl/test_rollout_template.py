"""S8b — the dev Template's next-token shift contract on the rollout/inference path (L1).

ut_plan S8-25 (vllm-mode encode applies no shift) and S8-31 (rollout SHIFTED_KEY == template SHIFTED_KEY).

S8-31 is N/A in dev2: the marker's SECOND declaration lived in the deleted ``swift.dev.rollout`` package,
and ``twinkle_agentic`` keys trainable tokens off ``labels != -100`` (its ledger's turn_token_spans), not
off ``_labels_shifted`` -- so there is no longer a cross-module literal to keep in sync. What remains is
the dev-side contract S8-25 pins: DevMixin shifts labels to next-token alignment ONLY in training mode
(never on the vllm/inference rollout path), only for shift-eligible task types, and idempotently.

The shift lives in DevMixin.encode, which calls super().encode (the legacy Template). A stub base mixed
in via shifted_template_class isolates the gating without loading a tokenizer.
"""


class _StubLegacyTemplate:
    """Stands in for a legacy Template: encode returns aligned labels; is_training/task_type drive the gate."""

    def __init__(self, is_training, task_type='causal_lm', pre_shifted=False):
        self.is_training = is_training
        self.task_type = task_type
        self._pre_shifted = pre_shifted

    def encode(self, inputs, return_template_inputs=False, return_length=False):
        encoded = {'input_ids': [1, 2, 3], 'labels': [-100, 2, 3]}
        if self._pre_shifted:
            from swift.dev.template.template import DevMixin
            encoded[DevMixin.SHIFTED_KEY] = True
        return encoded


def _shifted_stub(**kwargs):
    from swift.dev.template.template import shifted_template_class

    return shifted_template_class(_StubLegacyTemplate)(**kwargs)


def test_training_mode_shifts_labels_to_next_token():
    """S8-25 (training half): in training mode the aligned labels become next-token (out[i]=labels[i+1],
    last=-100) and the SHIFTED_KEY marker is stamped so a second encode cannot shift twice."""
    from swift.dev.template.template import DevMixin

    encoded = _shifted_stub(is_training=True).encode({'input_ids': [1, 2, 3]})
    assert encoded['labels'] == [2, 3, -100]  # [-100,2,3] shifted left by one, tail masked
    assert encoded[DevMixin.SHIFTED_KEY] is True


def test_vllm_mode_applies_no_shift():
    """S8-25 (rollout half): on the inference/rollout path (is_training False) labels are emitted aligned
    -- NO next-token shift and NO marker -- because rollout inputs are inference, not training targets."""
    from swift.dev.template.template import DevMixin

    encoded = _shifted_stub(is_training=False).encode({'input_ids': [1, 2, 3]})
    assert encoded['labels'] == [-100, 2, 3]  # untouched
    assert DevMixin.SHIFTED_KEY not in encoded


def test_no_shift_task_types_are_exempt_even_in_training():
    """S8-25: labels are not per-token targets on embedding/reranker/seq_cls, so the shift must not fire
    even in training mode (a seq_cls score head would be corrupted by a next-token roll)."""
    from swift.dev.template.template import DevMixin

    for task_type in ('seq_cls', 'embedding', 'reranker'):
        encoded = _shifted_stub(is_training=True, task_type=task_type).encode({'input_ids': [1, 2, 3]})
        assert encoded['labels'] == [-100, 2, 3], f'{task_type} must not shift'
        assert DevMixin.SHIFTED_KEY not in encoded


def test_shift_is_idempotent_via_marker():
    """S8-25: an already-shifted feature (SHIFTED_KEY set) is never shifted a second time."""
    from swift.dev.template.template import DevMixin

    encoded = _shifted_stub(is_training=True, pre_shifted=True).encode({'input_ids': [1, 2, 3]})
    assert encoded['labels'] == [-100, 2, 3]  # the stub's aligned labels pass through unshifted
    assert encoded[DevMixin.SHIFTED_KEY] is True


def test_preference_label_keys_are_all_shifted():
    """S8-25: the preference path emits chosen_/rejected_ labels; EVERY label key must shift (else dpo/kto
    feed aligned labels), with the matching loss_scale rolled in step."""

    class _PreferenceStub(_StubLegacyTemplate):

        def encode(self, inputs, return_template_inputs=False, return_length=False):
            return {
                'chosen_input_ids': [1, 2, 3],
                'chosen_labels': [-100, 2, 3],
                'chosen_loss_scale': [0.0, 1.0, 1.0],
                'rejected_input_ids': [4, 5, 6],
                'rejected_labels': [-100, 5, 6],
            }

    from swift.dev.template.template import shifted_template_class

    encoded = shifted_template_class(_PreferenceStub)(is_training=True).encode({'input_ids': [1]})
    assert encoded['chosen_labels'] == [2, 3, -100]
    assert encoded['rejected_labels'] == [5, 6, -100]
    assert encoded['chosen_loss_scale'] == [1.0, 1.0, 0.0]  # rolled in step with the labels
