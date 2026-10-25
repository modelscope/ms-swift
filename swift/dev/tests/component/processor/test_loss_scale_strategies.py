"""S18g — loss_scale strategy semantics: base strategies, overrides, family splits (L1).

The loss_scale subsystem decides which rendered parts contribute to the loss. The BASE strategies
(``default`` = assistant responses only, ``last_round`` = the last response only, ``all`` =
everything) are pinned on a 2-round conversation; a message's own ``loss`` field overrides the
strategy (loss=0 masks that response); and the agent families (qwen keeps think+answer whole at 1.0;
react splits into thought at 1.0 / action at 2.0) are pinned with their literal split outputs.

The existing processor tests only carry loss_scale as a field -- no strategy-truth tests existed.

Maps to ut_plan S18-36..39.
"""
import pytest

from swift.dev.template.loss_scale.base import ContextType
from swift.dev.template.loss_scale.mapping import get_loss_scale, loss_scale_map

OPEN, CLOSE = chr(60) + 'think' + chr(62), chr(60) + '/think' + chr(62)

CTX = ['Q1', 'A1', 'Q2', 'A2']
TYPES = [ContextType.OTHER, ContextType.RESPONSE, ContextType.OTHER, ContextType.RESPONSE]
MSGS = [
    {'role': 'user', 'content': 'Q1'},
    {'role': 'assistant', 'content': 'A1'},
    {'role': 'user', 'content': 'Q2'},
    {'role': 'assistant', 'content': 'A2'},
]


def _scales(strategy, messages=MSGS):
    _, weights = get_loss_scale(strategy)(CTX, TYPES, messages)
    return weights


class TestBaseStrategies:
    """S18-36/39: the three ALL_BASE_STRATEGY semantics on a 2-round conversation."""

    def test_default_trains_responses_only(self):
        assert _scales('default') == [0.0, 1.0, 0.0, 1.0]

    def test_last_round_trains_only_the_last_response(self):
        assert _scales('last_round') == [0.0, 0.0, 0.0, 1.0]

    def test_all_trains_everything(self):
        assert _scales('all') == [1.0, 1.0, 1.0, 1.0]

    def test_base_strategy_is_binary(self):
        """S18-36: the base strategies emit 0/1 weights carried in labels (-100 masks), so no
        per-token loss_scale field is needed."""
        assert get_loss_scale('default').is_binary is True


class TestMessageOverride:
    """A per-message ``loss`` field overrides the strategy: loss=0 forces masking even where the
    strategy would train (and loss=1 would force training even under last_round)."""

    def test_loss_zero_masks_that_response(self):
        messages = [dict(m) for m in MSGS]
        messages[1]['loss'] = 0  # mask A1 under 'default'
        assert _scales('default', messages) == [0.0, 0.0, 0.0, 1.0]

    def test_loss_one_forces_training_under_last_round(self):
        messages = [dict(m) for m in MSGS]
        messages[1]['loss'] = 1  # train A1 even though last_round would mask it
        assert _scales('last_round', messages) == [0.0, 1.0, 0.0, 1.0]


class TestFamilyStrategies:
    """S18-37/38: the registered agent families and their context splits."""

    def test_registered_families_all_resolve(self):
        """S18-38: every key in loss_scale_map resolves to a distinct LossScale subclass; the three
        base strategy names are NOT map keys (they parameterize the base class)."""
        assert set(loss_scale_map) == {'agentflan', 'alpha_umi', 'base', 'hermes',
                                       'ignore_empty_think', 'qwen', 'react'}
        classes = {get_loss_scale(name) for name in loss_scale_map}
        assert len(classes) == len(loss_scale_map), 'each key must resolve to its own instance'

    def test_qwen_keeps_think_and_answer_together(self):
        """The qwen strategy does NOT split a think-block response: the whole context stays one
        part at weight 1.0 (the model is trained to produce thought+answer jointly)."""
        contexts, weights = get_loss_scale('qwen').get_loss_scale(
            OPEN + 'reasoning' + CLOSE + 'answer', query='Q')
        assert contexts == [OPEN + 'reasoning' + CLOSE + 'answer']
        assert weights == [1.0]

    def test_react_splits_thought_from_action_at_double_weight(self):
        """The react strategy splits a ReAct response into the thought (weight 1.0) and the action
        parts (weight 2.0 -- the action is what the loss emphasizes)."""
        contexts, weights = get_loss_scale('react').get_loss_scale(
            OPEN + 'reasoning' + CLOSE + '\nAction: x', query='Q')
        assert contexts == [OPEN + 'reasoning' + CLOSE + '\n', 'Action:', ' x']
        assert weights == [1.0, 2.0, 2.0]

    def test_qwen_is_not_binary(self):
        """S18-36: qwen may emit fractional weights, so is_binary is not True (a per-token
        loss_scale field is emitted instead of label masking)."""
        assert get_loss_scale('qwen').is_binary is not True

    def test_unknown_strategy_name_rejected(self):
        """An unknown name fails loudly at resolution, never silently falls back to default."""
        with pytest.raises(KeyError):
            get_loss_scale('bogus')
