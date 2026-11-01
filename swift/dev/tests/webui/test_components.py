# Copyright (c) ModelScope Contributors. All rights reserved.
"""Component bindings: the loss/scorer dual forms and channel contracts (S20 20g).

Real bindings over the real twinkle registries -- no fakes. The oracle for
each case is the registry itself (type identity, callable protocol) or a
hand-computed value (a two-scorer column matrix)."""
import pytest

from swift.dev.webui.component import ComponentError
from swift.dev.webui.components.data import ScoreStage, StreamingDriverComponent
from swift.dev.webui.components.resources import LossComponent, ScorerComponent
from swift.dev.webui.components.training import TrainStepStage
from swift.dev.webui.registry import REGISTRY, LOSS_OPTIONS

from ._fakes import MODEL_ID


class TestLossComponent:
    """kind picks the native registry; code wraps the user's formula."""

    def test_kind_constructs_the_native_loss(self):  # S20-51
        from twinkle.loss import GRPOLoss
        loss = LossComponent('l1', {'kind': 'grpo'}).build({})
        assert isinstance(loss, GRPOLoss)

    def test_unknown_kind_names_the_registry(self):  # S20-52
        from twinkle.loss import torch_loss_mapping
        with pytest.raises(ComponentError) as excinfo:
            LossComponent('l1', {'kind': 'nope'}).build({})
        message = str(excinfo.value)
        assert 'nope' in message
        assert any(name in message for name in torch_loss_mapping)

    def test_unknown_kwarg_is_rejected_not_swallowed(self):  # S20-53
        """GRPO's ``**kwargs`` would silently swallow a bogus knob; the
        constructor whitelist must refuse it by name."""
        with pytest.raises(ComponentError) as excinfo:
            LossComponent('l1', {'kind': 'grpo', 'loss_kwargs': {'bogus': 1}}).build({})
        assert 'bogus' in str(excinfo.value)

    def test_code_form_wraps_the_formula(self):  # S20-54
        from twinkle.loss import Loss
        code = 'def loss(**lanes):\n    return lanes\n'
        user_loss = LossComponent('l2', {'code': code}).build({})
        assert isinstance(user_loss, Loss)
        lanes = user_loss('IN', 'OUT', advantages=[1, 2])
        assert lanes == {'inputs': 'IN', 'outputs': 'OUT', 'advantages': [1, 2]}


class TestScorerComponent:
    """kind reaches the orms plugin registry; code is the user callable."""

    def test_kind_constructs_the_native_scorer(self):  # S20-55
        from twinkle.reward import MathReward
        scorer = ScorerComponent('s1', {'kind': 'math'}).build({})
        assert isinstance(scorer, MathReward)

    def test_unknown_kind_fails_loudly(self):
        with pytest.raises(ComponentError) as excinfo:
            ScorerComponent('s1', {'kind': 'nope'}).build({})
        assert 'nope' in str(excinfo.value)

    def test_code_form_yields_the_user_callable(self):  # S20-56
        code = 'def score(rows):\n    return [float(len(r["messages"])) for r in rows]\n'
        scorer = ScorerComponent('s2', {'code': code}).build({})
        rows = [{'messages': ['a']}, {'messages': ['a', 'b']}]
        assert scorer(rows) == [1.0, 2.0]


class TestStageContracts:
    """The LOSS and SCORER channels into their consuming stages."""

    def test_score_emits_one_column_per_scorer(self):  # S20-57
        rows = [{'messages': [{'role': 'user', 'content': 'q'}]} for _ in range(3)]
        scorers = [('math', lambda rs: [1.0] * len(rs)),
                   ('format', lambda rs: [0.5] * len(rs))]
        stage = ScoreStage('sc', {})
        out = stage.invoke({'scorers': scorers, 'samples': rows}, [])
        assert out['reward']['values'] == [[1.0, 0.5]] * 3
        assert out['reward']['scorer_ids'] == ['math', 'format']

    def test_short_scorer_column_fails_loudly(self):  # S20-57b
        rows = [{'messages': []} for _ in range(3)]
        stage = ScoreStage('sc', {})
        bad = [('math', lambda rs: [1.0, 2.0])]
        with pytest.raises(ComponentError) as excinfo:
            stage.invoke({'scorers': bad, 'samples': rows}, [])
        assert 'scores every row' in str(excinfo.value)

    def test_train_step_without_a_wired_loss_fails(self):  # S20-58
        train = TrainStepStage('tr', {'tokenizer': MODEL_ID})
        with pytest.raises(ComponentError) as excinfo:
            train.invoke({'model': object(), 'loss': None}, [])
        assert 'wired loss' in str(excinfo.value)

    def test_driver_requires_a_wired_prompt_stream(self):  # S20-58b
        driver = StreamingDriverComponent('dr', {})
        with pytest.raises(ComponentError) as excinfo:
            driver._build_driver({'engine': [('s', object())]})
        assert 'prompt stream' in str(excinfo.value)


class TestRegistrySurface:
    """The razor checks the registry itself kept after the v4 lift."""

    def test_loss_options_all_resolve_in_twinkle(self):
        from twinkle.loss import torch_loss_mapping
        assert all(name in torch_loss_mapping for name in LOSS_OPTIONS)

    def test_driver_dead_knobs_stay_gone(self):
        params = {p.name for p in REGISTRY['streaming_driver'].params}
        assert not {'batch_size', 'buffer_size', 'driver_config'} & params

    def test_scorer_kind_is_an_open_name(self):
        params = {p.name: p for p in REGISTRY['scorer'].params}
        assert params['kind'].options == ()
