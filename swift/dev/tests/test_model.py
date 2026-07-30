"""Unit tests for swift.dev.model after the twinkle-inheritance refactor.

The dev Model classes now inherit twinkle's concrete Model implementations
directly, so tests focus on:
- the shared data-format / loss / InputProcessor contracts (lightweight), and
- the inheritance wiring (TransformersModel/MegatronModel subclass twinkle,
  TrainableModel == twinkle base).

Constructing a real twinkle Model needs a downloadable model + process group, so
those paths are covered by skip-guarded integration tests rather than unit tests.
"""
import inspect
import pytest
import torch
import torch.nn as nn
from twinkle.utils import selective_log_softmax

from swift.dev.data_format import InputFeature, LossOutput, ModelOutput
from swift.dev.loss import CrossEntropyLoss, GRPOLoss, Loss
from swift.dev.model.base import TrainableModel
from swift.dev.model.megatron.model import MegatronModel
from swift.dev.model.strategy import AccelerateStrategy, NativeFSDPStrategy
from swift.dev.model.transformers_model import TransformersModel
from swift.dev.processor import InputProcessor
from swift.utils import to_device

# ======================================================================
# Data format — now twinkle's InputFeature/ModelOutput/LossOutput (dict-based)
# ======================================================================


class TestInputFeature:

    def test_dict_construction(self):
        feat = InputFeature(
            input_ids=torch.zeros(2, 4, dtype=torch.long),
            attention_mask=torch.ones(2, 4, dtype=torch.long),
            labels=torch.randint(0, 10, (2, 4)),
        )
        assert feat['input_ids'] is not None
        assert feat['attention_mask'] is not None
        assert feat['labels'] is not None

    def test_plain_dict(self):
        feat = {'input_ids': torch.zeros(2, 4, dtype=torch.long)}
        assert 'input_ids' in feat
        assert feat['input_ids'].shape == (2, 4)

    def test_get_method(self):
        feat = InputFeature(input_ids=torch.zeros(2, 4, dtype=torch.long))
        assert feat.get('labels') is None
        assert feat.get('input_ids') is not None


class TestModelOutput:

    def test_creation_with_logps(self):
        out = ModelOutput(
            logits=torch.randn(2, 4, 10),
            loss=torch.tensor(1.0),
            logps=torch.randn(2, 4),
        )
        assert out['logits'] is not None
        assert out['loss'] is not None
        assert out['logps'] is not None

    def test_optional_fields(self):
        out = ModelOutput(logps=torch.randn(2, 4))
        assert out['logps'] is not None
        assert out.get('logits') is None


class TestLossOutput:

    def test_creation(self):
        out = LossOutput(loss=torch.tensor(0.5), num_tokens=10)
        assert out['loss'].item() == pytest.approx(0.5)
        assert out['num_tokens'] == 10

    def test_dict_access(self):
        out = LossOutput(loss=torch.tensor(1.0), num_tokens=128)
        assert 'loss' in out
        assert 'num_tokens' in out


class TestToDevice:

    def test_moves_tensors(self):
        data = {
            'input_ids': torch.zeros(2, 4, dtype=torch.long),
            'attention_mask': torch.ones(2, 4, dtype=torch.long),
            'length': 4,
        }
        result = to_device(data, 'cpu')
        # swift.utils.to_device returns a new mapping (not in-place); assert the moved tensors
        # land on the target device and non-tensor values are carried through unchanged.
        assert result['input_ids'].device == torch.device('cpu')
        assert result['attention_mask'].device == torch.device('cpu')
        assert result['length'] == 4


# ======================================================================
# Loss
# ======================================================================


class TestLossBase:

    def test_flags(self):
        loss = Loss()
        assert loss.require_logits is False
        assert loss.require_entropy is False
        assert loss.require_logps is True

    def test_call_signature(self):
        sig = inspect.signature(Loss.__call__)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'inputs' in params
        assert 'outputs' in params
        assert 'kwargs' in params


class TestCrossEntropyLoss:

    def test_flags(self):
        ce = CrossEntropyLoss()
        assert ce.require_logits is False
        assert ce.require_logps is True
        assert ce.require_entropy is False

    def test_from_logps(self):
        ce = CrossEntropyLoss()
        B, T = 2, 4
        labels = torch.randint(0, 10, (B, T))
        labels[0, -1] = -100
        logps = torch.randn(B, T)

        inputs = InputFeature(input_ids=torch.zeros(B, T, dtype=torch.long), labels=labels)
        outputs = ModelOutput(logps=logps)

        result = ce(inputs, outputs)
        assert isinstance(result, dict)
        assert result['loss'].isfinite()

    def test_from_logits_fallback(self):
        ce = CrossEntropyLoss()
        B, T, V = 2, 4, 10
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))

        inputs = InputFeature(input_ids=torch.zeros(B, T, dtype=torch.long), labels=labels)
        outputs = ModelOutput(logits=logits)

        result = ce(inputs, outputs)
        assert result['loss'].isfinite()


class TestGRPOLoss:

    def test_flags(self):
        grpo = GRPOLoss(epsilon=0.2, entropy_coef=0.0)
        assert grpo.require_logits is False
        assert grpo.require_logps is True
        assert grpo.require_entropy is False

    def test_entropy_flag(self):
        grpo_ent = GRPOLoss(epsilon=0.2, entropy_coef=0.01)
        assert grpo_ent.require_entropy is True

    def test_forward_with_advantages(self):
        grpo = GRPOLoss(epsilon=0.2, entropy_coef=0.0)
        B, T = 2, 8
        logps = torch.randn(B, T)
        old_logps = logps.detach() + torch.randn(B, T) * 0.1
        advantages = torch.randn(B)

        labels = torch.randint(0, 100, (B, T))
        inputs = InputFeature(input_ids=torch.zeros(B, T, dtype=torch.long), labels=labels)
        outputs = ModelOutput(logps=logps)

        result = grpo(inputs, outputs, advantages=advantages, old_logps=old_logps)
        assert isinstance(result, dict)
        assert result['loss'].isfinite()

    def test_no_advantages_returns_zero(self):
        grpo = GRPOLoss(epsilon=0.2)
        B, T = 2, 8
        logps = torch.randn(B, T)
        labels = torch.randint(0, 100, (B, T))
        inputs = InputFeature(input_ids=torch.zeros(B, T, dtype=torch.long), labels=labels)
        outputs = ModelOutput(logps=logps)

        result = grpo(inputs, outputs)
        assert result['loss'].item() == 0.0


# ======================================================================
# Strategy — now re-exported from twinkle (dev no longer owns a Strategy layer)
# ======================================================================


class TestStrategyReexport:

    def test_reexported_from_twinkle(self):
        from twinkle.model.transformers.strategy import AccelerateStrategy as TwAccelerate
        from twinkle.model.transformers.strategy import NativeFSDPStrategy as TwNativeFSDP
        assert AccelerateStrategy is TwAccelerate
        assert NativeFSDPStrategy is TwNativeFSDP

    def test_deepspeed_strategy_absent(self):
        """DeepSpeedStrategy is a future item and must not be exported yet."""
        import swift.dev.model.strategy as strategy_mod
        assert not hasattr(strategy_mod, 'DeepSpeedStrategy')


# ======================================================================
# InputProcessor — swift extension over twinkle (template collate hooks)
# ======================================================================


class TestInputProcessor:

    def test_subclasses_twinkle(self):
        from twinkle.processor import InputProcessor as TwinkleInputProcessor
        assert issubclass(InputProcessor, TwinkleInputProcessor)

    def test_prepare_inputs_numpy(self):
        proc = InputProcessor()
        import numpy as np
        inp = InputFeature(input_ids=np.array([1, 2, 3, 4]))
        result = proc.prepare_inputs([inp])[0]
        assert isinstance(result['input_ids'], torch.Tensor)
        assert result['input_ids'].dim() == 2

    def test_prepare_inputs_list(self):
        proc = InputProcessor()
        inp = InputFeature(input_ids=[1, 2, 3, 4])
        result = proc.prepare_inputs([inp])[0]
        assert isinstance(result['input_ids'], torch.Tensor)
        assert result['input_ids'].dim() == 2

    def test_to_transformers_dict_filters(self):
        proc = InputProcessor()
        inp = InputFeature(
            input_ids=torch.zeros(1, 4, dtype=torch.long),
            labels=torch.zeros(1, 4, dtype=torch.long),
        )
        inp['extra_field'] = torch.ones(1, 4)
        result = proc.to_transformers_dict([inp])[0]
        assert 'input_ids' in result
        assert 'labels' in result
        assert 'extra_field' not in result

    def test_collate_fn_callback(self):
        called = [False]

        def my_collate(inputs):
            called[0] = True
            return inputs[0]

        proc = InputProcessor(collate_fn=my_collate)
        inp = InputFeature(input_ids=torch.zeros(1, 4, dtype=torch.long))
        proc.collate_fn([inp])
        assert called[0] is True

    def test_full_pipeline_single_sample(self):
        proc = InputProcessor()
        inp = InputFeature(
            input_ids=torch.randint(0, 100, (1, 8)),
            attention_mask=torch.ones(1, 8, dtype=torch.long),
            labels=torch.randint(0, 100, (1, 8)),
            position_ids=torch.arange(8).unsqueeze(0),
        )
        result = proc([inp])
        assert isinstance(result, dict)
        assert 'input_ids' in result

    def test_template_wiring(self):
        proc = InputProcessor()
        assert proc._template is None
        proc._template = 'test'
        assert proc._template == 'test'

    def test_is_packed_position_ids(self):
        single = torch.arange(6).unsqueeze(0)
        assert InputProcessor._is_packed_position_ids(single) is False
        packed = torch.tensor([[0, 1, 2, 0, 1, 2]])
        assert InputProcessor._is_packed_position_ids(packed) is True

    def test_unpack_by_position_ids(self):
        position_ids = torch.tensor([[0, 1, 2, 0, 1]])
        tensor = torch.tensor([[10, 20, 30, 40, 50]])
        unpacked = InputProcessor._unpack_by_position_ids(position_ids, tensor, padding_values=[0])
        assert unpacked[0].shape == (2, 3)
        assert unpacked[0][0, 0] == 10
        assert unpacked[0][0, 2] == 30
        assert unpacked[0][1, 0] == 40
        assert unpacked[0][1, 1] == 50
        assert unpacked[0][1, 2] == 0


# ======================================================================
# Model contract & inheritance wiring
# ======================================================================


class TestTrainableModelContract:

    def test_is_twinkle_base(self):
        """TrainableModel is twinkle's TwinkleModel (design: twinkle is the contract)."""
        from twinkle.model.base import TwinkleModel
        assert TrainableModel is TwinkleModel

    def test_contract_methods_present(self):
        """The twinkle contract exposes the atomic training method set."""
        expected = [
            'forward',
            'forward_only',
            'forward_backward',
            'backward',
            'calculate_loss',
            'clip_grad_norm',
            'clip_grad_and_step',
            'step',
            'zero_grad',
            'lr_step',
            'set_loss',
            'set_optimizer',
            'set_lr_scheduler',
            'set_template',
            'set_processor',
            'save',
            'load',
            'resume_from_checkpoint',
            'get_state_dict',
        ]
        for name in expected:
            assert hasattr(TrainableModel, name), f"Missing contract method: {name}"


class TestTransformersModelInheritance:

    def test_inherits_twinkle(self):
        from twinkle.model.transformers import TransformersModel as TwinkleTransformersModel
        assert issubclass(TransformersModel, TwinkleTransformersModel)
        assert issubclass(TransformersModel, TrainableModel)

    def test_inherits_forward_backward(self):
        """forward/backward/save come from twinkle, not re-implemented in dev."""
        assert TransformersModel.forward is not None
        assert TransformersModel.forward_backward is not None
        assert TransformersModel.save is not None


class TestMegatronModelInheritance:

    def test_inherits_twinkle(self):
        from twinkle.model.megatron import MegatronModel as TwinkleMegatronModel
        assert issubclass(MegatronModel, TwinkleMegatronModel)
        assert issubclass(MegatronModel, TrainableModel)


# ======================================================================
# selective_log_softmax (twinkle util used by the forward path)
# ======================================================================


class TestSelectiveLogSoftmax:

    def test_output_shape(self):
        B, T, V = 2, 4, 10
        logits = torch.randn(B, T, V)
        index = torch.randint(0, V, (B, T))
        result = selective_log_softmax(logits, index)
        assert result.shape == (B, T)

    def test_ignore_positions(self):
        B, T, V = 2, 6, 10
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        labels[0, -1] = -100
        masked_labels = labels.clone()
        masked_labels[masked_labels == -100] = 0
        result = selective_log_softmax(logits, masked_labels)
        assert result.shape == (B, T)
        assert result.isfinite().all()

    def test_values_are_log_probabilities(self):
        B, T, V = 2, 4, 100
        logits = torch.randn(B, T, V)
        index = torch.randint(0, V, (B, T))
        result = selective_log_softmax(logits, index)
        assert (result <= 0).all()


# ======================================================================
# Contract consistency — swift data formats ARE twinkle's now
# ======================================================================


def test_contract_consistency_with_twinkle():
    """swift data formats are re-exports of twinkle's (identity)."""
    from twinkle.data_format.input_feature import InputFeature as TwinkleInputFeature
    from twinkle.data_format.output import LossOutput as TwinkleLossOutput
    from twinkle.data_format.output import ModelOutput as TwinkleModelOutput
    assert InputFeature is TwinkleInputFeature
    assert ModelOutput is TwinkleModelOutput
    assert LossOutput is TwinkleLossOutput
