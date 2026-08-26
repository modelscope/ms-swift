"""Multi-Token Prediction: joint training, and the three places it used to silently do nothing.

Why this exists. Before this feature, setting ``mtp_num_layers`` on an RL/twinkle run produced MTP
layers that were built, loaded, exported AND weight-synced -- and never received a single gradient,
with no error and no mtp_loss to show it. Three independent breaks caused that, and each has a test
here because every one of them fails *quietly*:

1. **The training gate.** mcore-bridge ran the MTP block only ``if labels is not None``, but twinkle
   (like every RL trainer) pops ``labels`` off the batch and computes log-probs from logits, so the
   model never saw them. Fixed by a separate ``mtp_labels`` channel; tested via
   ``MegatronModel._mtp_training_enabled``, which decides when to use it.

2. **The forward_only waste.** Megatron's ``forward_only`` schedule skips the backward but neither
   switches the module to eval nor disables autograd, so ``self.training`` and
   ``torch.is_grad_enabled()`` are both True on an old/reference log-prob pass. Nothing inside the
   model can tell it apart from a training pass, which is why the gate must be told.

3. **The vLLM drafter.** The target model's ``load_weights`` deliberately drops MTP keys and vLLM
   exposes no way to update the drafter, so rollout kept the draft head it started with.

The config rules are errors rather than warnings on purpose: a knob that reads as "MTP is on" while
doing nothing is worse than a failed launch.
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace


# === The config contract (pure function of the Configs, no I/O) ===


def _configs(**model_overrides):
    from swift.dev.config import DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig, TrainConfig
    return {
        'model_config': ModelConfig(model='dummy', **model_overrides),
        'template_config': TemplateConfig(template='qwen2_5'),
        'dataset_config': DatasetConfig(dataset=['dummy']),
        'train_config': TrainConfig(),
        'distributed_config': DistributedConfig(backend='megatron', nproc_per_node=1),
    }


def _validate(**model_overrides):
    from swift.dev.config import validate_configs
    validate_configs(**_configs(**model_overrides))


def test_mtp_off_by_default_is_accepted():
    _validate()


def test_loading_mtp_without_training_it_is_a_valid_combination():
    """The "carry the heads to the exported checkpoint" case, which serving then uses as a drafter."""
    _validate(mtp_num_layers=1)
    _validate(mtp_num_layers=1, mtp_freeze=True)


def test_joint_training_is_accepted():
    _validate(mtp_num_layers=1, enable_mtp_training=True, mtp_loss_scaling_factor=0.2)
    _validate(mtp_num_layers=2, enable_mtp_training=True, mtp_decoder_input_detach=True)


@pytest.mark.parametrize('attr, value', [
    ('mtp_loss_scaling_factor', 0.2),
    ('enable_mtp_training', True),
    ('mtp_freeze', True),
    ('mtp_decoder_input_detach', True),
])
def test_every_dependent_knob_requires_mtp_num_layers(attr, value):
    """Without ``mtp_num_layers`` no MTP block exists, so these would configure nothing at all."""
    with pytest.raises(ValueError, match='mtp_num_layers'):
        _validate(**{attr: value})


def test_mtp_is_rejected_on_the_transformers_backend():
    """MTP lives in mcore-bridge; the HF path has no equivalent, so it cannot be quietly dropped."""
    from swift.dev.config import DistributedConfig, validate_configs
    kwargs = _configs(mtp_num_layers=1)
    kwargs['distributed_config'] = DistributedConfig(backend='hf')
    with pytest.raises(ValueError, match='megatron backend'):
        validate_configs(**kwargs)


def test_freeze_and_train_are_mutually_exclusive():
    """One drops the MTP gradient and the other asks for it; obeying either silently would mislead."""
    with pytest.raises(ValueError, match='contradictory'):
        _validate(mtp_num_layers=1, mtp_freeze=True, enable_mtp_training=True)


@pytest.mark.parametrize('attr', ['mtp_loss_scaling_factor', 'mtp_decoder_input_detach'])
def test_loss_shaping_knobs_require_training_to_be_on(attr):
    """Both describe the MTP *loss*, which is never computed unless joint training is enabled."""
    value = 0.2 if attr == 'mtp_loss_scaling_factor' else True
    with pytest.raises(ValueError, match='enable_mtp_training'):
        _validate(mtp_num_layers=1, **{attr: value})


def test_zero_layers_is_rejected_rather_than_read_as_off():
    """``0`` looks like "off" but reaches mcore as a configured-but-empty MTP block; None means off."""
    with pytest.raises(ValueError, match='must be >= 1'):
        _validate(mtp_num_layers=0, )


def test_lora_plus_joint_training_warns_instead_of_failing(caplog):
    """LoRA freezes base params, and the MTP layers are base params -- but an adapter *may* cover them.

    Undecidable from ``target_modules`` alone (it may be 'all-linear'), so this warns here and twinkle
    re-checks against the built model.

    Calls ``_check_mtp`` rather than ``validate_configs`` because the latter currently raises on ANY
    non-None tuner_config: ``_HF_ONLY`` still names ``lisa_activated_layers``, which TunerConfig no
    longer declares. That is a separate, pre-existing defect (it also fails four tests in
    test_dataset_api.py) and routing around it keeps this test about the MTP rule.
    """
    from swift.dev.config import ModelConfig, TunerConfig
    from swift.dev.config.validate import _check_mtp
    with caplog.at_level('WARNING'):
        _check_mtp(
            ModelConfig(model='dummy', mtp_num_layers=1, enable_mtp_training=True),
            is_megatron=True,
            tuner_config=TunerConfig(tuner_type='lora'),
        )
    assert 'enable_mtp_training' in caplog.text


# === The builder mapping (ModelConfig -> mcore-bridge ModelConfig kwargs) ===


def _mtp_kwargs(**model_overrides):
    from swift.dev.builders.model import _apply_mtp_kwargs
    from swift.dev.config import ModelConfig
    kwargs: dict = {}
    _apply_mtp_kwargs(kwargs, ModelConfig(model='dummy', **model_overrides))
    return kwargs


def test_no_mtp_kwarg_leaks_when_mtp_is_off():
    """An MTP-free run must reach the bridge with exactly the kwargs it had before MTP existed."""
    assert _mtp_kwargs() == {}


def test_unset_scaling_factor_is_not_forwarded_as_none():
    """mcore's own default is 0.1; forwarding None would overwrite it with None."""
    assert _mtp_kwargs(mtp_num_layers=1) == {'mtp_num_layers': 1}


def test_each_knob_reaches_the_bridge_under_its_own_name():
    assert _mtp_kwargs(
        mtp_num_layers=2,
        enable_mtp_training=True,
        mtp_loss_scaling_factor=0.2,
        mtp_decoder_input_detach=True,
    ) == {
        'mtp_num_layers': 2,
        'enable_mtp_training': True,
        'mtp_loss_scaling_factor': 0.2,
        'mtp_decoder_input_detach': True,
    }


def test_the_bridge_config_declares_every_knob_the_builder_forwards():
    """Guard against a rename turning a forwarded kwarg into a TypeError at model-build time.

    ``get_model_config`` passes **kwargs straight into the bridge's ModelConfig, so a field that no
    longer exists is only discovered when a real Megatron model is built -- late, and on a GPU.

    The accepted names are the union of two class bodies, because ``ModelConfig`` subclasses megatron's
    ``TransformerConfig``: ``mtp_num_layers`` and ``mtp_loss_scaling_factor`` are megatron's, while
    ``mtp_freeze`` / ``enable_mtp_training`` / ``mtp_decoder_input_detach`` are mcore-bridge's own.

    Read statically rather than by importing: ``import mcore_bridge`` pulls in megatron.core, which
    needs a working CUDA/TE stack, so an import-based check would skip on exactly the machines where
    the rest of this file runs.
    """
    bridge_fields = _dataclass_fields_from_source(_bridge_config_source(), 'ModelConfig')
    megatron_fields = _dataclass_fields_from_source(_megatron_transformer_config_source(), 'TransformerConfig')
    if bridge_fields is None or megatron_fields is None:
        pytest.skip('mcore-bridge / megatron-core sources not available for a static field check')

    forwarded = set(
        _mtp_kwargs(
            mtp_num_layers=2,
            enable_mtp_training=True,
            mtp_loss_scaling_factor=0.2,
            mtp_decoder_input_detach=True,
        ))
    accepted = bridge_fields | megatron_fields
    assert forwarded <= accepted, f'not accepted by mcore-bridge ModelConfig: {forwarded - accepted}'
    # The knobs mcore-bridge owns must be its own, not silently satisfied by an upstream namesake.
    assert {'mtp_freeze', 'enable_mtp_training', 'mtp_decoder_input_detach'} <= bridge_fields


def _bridge_config_source():
    from pathlib import Path
    return (Path(__file__).resolve().parents[3] / 'mcore-bridge' / 'src' / 'mcore_bridge' / 'config'
            / 'model_config.py')


def _megatron_transformer_config_source():
    """megatron is a namespace package, so ``__file__`` is None and ``__path__`` is the way in."""
    import megatron
    from pathlib import Path
    for root in getattr(megatron, '__path__', []):
        candidate = Path(root) / 'core' / 'transformer' / 'transformer_config.py'
        if candidate.exists():
            return candidate
    return None


def _dataclass_fields_from_source(source, class_name: str):
    """Annotated field names in one class body, parsed from source. None when unavailable."""
    import ast
    if source is None or not source.exists():
        return None
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.target.id
                for item in node.body if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            }
    return None


# === The training gate (which passes feed the MTP heads) ===


def _gate(*, enable=True, **call):
    """Call MegatronModel._mtp_training_enabled unbound, against a stub holding only what it reads.

    Unbound because constructing a MegatronModel needs a live Megatron process group; the gate itself
    reads nothing but ``strategy.config`` and its own arguments.
    """
    pytest.importorskip('twinkle', reason='needs twinkle for MegatronModel')
    from twinkle.model.megatron.megatron import MegatronModel

    stub = SimpleNamespace(
        strategy=SimpleNamespace(config=SimpleNamespace(enable_mtp_training=enable)),
        _mtp_trainable_checked=True,  # skip the built-model inspection, which needs a real model
    )
    kwargs = {'forward_only': False, 'task': 'causal_lm', 'disable_lora': False}
    kwargs.update(call)
    return MegatronModel._mtp_training_enabled(stub, **kwargs)


def test_gate_open_on_a_plain_training_pass():
    assert _gate() is True


def test_gate_closed_when_not_configured():
    """The default, and the reason an MTP-free run pays nothing for this feature existing."""
    assert _gate(enable=False) is False


def test_gate_closed_on_forward_only():
    """old/reference log-probs: no backward follows, so the MTP forward would be pure waste.

    This is the case no in-model check could catch, since megatron leaves training mode and autograd
    both enabled on such a pass.
    """
    assert _gate(forward_only=True) is False


@pytest.mark.parametrize('task', ['embedding', 'seq_cls', 'reranker', 'generative_reranker'])
def test_gate_closed_for_pooling_tasks(task):
    """A pooling head has no next-token target to give the MTP heads."""
    assert _gate(task=task) is False


def test_gate_closed_on_the_base_policy_pass_of_a_lora_run():
    """``disable_lora`` marks a reference pass, whose gradients are discarded by construction."""
    assert _gate(disable_lora=True) is False


def test_gate_tolerates_a_config_without_the_field():
    """A twinkle model built against an older bridge config must read as "off", not raise."""
    pytest.importorskip('twinkle', reason='needs twinkle for MegatronModel')
    from twinkle.model.megatron.megatron import MegatronModel

    stub = SimpleNamespace(strategy=SimpleNamespace(config=SimpleNamespace()), _mtp_trainable_checked=True)
    assert MegatronModel._mtp_training_enabled(stub, forward_only=False, task='causal_lm', disable_lora=False) is False


# === The train loop's metric drain ===


class _FakeModel:
    """A model exposing only what _record_step touches, with a drainable MTP tracker."""

    def __init__(self, mtp=None):
        self._mtp = mtp
        self.pop_calls = 0
        if mtp is None:
            # Absent attribute, matching the transformers backend / an MTP-free megatron run.
            del self.__dict__['_mtp']
            self._mtp = None
        else:
            self.pop_mtp_metrics = self._pop

    def _pop(self):
        self.pop_calls += 1
        # Drained: a second read returns nothing, which is what keeps the logged loss from growing.
        result, self._mtp = self._mtp, {}
        return result

    def calculate_metric(self, is_training):
        return {'loss': 1.5, 'grad_norm': 0.25}


def _loop_with(model):
    from swift.dev.recipe.train_loop import SFTLoop
    loop = SFTLoop.__new__(SFTLoop)
    loop.model = model
    loop.global_step = 0
    loop.history = []
    loop.logging_steps = 0
    loop.save_steps = 0
    loop.eval_steps = 0
    loop.eval_dataloader = None
    return loop


def test_mtp_metrics_land_in_the_step_record():
    model = _FakeModel({'mtp_1_loss': 2.0, 'mtp_loss': 2.0, 'mtp_1_accept': 0.4})
    loop = _loop_with(model)
    loop._record_step()
    record = loop.history[-1]
    assert record['mtp_1_loss'] == 2.0
    assert record['mtp_loss'] == 2.0
    assert record['mtp_1_accept'] == 0.4
    assert record['loss'] == 1.5


def test_the_record_is_unchanged_when_the_model_has_no_mtp():
    loop = _loop_with(_FakeModel())
    loop._record_step()
    assert set(loop.history[-1]) == {'step', 'loss', 'grad_norm'}


def test_an_empty_drain_adds_no_keys():
    """MTP configured but nothing accumulated (e.g. a non-last pipeline stage) must not add noise."""
    loop = _loop_with(_FakeModel({}))
    loop._record_step()
    assert set(loop.history[-1]) == {'step', 'loss', 'grad_norm'}


def test_each_step_drains_exactly_once():
    """Two reads per step would double-count; zero would report a tracker that grows forever."""
    model = _FakeModel({'mtp_loss': 1.0})
    loop = _loop_with(model)
    loop._record_step()
    loop._record_step()
    assert model.pop_calls == 2
    assert loop.history[0]['mtp_loss'] == 1.0
    assert 'mtp_loss' not in loop.history[1]


# --- The legacy Megatron CLI bridge -----------------------------------------------------------------
# args_to_configs copies same-name fields, and the two surfaces disagree about MTP in two ways: the
# scaling factor is non-Optional on the legacy side (so its default reads as an explicit choice), and
# enable_mtp_training has no legacy flag at all (legacy passes labels, so it never needed one).


class _LegacyArgs:
    """Just the MTP surface of MegatronSftArguments, with legacy's own defaults."""

    def __init__(self, mtp_num_layers=None, mtp_loss_scaling_factor=0.1, mtp_decoder_input_detach=False):
        self.mtp_num_layers = mtp_num_layers
        self.mtp_loss_scaling_factor = mtp_loss_scaling_factor
        self.mtp_decoder_input_detach = mtp_decoder_input_detach


def _bridged(**kwargs):
    """A ModelConfig filled the way args_to_configs fills it, then reconciled."""
    from swift.dev.cli.megatron import _fill_from_args, _fix_mtp
    from swift.dev.config import ModelConfig
    args = _LegacyArgs(**kwargs)
    model_config = _fill_from_args(ModelConfig(model='dummy'), args)
    _fix_mtp(model_config, args)
    return model_config


def _validate_bridged(model_config):
    """Run the full cross-config validation over an already-built ModelConfig.

    Separate from ``_validate``, which builds its own from keyword overrides -- here the object the
    bridge produced is precisely what is under test.
    """
    from swift.dev.config import validate_configs
    configs = _configs()
    configs['model_config'] = model_config
    validate_configs(**configs)


def test_a_legacy_run_without_mtp_stays_without_mtp():
    """Regression: legacy's ``mtp_loss_scaling_factor: float = 0.1`` is copied on EVERY run.

    Left alone it reaches dev as an explicit scaling factor, and validate_configs then rejects a
    perfectly ordinary non-MTP run for configuring a factor with no MTP layer to scale -- i.e. adding
    the MTP knobs would have broken the whole legacy Megatron CLI, not just its MTP users.
    """
    model_config = _bridged()
    assert model_config.mtp_num_layers is None
    assert model_config.mtp_loss_scaling_factor is None
    assert model_config.enable_mtp_training is False
    _validate_bridged(model_config)  # must not raise


def test_legacy_mtp_num_layers_alone_still_trains_the_mtp_heads():
    """--mtp_num_layers means "train them" on the legacy surface and must keep meaning that.

    legacy's trainer hands ``labels`` to the model, so the MTP loss is computed and logged off that one
    flag (megatron/trainers/base.py tracks it whenever mtp_num_layers is set). dev withholds labels, so
    without deriving enable_mtp_training the same command line would build MTP layers, never train
    them, and report no mtp_loss -- silently.
    """
    model_config = _bridged(mtp_num_layers=1)
    assert model_config.enable_mtp_training is True
    assert model_config.mtp_loss_scaling_factor == 0.1
    _validate_bridged(model_config)


def test_legacy_mtp_knobs_survive_the_bridge():
    model_config = _bridged(mtp_num_layers=2, mtp_loss_scaling_factor=0.2, mtp_decoder_input_detach=True)
    assert model_config.mtp_num_layers == 2
    assert model_config.mtp_loss_scaling_factor == 0.2
    assert model_config.mtp_decoder_input_detach is True
    _validate_bridged(model_config)
