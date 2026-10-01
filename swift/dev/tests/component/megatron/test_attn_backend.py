"""attn_impl must select a real Megatron attention kernel, as an enum, defaulting to legacy's flash.

Why this exists. Nothing in dev used to set TransformerConfig.attention_backend on the mcore-bridge
path, so it stayed at mcore's own default AttnBackend.auto
(Megatron-LM transformer_config.py:144), while legacy swift explicitly asks for 'flash'
(MegatronArguments.attention_backend = 'flash', megatron_args.py:499, converted to the enum in
_init_attention_backend). Under auto, transformer_engine picks per shape -- and for the dense.sh
forward (Qwen2.5 bf16, causal, THD from padding_free, head_dim 64, 14 q heads / 2 KV groups) its own
dispatcher reports use_flash=False, use_fused=True (NVTE_F16_arbitrary_seqlen). The two pipelines were
therefore running DIFFERENT attention kernels on identical configuration.

The enum assertion is the load-bearing one: mcore compares this field by identity
(``self.attention_backend == AttnBackend.flash``, transformer_config.py:2720) and TransformerConfig
is a plain dataclass with no coercion, so a string 'flash' would type-check and then compare unequal
to every member -- a silent fallback of exactly the kind this refactor keeps finding.
"""
from __future__ import annotations

import pytest
import sys
from contextlib import contextmanager
from unittest import mock

pytest.importorskip('megatron.core', reason='needs megatron-core for AttnBackend')

from megatron.core.transformer.enums import AttnBackend  # noqa: E402

from swift.dev.naming import flash_version_pin, resolve_megatron_attn_backend  # noqa: E402


@pytest.mark.parametrize('value, expected', [
    ('flash', AttnBackend.flash),
    ('fused', AttnBackend.fused),
    ('unfused', AttnBackend.unfused),
    ('local', AttnBackend.local),
    ('auto', AttnBackend.auto),
])
def test_each_megatron_value_maps_to_its_enum(value, expected):
    assert resolve_megatron_attn_backend(value) is expected


def test_default_is_flash_not_mcore_auto():
    """Unset attn_impl must mean legacy's flash, NOT mcore's auto.

    This is a deliberate divergence from the mcore default; see the module docstring for the measured
    consequence of leaving auto in place.
    """
    assert resolve_megatron_attn_backend(None) is AttnBackend.flash


def test_result_is_an_enum_member_not_a_string():
    """The whole point: mcore compares by identity, so a string would silently mean 'not flash'."""
    for value in (None, 'flash', 'fused', 'auto'):
        got = resolve_megatron_attn_backend(value)
        assert isinstance(got, AttnBackend), f'{value!r} resolved to {type(got).__name__}, not the enum'
        assert not isinstance(got, str), f'{value!r} resolved to a bare string'


def test_case_is_normalised():
    assert resolve_megatron_attn_backend('FLASH') is AttnBackend.flash
    assert resolve_megatron_attn_backend('Fused') is AttnBackend.fused


@pytest.mark.parametrize(
    'hf_value, expected',
    [
        ('flash_attn', AttnBackend.flash),  # HF's alias for FA; carries no version of its own
        ('sdpa', AttnBackend.unfused),  # TE: unfused IS the native PyTorch impl
        ('eager', AttnBackend.local),  # mcore's own non-TE pytorch attention
    ])
def test_transformers_style_values_are_translated_not_rejected(hf_value, expected):
    """attn_impl is ONE shared field, so transformers names must work on this backend too.

    Every one of these has a real Megatron counterpart, so refusing them would make e.g.
    `--attn_impl sdpa` unusable on Megatron for no reason -- legacy's HF path is essentially a
    pass-through of this value, and the capability exists on both sides. Mapping evidence:
      flash_attn -> flash   (same FA kernel, no version pinned: TE picks the installed build)
      sdpa       -> unfused (TE calls unfused "the native PyTorch
                            implementation", dot_product_attention.py:848)
      eager      -> local   (mcore's own pytorch attention)

    flash_attention_2/_3/_4 are NOT here: they name a specific FA version, i.e. the same intent as
    Megatron's flash_2/_3/_4, and are refused together with them (see
    test_flash_version_pinning_fails_fast).
    """
    assert resolve_megatron_attn_backend(hf_value) is expected


def test_only_kernels_without_a_counterpart_are_rejected():
    """flex_attention has no Megatron/TE implementation at all, so it cannot be translated.

    Verified by absence: no 'flex' hit in megatron-core's enums.py nor in transformer_engine's
    dot_product_attention. Rejecting is right here precisely because silently substituting some other
    kernel would change what the model computes.
    """
    with pytest.raises(NotImplementedError) as exc:
        resolve_megatron_attn_backend('flex_attention')
    message = str(exc.value)
    assert 'no Megatron equivalent' in message, message
    # The message must list both what Megatron has and what does translate, or the user is stuck.
    assert 'flash' in message and 'sdpa' in message, message


@pytest.mark.parametrize(
    'pinned, version',
    [
        ('flash_2', 2),
        ('flash_3', 3),
        ('flash_4', 4),  # Megatron spelling
        ('flash_attention_2', 2),
        ('flash_attention_3', 3),
        ('flash_attention_4', 4),  # transformers
    ])
def test_flash_version_pin_selects_the_flash_kernel(pinned, version):
    """A version pin resolves to the flash KERNEL; the version is enforced separately.

    Mirrors legacy, which collapses its own flash_N to 'flash' after mutating TE globals
    (megatron_args.py:917) -- the enum carries the kernel, apply_flash_version_pin carries the version.
    BOTH spellings must agree: flash_attention_N used to map to plain 'flash' with the pin silently
    dropped, while flash_N raised, so identical intent had two different outcomes.
    """
    assert resolve_megatron_attn_backend(pinned) is AttnBackend.flash
    assert flash_version_pin(pinned) == version


@pytest.mark.parametrize('unversioned', ['flash', 'flash_attn', 'sdpa', 'eager', 'auto'])
def test_unversioned_names_pin_nothing(unversioned):
    """Only a trailing integer makes a name a pin; bare aliases stay a plain kernel choice."""
    assert flash_version_pin(unversioned) is None


def test_unsupported_pin_version_fails_fast():
    """TE has an availability flag per FA major version; a version it has none for cannot be pinned."""
    with pytest.raises(NotImplementedError, match='no availability flag'):
        resolve_megatron_attn_backend('flash_5')


def test_apply_flash_version_pin_is_a_noop_without_a_pin():
    """No pin -> TE's own flags are left exactly as they were, so its dispatcher keeps its choice."""
    from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils

    from swift.dev.naming import apply_flash_version_pin

    flags = ('is_installed', 'v3_is_installed', 'v4_is_installed')
    before = {f: getattr(FlashAttentionUtils, f, None) for f in flags}
    assert apply_flash_version_pin('flash') is None
    assert apply_flash_version_pin(None) is None
    assert {f: getattr(FlashAttentionUtils, f, None) for f in flags} == before


def test_apply_flash_version_pin_disables_the_other_versions():
    """The pin works by turning the OTHER versions off -- that is TE's only version lever.

    Run against a stub rather than the real TE class: this mutates module-level state, so touching the
    real one would leak into every later test in the process (legacy has the same hazard, which is why
    the pin is applied once at strategy construction).
    """
    from swift.dev import naming

    class _Stub:
        is_installed = True
        v3_is_installed = True
        v4_is_installed = True

    module = mock.MagicMock()
    module.FlashAttentionUtils = _Stub
    with mock.patch.dict(sys.modules, {'transformer_engine.pytorch.attention.dot_product_attention.utils': module}):
        assert naming.apply_flash_version_pin('flash_3') == 3
    assert _Stub.v3_is_installed is True, 'the requested version must stay enabled'
    assert _Stub.is_installed is False and _Stub.v4_is_installed is False, \
        'the other versions must be disabled so TE cannot fall back to them'


def test_apply_flash_version_pin_rejects_a_version_that_is_not_installed():
    """Asking for a build that is not installed must fail loudly, not fall back to another version."""
    from swift.dev import naming

    class _Stub:
        is_installed = True
        v3_is_installed = False
        v4_is_installed = False

    module = mock.MagicMock()
    module.FlashAttentionUtils = _Stub
    with mock.patch.dict(sys.modules, {'transformer_engine.pytorch.attention.dot_product_attention.utils': module}):
        with pytest.raises(ValueError, match='not installed'):
            naming.apply_flash_version_pin('flash_3')
    assert _Stub.is_installed is True, 'a failed pin must not have disabled anything'


def test_unknown_value_is_rejected():
    with pytest.raises(NotImplementedError, match='Unknown attn_impl'):
        resolve_megatron_attn_backend('nonsense')


@contextmanager
def _capture_megatron_kwargs():
    """Run build_model's Megatron path with a bogus model id and capture MegatronModel's kwargs.

    build_model resolves the HF config from ``model_config.model`` BEFORE it constructs
    MegatronModel, so a driver-side forwarding test must stub that load (no network / no real
    checkpoint) or it dies on the fake path before ever reaching the constructor. Yields the mock
    class so the caller asserts on ``.call_args.kwargs`` -- the exact kwargs that ride the Ray worker
    boundary, which is what bug#4 turned on.
    """
    from unittest.mock import MagicMock, patch

    with patch('swift.dev.builders.model._resolve_model_loader', return_value=None), \
            patch('swift.dev.builders.model._build_hf_config', return_value=MagicMock()), \
            patch('twinkle.model.megatron.MegatronModel') as mock_model:
        yield mock_model


def test_build_model_forwards_the_backend_to_megatron():
    """The resolver is useless unless build_model actually passes it down.

    Asserted on the kwargs handed to twinkle's MegatronModel, which is where mcore reads it from
    (get_model_config does config_kwargs.update(kwargs), strategy/megatron.py:271-272).
    """
    from swift.dev.builders.model import build_model
    from swift.dev.config import DistributedConfig, ModelConfig

    with _capture_megatron_kwargs() as mock_model:
        build_model(
            ModelConfig(model='/does/not/matter', attn_impl='fused'),
            DistributedConfig(backend='megatron', mode='local', nproc_per_node=1))
    assert mock_model.call_args is not None, 'MegatronModel was never constructed'
    forwarded = mock_model.call_args.kwargs.get('attention_backend')
    assert forwarded is AttnBackend.fused, f'build_model forwarded {forwarded!r}'


def test_build_model_forwards_attn_impl_for_the_worker_side_pin():
    """The raw attn_impl string must reach MegatronModel, not just the resolved enum.

    The version pin is applied by flipping transformer_engine module globals, a PER-PROCESS side
    effect. In Ray mode build_model runs on the driver, which is not the process that builds or runs
    the model, so the pin cannot be applied there -- MegatronStrategy (worker-side) needs the
    original string. Forwarding only the enum would collapse flash_3 to AttnBackend.flash and lose the
    version for good.
    """
    from swift.dev.builders.model import build_model
    from swift.dev.config import DistributedConfig, ModelConfig

    with _capture_megatron_kwargs() as mock_model:
        build_model(
            ModelConfig(model='/does/not/matter', attn_impl='flash_3'),
            DistributedConfig(backend='megatron', mode='local', nproc_per_node=1))
    kwargs = mock_model.call_args.kwargs
    assert kwargs.get('attention_backend') is AttnBackend.flash, 'the kernel choice'
    assert kwargs.get('attn_impl') == 'flash_3', 'the raw string the worker needs for the pin'


def test_strategy_applies_the_pin_and_keeps_attn_impl_out_of_the_mcore_config():
    """MegatronStrategy.__init__ applies the pin on the worker AND binds attn_impl as a named param.

    attn_impl is a caller field name; mcore's ModelConfig has no such attribute, so letting it ride
    in **kwargs into get_model_config is exactly bug#4's TypeError. A named param is bound by Python
    and never reaches that **kwargs. The pin runs here rather than in build_model because it flips
    transformer_engine module globals -- in Ray mode this __init__ is the worker that builds the
    model, while build_model runs on the driver.
    """
    from twinkle.model.megatron.strategy.megatron import MegatronStrategy

    seen = {}

    def _fake_get_model_config(self, hf_config, parallel_kwargs, **kwargs):
        seen['config_kwargs'] = kwargs
        return object()

    class _FakeMesh:
        tp_world_size = pp_world_size = cp_world_size = 1
        ep_size = etp_world_size = 1
        vpp_size = 1
        order = None
        sequence_parallel = False

    with mock.patch('twinkle.model.megatron._flash_attn.apply_flash_version_pin', return_value=3) as pin, \
            mock.patch('megatron.core.mpu.initialize_model_parallel'), \
            mock.patch('megatron.core.tensor_parallel.random.model_parallel_cuda_manual_seed'), \
            mock.patch.object(MegatronStrategy, 'get_model_config', _fake_get_model_config), \
            mock.patch.object(MegatronStrategy, '_finalize_quantized_param_config'), \
            mock.patch.object(MegatronStrategy, '_check_fsdp'):
        MegatronStrategy('/model/dir', device_mesh=_FakeMesh(), attn_impl='flash_3', config=mock.MagicMock())
    pin.assert_called_once_with('flash_3')
    assert 'attn_impl' not in seen['config_kwargs'], \
        'attn_impl must be bound by __init__, never forwarded into ModelConfig'


def test_both_bridge_backends_get_the_same_backend():
    """mcore-bridge and megatron-bridge must not disagree on the attention kernel.

    megatron_bridge.py used to hardcode AttnBackend.flash for itself only, leaving the mcore-bridge
    path on auto: the same dev Config ran a different kernel depending on bridge_backend. Both now
    read the single resolved value, so this compares them rather than trusting the removal.
    """
    from swift.dev.builders.model import build_model
    from swift.dev.config import DistributedConfig, ModelConfig

    seen = {}
    for bridge in ('mcore-bridge', 'megatron-bridge'):
        with _capture_megatron_kwargs() as mock_model:
            build_model(
                ModelConfig(model='/does/not/matter'),
                DistributedConfig(backend='megatron', mode='local', nproc_per_node=1, bridge_backend=bridge))
        seen[bridge] = mock_model.call_args.kwargs.get('attention_backend')
    assert seen['mcore-bridge'] is seen['megatron-bridge'] is AttnBackend.flash, \
        f'bridge backends disagree on the attention kernel: {seen}'
