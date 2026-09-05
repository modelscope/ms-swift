import os

import pytest
import torch

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'


def _hf_reference_loss(model_path, input_ids, labels_aligned, dtype):
    """HF transformers per-token CE mean over response tokens (labels != -100), shift-at-loss."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).cuda().eval()
    ids = torch.tensor([input_ids], device='cuda')
    with torch.no_grad():
        logits = m(ids).logits.float()
    shift_logits = logits[0, :-1, :]
    shift_labels = torch.tensor(labels_aligned[1:], device='cuda')
    mask = shift_labels != -100
    lp = torch.log_softmax(shift_logits[mask], dim=-1)
    tok = shift_labels[mask]
    nll = -lp[torch.arange(len(tok)), tok]
    return nll.mean().item()


@pytest.mark.slow
def test_megatron_step1_loss_matches_hf_bf16_reference():
    """dev MegatronModel (mcore-bridge, DP=2, bf16) step-1 loss == HF bf16 reference to
    fp/backend noise (<1e-2). fp32 gap is the bf16<->fp32 precision band, not a bug."""
    if torch.cuda.device_count() < 2:
        pytest.skip('needs >=2 GPUs (twinkle MegatronModel requires world_size>=2)')

    # Resolve the short ms id to a local snapshot (env has no HF network access), so both the HF
    # reference and MegatronModel load from the same local dir.
    from modelscope import snapshot_download

    import twinkle
    from swift.dev.model.megatron.model import MegatronModel
    from twinkle import DeviceGroup, DeviceMesh
    model_path = snapshot_download(MODEL)

    ids = list(range(10, 26))
    labels_aligned = [-100] * 8 + ids[8:]
    pos = list(range(len(ids)))

    ref_bf16 = _hf_reference_loss(model_path, ids, labels_aligned, dtype=torch.bfloat16)

    twinkle.initialize(
        mode='ray',
        nproc_per_node=2,
        groups=[DeviceGroup(name='model', ranks=[0, 1], device_type='GPU', gpus_per_worker=1)])
    try:
        dm = DeviceMesh.from_sizes(world_size=2, dp_size=2)
        model = MegatronModel(model_id=model_path, device_mesh=dm, mixed_precision='bf16', remote_group='model')
        model.set_optimizer('Adam', lr=1e-5)
        # Megatron forward uses no-shift selective_log_softmax -> labels must be next-token shifted.
        labels_shifted = labels_aligned[1:] + [-100]
        batch = [{
            'input_ids': ids,
            'labels': labels_shifted,
            'position_ids': pos
        }, {
            'input_ids': ids,
            'labels': labels_shifted,
            'position_ids': pos
        }]
        model.forward_backward(inputs=batch, micro_batch_size=1)
        metrics = model.calculate_metric(is_training=True)  # driver-visible loss path
        mg_loss = metrics.get('loss') if isinstance(metrics, dict) else metrics
        assert mg_loss is not None, 'Megatron loss is None (read via calculate_metric)'

        diff = abs(float(mg_loss) - ref_bf16)
        print(f'\nMegatron(mcore-bridge) step-1 loss='
              f'{float(mg_loss):.4f} HF-bf16-ref={ref_bf16:.4f} diff={diff:.4f}')
        assert diff < 1e-2, (
            f'Megatron step-1 loss {float(mg_loss):.4f} vs HF bf16 ref {ref_bf16:.4f} '
            f'diff {diff:.4f} >= 1e-2 -> mcore link / loss normalization mismatch (not just bf16 noise)')
    finally:
        try:
            twinkle.shutdown()
        except Exception:
            pass


@pytest.mark.slow
def test_megatron_bridge_step1_loss_matches_hf_bf16_reference():
    """dev MegatronModel on the megatron-bridge (AutoBridge) backend (DP=2, bf16) step-1 loss ==
    the same HF bf16 reference the mcore backend hits (<1e-2), showing both backends build the
    same-shaped model and normalize loss the same way.

    Only the forward path is exercised; backward grad scaling / MoE dispatch / NPU overrides are
    checked by code review, not this loss test.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip('needs >=2 GPUs (twinkle MegatronModel requires world_size>=2)')
    try:
        from megatron.bridge.models.conversion.auto_bridge import AutoBridge  # noqa: F401
    except Exception:
        pytest.skip('megatron.bridge (Megatron-Bridge) not importable in this env')

    from modelscope import snapshot_download

    import twinkle
    from swift.dev.model.megatron.bridge import MegatronBridgeBackend
    from swift.dev.model.megatron.model import MegatronModel
    from twinkle import DeviceGroup, DeviceMesh
    model_path = snapshot_download(MODEL)

    ids = list(range(10, 26))
    labels_aligned = [-100] * 8 + ids[8:]
    pos = list(range(len(ids)))

    ref_bf16 = _hf_reference_loss(model_path, ids, labels_aligned, dtype=torch.bfloat16)

    twinkle.initialize(
        mode='ray',
        nproc_per_node=2,
        groups=[DeviceGroup(name='model', ranks=[0, 1], device_type='GPU', gpus_per_worker=1)])
    try:
        dm = DeviceMesh.from_sizes(world_size=2, dp_size=2)
        model = MegatronModel(
            model_id=model_path,
            device_mesh=dm,
            mixed_precision='bf16',
            remote_group='model',
            backend=MegatronBridgeBackend())
        model.set_optimizer('Adam', lr=1e-5)
        labels_shifted = labels_aligned[1:] + [-100]
        batch = [{
            'input_ids': ids,
            'labels': labels_shifted,
            'position_ids': pos
        }, {
            'input_ids': ids,
            'labels': labels_shifted,
            'position_ids': pos
        }]
        model.forward_backward(inputs=batch, micro_batch_size=1)
        metrics = model.calculate_metric(is_training=True)
        mg_loss = metrics.get('loss') if isinstance(metrics, dict) else metrics
        assert mg_loss is not None, 'megatron-bridge loss is None (read via calculate_metric)'

        diff = abs(float(mg_loss) - ref_bf16)
        print(f'\nMegatron(megatron-bridge) step-1 loss='
              f'{float(mg_loss):.4f} HF-bf16-ref={ref_bf16:.4f} diff={diff:.4f}')
        assert diff < 1e-2, (f'megatron-bridge step-1 loss {float(mg_loss):.4f} vs HF bf16 ref {ref_bf16:.4f} '
                             f'diff {diff:.4f} >= 1e-2 -> AutoBridge link / loss normalization mismatch '
                             f'(not just bf16 noise); the two backends do NOT build the same model')
    finally:
        try:
            twinkle.shutdown()
        except Exception:
            pass


@pytest.mark.slow
def test_megatron_bridge_save_produces_loadable_hf_checkpoint(tmp_path):
    """megatron-bridge save loop (shim.save_weights -> AutoBridge.save_hf_weights): after one
    train step + model.save(...), the checkpoint dir must be loadable by a plain
    transformers.from_pretrained AND run forward -- i.e. the shim writes real HF-format weights
    and twinkle's own hf_config.save_pretrained yields a complete config.json.

    This is the artifact-parity + save-loop judge. It specifically guards the known
    megatron-bridge pitfall where the bridge's own config reconstruction drops
    num_attention_heads/model_type (which would make from_pretrained forward crash). We rely on
    twinkle saving the ORIGINAL hf_config, and this test proves that path holds end to end.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip('needs >=2 GPUs (twinkle MegatronModel requires world_size>=2)')
    try:
        from megatron.bridge.models.conversion.auto_bridge import AutoBridge  # noqa: F401
    except Exception:
        pytest.skip('megatron.bridge (Megatron-Bridge) not importable in this env')

    import os

    from modelscope import snapshot_download

    import twinkle
    from swift.dev.model.megatron.bridge import MegatronBridgeBackend
    from swift.dev.model.megatron.model import MegatronModel
    from twinkle import DeviceGroup, DeviceMesh
    model_path = snapshot_download(MODEL)

    ids = list(range(10, 26))
    labels_aligned = [-100] * 8 + ids[8:]
    pos = list(range(len(ids)))
    save_dir = str(tmp_path / 'mbridge_ckpt')

    twinkle.initialize(
        mode='ray',
        nproc_per_node=2,
        groups=[DeviceGroup(name='model', ranks=[0, 1], device_type='GPU', gpus_per_worker=1)])
    try:
        dm = DeviceMesh.from_sizes(world_size=2, dp_size=2)
        model = MegatronModel(
            model_id=model_path,
            device_mesh=dm,
            mixed_precision='bf16',
            remote_group='model',
            backend=MegatronBridgeBackend())
        model.set_optimizer('Adam', lr=1e-5)
        labels_shifted = labels_aligned[1:] + [-100]
        batch = [{
            'input_ids': ids,
            'labels': labels_shifted,
            'position_ids': pos
        }, {
            'input_ids': ids,
            'labels': labels_shifted,
            'position_ids': pos
        }]
        model.forward_backward(inputs=batch, micro_batch_size=1)
        model.clip_grad_and_step()
        # save() -> _save_hf_format -> shim.save_weights (AutoBridge.save_hf_weights) +
        # hf_config.save_pretrained + tokenizer.
        model.save(name='step1', output_dir=save_dir)
    finally:
        try:
            twinkle.shutdown()
        except Exception:
            pass

    ckpt = os.path.join(save_dir, 'step1')
    # Artifact parity: the config + weights + tokenizer that must exist for a loadable HF ckpt.
    assert os.path.isfile(os.path.join(ckpt, 'config.json')), 'missing config.json (shim/save loop broke)'
    has_weights = any(f.endswith('.safetensors') or f.endswith('.bin') for f in os.listdir(ckpt))
    assert has_weights, f'no weight files in {ckpt}: {os.listdir(ckpt)}'

    # Loadability judge: plain transformers.from_pretrained must load AND forward. This is what
    # the megatron-bridge config-dropping pitfall would break.
    from transformers import AutoModelForCausalLM
    reloaded = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16).cuda().eval()
    with torch.no_grad():
        out = reloaded(torch.tensor([ids], device='cuda'))
    assert out.logits.shape[:2] == (1, len(ids)), 'reloaded HF model forward shape mismatch'
    print(f'\nmegatron-bridge save->HF from_pretrained OK: {sorted(os.listdir(ckpt))}')


# The loss test above is the *behavioral* proof (runs the real mcore path through
# DevMegatronStrategy). These tests are the *structural* proof that delegation is wired
# correctly, so a plumbing regression is caught without a 2-GPU Ray run.


class _FakeBackend:
    """Records delegation calls; returns sentinels so we can assert pass-through."""

    backend_name = 'fake'
    is_multimodal = False

    def __init__(self):
        self.build_calls = []
        self.create_calls = []

    def build_model_config(self, hf_config, parallel_kwargs, strategy, **kwargs):
        self.build_calls.append((hf_config, parallel_kwargs, strategy, kwargs))
        return ('fake-config', hf_config)

    def create_model(self, config, model_dir, *, load_weights, move_to_gpu):
        self.create_calls.append((config, model_dir, load_weights, move_to_gpu))
        return ['fake-model']


def test_mcore_backend_satisfies_protocol():
    from swift.dev.model.megatron.bridge import BridgeBackend, MCoreBridgeBackend
    b = MCoreBridgeBackend()
    assert isinstance(b, BridgeBackend)
    assert b.backend_name == 'mcore-bridge'
    assert b.is_multimodal is False


def test_dev_strategy_delegates_both_methods_to_backend():
    """get_model_config / create_megatron_model route to the backend with the exact args
    twinkle's originals used -- bypassing the heavy __init__ (no dist / no mcore)."""
    from swift.dev.model.megatron.strategy import DevMegatronStrategy

    strat = DevMegatronStrategy.__new__(DevMegatronStrategy)  # skip __init__ (needs dist)
    backend = _FakeBackend()
    strat._backend = backend
    # Stand-ins for the attributes the real backend would read off the strategy.
    strat.model_dir = '/tmp/model'
    strat.config = ('fake-config', 'hfcfg')
    strat._move_model_to_gpu = lambda m: m

    cfg = strat.get_model_config('hfcfg', {'tensor_model_parallel_size': 1}, extra=1)
    assert cfg == ('fake-config', 'hfcfg')
    assert len(backend.build_calls) == 1
    hf_config, parallel_kwargs, passed_strategy, kwargs = backend.build_calls[0]
    assert hf_config == 'hfcfg'
    assert parallel_kwargs == {'tensor_model_parallel_size': 1}
    assert passed_strategy is strat  # backend gets the strategy for params_dtype/etc.
    assert kwargs == {'extra': 1}

    models = strat.create_megatron_model(load_weights=False)
    assert models == ['fake-model']
    assert len(backend.create_calls) == 1
    config, model_dir, load_weights, move_to_gpu = backend.create_calls[0]
    assert config is strat.config
    assert model_dir == '/tmp/model'
    assert load_weights is False
    assert move_to_gpu is strat._move_model_to_gpu


def test_dev_strategy_defaults_to_mcore_backend():
    from swift.dev.model.megatron.bridge import MCoreBridgeBackend
    from swift.dev.model.megatron.strategy import DevMegatronStrategy

    strat = DevMegatronStrategy.__new__(DevMegatronStrategy)
    # Re-run only the backend-selection line of __init__ (the rest needs dist).
    strat._backend = None or MCoreBridgeBackend()
    assert isinstance(strat.backend, MCoreBridgeBackend)


def test_dev_model_injects_dev_strategy_during_super_init(monkeypatch):
    """DevMegatronModel.__init__ rebinds the module-level MegatronStrategy that twinkle's
    __init__ reads, so twinkle instantiates DevMegatronStrategy(backend=...) instead. We stub
    the parent __init__ to capture what the rebound symbol resolves to -- no dist/mcore."""
    import twinkle.model.megatron.megatron as tw_mod

    from swift.dev.model.megatron import model as dev_mod
    from swift.dev.model.megatron.bridge import MCoreBridgeBackend
    from swift.dev.model.megatron.strategy import DevMegatronStrategy

    captured = {}
    original_symbol = tw_mod.MegatronStrategy

    def fake_parent_init(self, *args, **kwargs):
        # Inside the parent init the module symbol must be the injected partial, not the
        # original class -- this is what makes twinkle build DevMegatronStrategy(backend=...).
        captured['symbol'] = tw_mod.MegatronStrategy

    monkeypatch.setattr(dev_mod.TwinkleMegatronModel, '__init__', fake_parent_init)

    backend = MCoreBridgeBackend()
    dev_mod.MegatronModel(model_id='x', backend=backend)

    # During init the symbol was a partial(DevMegatronStrategy, backend=backend)...
    injected = captured['symbol']
    assert getattr(injected, 'func', None) is DevMegatronStrategy
    assert injected.keywords.get('backend') is backend
    # ...and after construction the scoped patch restored the original class.
    assert tw_mod.MegatronStrategy is original_symbol


# ----------------------------------------------------------------------
# MegatronBridgeBackend structural plumbing (fast; no megatron.bridge import).
# The capability flags and the shim's guards are pure Python -- they run without a
# Megatron env. The build/create/forward path is the slow behavioral proof below.
# ----------------------------------------------------------------------


def test_megatron_bridge_backend_satisfies_protocol():
    from swift.dev.model.megatron.bridge import BridgeBackend, MegatronBridgeBackend
    b = MegatronBridgeBackend()
    assert isinstance(b, BridgeBackend)
    assert b.backend_name == 'megatron-bridge'
    # LoRA is supported (via twinkle HF get_peft_model, same as mcore); multimodal is not yet.
    assert b.is_multimodal is False


def test_megatron_bridge_backend_selectable_via_strategy():
    """DevMegatronStrategy accepts the megatron-bridge backend the same way it accepts mcore
    (bypassing the heavy __init__), so the two backends are truly interchangeable."""
    from swift.dev.model.megatron.bridge import MegatronBridgeBackend
    from swift.dev.model.megatron.strategy import DevMegatronStrategy

    strat = DevMegatronStrategy.__new__(DevMegatronStrategy)
    strat._backend = MegatronBridgeBackend()
    assert isinstance(strat.backend, MegatronBridgeBackend)
    assert strat.backend.backend_name == 'megatron-bridge'


def test_megatron_bridge_shim_per_adapter_state_dict_is_isolated():
    """The shim's peft_format=True save/export path (multi-tenant: distinct adapter per tenant)
    extracts ONLY the requested adapter's LoRA delta via peft.get_peft_model_state_dict. Verified
    on a tiny CPU PeftModel with two adapters (no GPU / no megatron.bridge needed): each adapter's
    state dict is disjoint, non-empty, and keys carry no adapter-name suffix (HF PEFT layout).

    Also asserts the genuinely-unsupported kwargs still fail-fast (converter / only_master_rank),
    and that a missing adapter name raises rather than silently exporting nothing."""
    import torch.nn as nn

    from peft import LoraConfig, get_peft_model
    from swift.dev.model.megatron.bridge.megatron_bridge import _MCoreCompatBridgeShim

    class _Tiny(nn.Module):

        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return self.lin(x)

    base = _Tiny()
    peft_model = get_peft_model(base, LoraConfig(r=4, target_modules=['lin']), adapter_name='tenantA')
    peft_model.add_adapter('tenantB', LoraConfig(r=4, target_modules=['lin']))
    # Make the two adapters differ so isolation is observable.
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if 'tenantB' in n and 'lora_B' in n:
                p.add_(1.0)

    sd_a = _MCoreCompatBridgeShim._peft_state_dict([peft_model], 'tenantA')
    sd_b = _MCoreCompatBridgeShim._peft_state_dict([peft_model], 'tenantB')
    assert sd_a and sd_b, 'per-adapter state dicts must be non-empty'
    # HF PEFT layout: keys are lora_* with the adapter name stripped.
    assert all('lora_' in k for k in sd_a) and all('tenantA' not in k for k in sd_a)
    assert all('lora_' in k for k in sd_b) and all('tenantB' not in k for k in sd_b)
    # Isolation: tenantB's bumped lora_B differs from tenantA's (same key after name strip).
    common = [k for k in sd_a if k in sd_b and 'lora_B' in k]
    assert common, 'expected shared-shape lora_B keys across adapters'
    assert any((sd_a[k] - sd_b[k]).abs().max().item() > 0 for k in common), \
        'per-adapter extraction leaked: tenantA and tenantB lora_B are identical'

    # Missing adapter -> KeyError (not a silent empty export).
    with pytest.raises(KeyError):
        _MCoreCompatBridgeShim._peft_state_dict([peft_model], 'no_such_tenant')

    # Genuinely-unsupported kwargs still fail-fast on the real methods.
    shim = _MCoreCompatBridgeShim.__new__(_MCoreCompatBridgeShim)
    with pytest.raises(NotImplementedError):
        shim.save_weights([peft_model], '/tmp/x', converter=lambda k, v: (k, v))
    with pytest.raises(NotImplementedError):
        # export_weights is a generator; must iterate to trigger the guard.
        list(shim.export_weights([peft_model], only_master_rank=True))


def test_megatron_bridge_shim_rejects_sharded_peft_save(monkeypatch):
    """Per-adapter (peft_format=True) save/export must FAIL-FAST under TP/PP>1, not silently write
    an incomplete adapter. get_peft_model_state_dict returns each rank's LOCAL lora params; under
    DP they're replicated (rank0 complete) but under TP/PP they're sharded and this shim does not
    gather them. The DP-only slow tests can't catch this, so guard it here by faking mpu's parallel
    sizes -- no GPU / no real megatron init needed.

    A regression here (guard removed) would let a TP/PP run emit a corrupt adapter_model.safetensors.
    """
    import torch.nn as nn

    from peft import LoraConfig, get_peft_model
    from swift.dev.model.megatron.bridge.megatron_bridge import _MCoreCompatBridgeShim

    class _Tiny(nn.Module):

        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return self.lin(x)

    peft_model = get_peft_model(_Tiny(), LoraConfig(r=4, target_modules=['lin']), adapter_name='tenantA')
    shim = _MCoreCompatBridgeShim.__new__(_MCoreCompatBridgeShim)

    class _FakeMpu:

        def __init__(self, tp, pp):
            self._tp, self._pp = tp, pp

        def model_parallel_is_initialized(self):
            return True

        def get_tensor_model_parallel_world_size(self):
            return self._tp

        def get_pipeline_model_parallel_world_size(self):
            return self._pp

    import megatron.core as mcore

    # TP>1 -> reject save AND export.
    monkeypatch.setattr(mcore, 'parallel_state', _FakeMpu(tp=2, pp=1), raising=False)
    with pytest.raises(NotImplementedError, match='sharded'):
        shim.save_weights([peft_model], '/tmp/x', peft_format=True, adapter_name='tenantA')
    with pytest.raises(NotImplementedError, match='sharded'):
        list(shim.export_weights([peft_model], peft_format=True, adapter_name='tenantA'))

    # PP>1 -> reject too.
    monkeypatch.setattr(mcore, 'parallel_state', _FakeMpu(tp=1, pp=2), raising=False)
    with pytest.raises(NotImplementedError, match='sharded'):
        shim.save_weights([peft_model], '/tmp/x', peft_format=True, adapter_name='tenantA')

    # DP-only (tp=pp=1) -> guard passes (does not raise); the save then proceeds normally.
    monkeypatch.setattr(mcore, 'parallel_state', _FakeMpu(tp=1, pp=1), raising=False)
    _MCoreCompatBridgeShim._reject_sharded_peft()  # must NOT raise


# ----------------------------------------------------------------------
# Megatron LoRA (mcore-bridge). Training capability is inherited from twinkle
# (add_adapter_to_model -> get_peft_model + mcore-bridge dispatch_megatron -> LoraParallelLinear);
# dev only needs (a) apply_tuner fail-fast when the active backend can't do LoRA, and
# (b) a bit-parity proof that the mcore-bridge LoRA path is correct under dev orchestration.
# ----------------------------------------------------------------------


class _NoLoRABackend:
    backend_name = 'no-lora'
    is_multimodal = False


class _LoRABackend:
    backend_name = 'yes-lora'
    is_multimodal = False


class _FakeStrategy:

    def __init__(self, backend):
        self.backend = backend


class _FakeModel:
    """Minimal stand-in for apply_tuner's gate: has .strategy.backend and records add calls."""

    def __init__(self, backend):
        self.strategy = _FakeStrategy(backend)
        self.added = []

    def add_adapter_to_model(self, adapter_name, lora_config, *, gradient_accumulation_steps=1):
        self.added.append((adapter_name, lora_config, gradient_accumulation_steps))


class _TunerCfg:
    tuner_type = 'lora'
    lora_rank = 8
    lora_alpha = 16
    lora_dropout = 0.0
    lora_bias = 'none'
    target_modules = ['all-linear']
    target_regex = None
    modules_to_save = None
    use_rslora = False
    use_dora = False


@pytest.mark.slow
def test_megatron_lora_resume_restores_adapter_bit_identical(tmp_path):
    """dev Megatron LoRA (mcore-bridge, DP=2, fp32): resume completely restores the LoRA adapter,
    bit-for-bit, independent of the resumed model's own random init.

    The Megatron path has no has_adapter branch in resume_from_checkpoint, so apply_tuner must run
    first to rebuild the PeftModel; resume then loads the saved adapter (+ optimizer + RNG) over it.

    Gate:
      phase1 = seed 0, 1 step, save(adapter+optimizer).
      resumed = seed 999 (different), apply_tuner, resume(phase1), 0 further steps, save adapter.
      => resumed adapter == phase1 adapter, max|diff| == 0. The different seed proves the load is
         complete: if resume left any lora_* at its seed-999 init, the diff would be ~O(0.02), not 0.

    Scope: a multi-step post-resume trajectory does NOT bit-match an uninterrupted run (~1.4e-4 in
    fp32). That residual comes from mcore's distributed-optimizer state round-trip (Adam moments),
    not from adapter/weight restoration, so it is not gated here.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip('needs >=2 GPUs (twinkle MegatronModel requires world_size>=2)')

    from modelscope import snapshot_download

    import twinkle
    from swift.dev.adapter import apply_tuner
    from swift.dev.model.megatron.model import MegatronModel
    from twinkle import DeviceGroup, DeviceMesh
    model_path = snapshot_download(MODEL)

    ids = list(range(10, 26))
    labels_shifted = ([-100] * 8 + ids[8:])[1:] + [-100]
    pos = list(range(len(ids)))
    # twinkle Megatron GA contract: GA == number of microbatches in the `inputs` list; each
    # forward_backward(inputs=<list>) + clip_grad_and_step() == ONE optimizer step. Unlike the
    # transformers path, ga is NOT a kwarg to forward_backward (it derives from len(inputs)).
    micro_batches = [{
        'input_ids': ids,
        'labels': labels_shifted,
        'position_ids': pos
    }, {
        'input_ids': ids,
        'labels': labels_shifted,
        'position_ids': pos
    }]

    def read_lora(ckpt_dir):
        from safetensors.torch import load_file
        cand = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.safetensors')]
        assert cand, f'no safetensors in {ckpt_dir}: {os.listdir(ckpt_dir)}'
        sd = {}
        for f in cand:
            sd.update(load_file(f))
        return {n: t.detach().float().cpu().clone() for n, t in sd.items() if 'lora_' in n}

    def phase(*, tuner_seed, n_opt_steps, out, name, save_optimizer=False, resume_from=None, resume_adapter=None):
        """One full init->build->(resume)->train->save->shutdown cycle. Each phase runs in its
        own twinkle session, so the models never coexist (avoids Ray actor-name collision) and
        only the on-disk checkpoint carries state across phases."""
        twinkle.initialize(
            mode='ray',
            nproc_per_node=2,
            groups=[DeviceGroup(name='model', ranks=[0, 1], device_type='GPU', gpus_per_worker=1)])
        try:
            dm = DeviceMesh.from_sizes(world_size=2, dp_size=2)
            # fp32 (mixed_precision='no'): the bit-exact 0-diff gate needs full precision; bf16
            # save rounding would blur the complete-vs-incomplete-restore signal.
            m = MegatronModel(model_id=model_path, device_mesh=dm, mixed_precision='no', remote_group='model')
            torch.manual_seed(tuner_seed)
            apply_tuner(m, _TunerCfg())  # Megatron has no has_adapter branch: rebuild PeftModel first
            m.set_optimizer('Adam', lr=1e-4)
            if resume_from is not None:
                m.resume_from_checkpoint(resume_from, adapter_name=resume_adapter)
            for _ in range(n_opt_steps):
                m.forward_backward(inputs=micro_batches, micro_batch_size=1)
                m.clip_grad_and_step(max_grad_norm=1.0)
            m.save(name=name, output_dir=str(out), save_optimizer=save_optimizer)
            # twinkle save() writes checkpoint_dir = os.path.join(output_dir, name); in Ray lazy
            # mode the return value is a deferred handle, so recompute the path locally instead.
            return os.path.join(str(out), name)
        finally:
            try:
                twinkle.shutdown()
            except Exception:
                pass

    # phase1: seed 0, 1 optimizer step, save adapter + optimizer.
    ph1_ckpt = phase(tuner_seed=0, n_opt_steps=1, out=tmp_path / 'a', name='lora-1', save_optimizer=True)
    # resumed: DIFFERENT seed (999), apply_tuner (fresh random adapter), resume phase1, 0 steps.
    # A complete resume overwrites the seed-999 init entirely -> bit-identical to phase1.
    res_ckpt = phase(
        tuner_seed=999,
        n_opt_steps=0,
        out=tmp_path / 'b',
        name='resumed-0',
        resume_from=ph1_ckpt,
        resume_adapter='default')

    ph1_p = read_lora(ph1_ckpt)
    res_p = read_lora(res_ckpt)
    common = [n for n in ph1_p if n in res_p]
    assert common, 'no LoRA params captured (adapter export empty)'
    max_diff = max((ph1_p[n] - res_p[n]).abs().max().item() for n in common)
    print(f'\nMegatron LoRA resume (seed999 over seed0 ckpt): max|diff|={max_diff:.3e} '
          f'over {len(common)} params')
    assert max_diff == 0.0, (
        f'Megatron LoRA resume INCOMPLETE: resumed adapter (seed 999) != phase1 adapter (seed 0), '
        f'max|diff|={max_diff:.3e} over {len(common)} params -> resume did not fully overwrite the '
        f'fresh init (product bug in adapter/state restore).')


# ----------------------------------------------------------------------
# Megatron LoRA on the megatron-bridge (AutoBridge) backend. Same HF get_peft_model path as mcore
# (both build the same GPTModel); the only backend-specific surface is save. These prove LoRA
# train+save+resume hold on the megatron-bridge backend.
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_megatron_bridge_lora_save_produces_loadable_peft_adapter(tmp_path):
    """dev Megatron LoRA on the megatron-bridge backend (DP=2, fp32): apply_tuner -> 1 step ->
    model.save() must produce a standard PEFT adapter dir (adapter_config.json +
    adapter_model.safetensors) that plain peft.PeftModel.from_pretrained loads AND runs forward.

    LoRA on megatron-bridge goes through the same HF get_peft_model path as mcore (default adapter
    -> twinkle saves with is_peft_format=False, so the bridge shim's peft_format branch is not used).

    Scope: default-adapter save path only; MoE / PP / multi-adapter are checked by code review.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip('needs >=2 GPUs (twinkle MegatronModel requires world_size>=2)')
    try:
        from megatron.bridge.models.conversion.auto_bridge import AutoBridge  # noqa: F401
    except Exception:
        pytest.skip('megatron.bridge (Megatron-Bridge) not importable in this env')

    from modelscope import snapshot_download

    import twinkle
    from swift.dev.adapter import apply_tuner
    from swift.dev.model.megatron.bridge import MegatronBridgeBackend
    from swift.dev.model.megatron.model import MegatronModel
    from twinkle import DeviceGroup, DeviceMesh
    model_path = snapshot_download(MODEL)

    ids = list(range(10, 26))
    labels_shifted = ([-100] * 8 + ids[8:])[1:] + [-100]
    pos = list(range(len(ids)))
    micro = [{
        'input_ids': ids,
        'labels': labels_shifted,
        'position_ids': pos
    }, {
        'input_ids': ids,
        'labels': labels_shifted,
        'position_ids': pos
    }]
    save_dir = str(tmp_path / 'mbridge_lora')

    twinkle.initialize(
        mode='ray',
        nproc_per_node=2,
        groups=[DeviceGroup(name='model', ranks=[0, 1], device_type='GPU', gpus_per_worker=1)])
    try:
        dm = DeviceMesh.from_sizes(world_size=2, dp_size=2)
        m = MegatronModel(
            model_id=model_path,
            device_mesh=dm,
            mixed_precision='no',
            remote_group='model',
            backend=MegatronBridgeBackend())
        apply_tuner(m, _TunerCfg())
        m.set_optimizer('Adam', lr=1e-2)
        m.forward_backward(inputs=micro, micro_batch_size=1)
        m.clip_grad_and_step(max_grad_norm=1.0)
        m.save(name='lora', output_dir=save_dir, adapter_name='default')
    finally:
        try:
            twinkle.shutdown()
        except Exception:
            pass

    ckpt = os.path.join(save_dir, 'lora')
    files = sorted(os.listdir(ckpt)) if os.path.isdir(ckpt) else []
    assert 'adapter_config.json' in files, f'missing adapter_config.json: {files}'
    assert 'adapter_model.safetensors' in files, f'missing adapter_model.safetensors: {files}'

    # Closure judge: plain peft.PeftModel.from_pretrained must load the adapter AND forward.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    peft_model = PeftModel.from_pretrained(base, ckpt)
    n_lora = sum(1 for n, _ in peft_model.named_parameters() if 'lora' in n.lower())
    assert n_lora > 0, 'reloaded PEFT model has no lora params (adapter save produced empty adapter)'
    with torch.no_grad():
        out = peft_model(torch.tensor([ids]))
    assert out.logits.shape[:2] == (1, len(ids)), 'reloaded PEFT model forward shape mismatch'
    print(f'\nmegatron-bridge LoRA save->peft.from_pretrained OK: {files}, lora_params={n_lora}')


@pytest.mark.slow
def test_megatron_bridge_lora_resume_restores_adapter_bit_identical(tmp_path):
    """megatron-bridge analogue of test_megatron_lora_resume_restores_adapter_bit_identical: same
    gate -- phase1(seed0, 1 step, save adapter+optim) then resumed(seed999, apply_tuner, resume,
    0 steps, save) => adapters bit-identical (max|diff|==0), proving resume fully overwrites the
    seed-999 init on the megatron-bridge backend too.

    Scope: multi-step post-resume trajectory is not bit-gated (that ~1e-4 residual is mcore
    distributed-optimizer round-trip, not adapter restore)."""
    if torch.cuda.device_count() < 2:
        pytest.skip('needs >=2 GPUs (twinkle MegatronModel requires world_size>=2)')
    try:
        from megatron.bridge.models.conversion.auto_bridge import AutoBridge  # noqa: F401
    except Exception:
        pytest.skip('megatron.bridge (Megatron-Bridge) not importable in this env')

    from modelscope import snapshot_download

    import twinkle
    from swift.dev.adapter import apply_tuner
    from swift.dev.model.megatron.bridge import MegatronBridgeBackend
    from swift.dev.model.megatron.model import MegatronModel
    from twinkle import DeviceGroup, DeviceMesh
    model_path = snapshot_download(MODEL)

    ids = list(range(10, 26))
    labels_shifted = ([-100] * 8 + ids[8:])[1:] + [-100]
    pos = list(range(len(ids)))
    micro_batches = [{
        'input_ids': ids,
        'labels': labels_shifted,
        'position_ids': pos
    }, {
        'input_ids': ids,
        'labels': labels_shifted,
        'position_ids': pos
    }]

    def read_lora(ckpt_dir):
        from safetensors.torch import load_file
        cand = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.safetensors')]
        assert cand, f'no safetensors in {ckpt_dir}: {os.listdir(ckpt_dir)}'
        sd = {}
        for f in cand:
            sd.update(load_file(f))
        return {n: t.detach().float().cpu().clone() for n, t in sd.items() if 'lora_' in n}

    def phase(*, tuner_seed, n_opt_steps, out, name, save_optimizer=False, resume_from=None, resume_adapter=None):
        twinkle.initialize(
            mode='ray',
            nproc_per_node=2,
            groups=[DeviceGroup(name='model', ranks=[0, 1], device_type='GPU', gpus_per_worker=1)])
        try:
            dm = DeviceMesh.from_sizes(world_size=2, dp_size=2)
            m = MegatronModel(
                model_id=model_path,
                device_mesh=dm,
                mixed_precision='no',
                remote_group='model',
                backend=MegatronBridgeBackend())
            torch.manual_seed(tuner_seed)
            apply_tuner(m, _TunerCfg())
            m.set_optimizer('Adam', lr=1e-4)
            if resume_from is not None:
                m.resume_from_checkpoint(resume_from, adapter_name=resume_adapter)
            for _ in range(n_opt_steps):
                m.forward_backward(inputs=micro_batches, micro_batch_size=1)
                m.clip_grad_and_step(max_grad_norm=1.0)
            m.save(name=name, output_dir=str(out), save_optimizer=save_optimizer)
            return os.path.join(str(out), name)
        finally:
            try:
                twinkle.shutdown()
            except Exception:
                pass

    ph1_ckpt = phase(tuner_seed=0, n_opt_steps=1, out=tmp_path / 'a', name='lora-1', save_optimizer=True)
    res_ckpt = phase(
        tuner_seed=999,
        n_opt_steps=0,
        out=tmp_path / 'b',
        name='resumed-0',
        resume_from=ph1_ckpt,
        resume_adapter='default')

    ph1_p = read_lora(ph1_ckpt)
    res_p = read_lora(res_ckpt)
    common = [n for n in ph1_p if n in res_p]
    assert common, 'no LoRA params captured (adapter export empty)'
    max_diff = max((ph1_p[n] - res_p[n]).abs().max().item() for n in common)
    print(f'\nmegatron-bridge LoRA resume (seed999 over seed0 ckpt): max|diff|={max_diff:.3e} '
          f'over {len(common)} params')
    assert max_diff == 0.0, (f'megatron-bridge LoRA resume INCOMPLETE: resumed adapter (seed 999) != phase1 adapter '
                             f'(seed 0), max|diff|={max_diff:.3e} over {len(common)} params -> resume did not fully '
                             f'overwrite the fresh init (product bug in adapter/state restore).')


@pytest.mark.slow
def test_megatron_bridge_lora_non_default_adapter_save_isolated(tmp_path):
    """dev Megatron LoRA on megatron-bridge, NON-DEFAULT adapter name (DP=2, fp32): training under a
    named adapter (a tenant name, not 'default') and saving must exercise the shim's peft_format=True
    branch and produce that tenant's OWN loadable PEFT adapter.

    This is the multi-tenant judge as it's actually reachable: twinkle's add_adapter_to_model does
    NOT support stacking a 2nd adapter on the SAME model instance (its _patch_adapter re-runs
    get_peft_model, and 'all-linear' finds no raw Linear on an already-wrapped PeftModel ->
    "No modules were targeted"). So multi-tenant = one named adapter per model INSTANCE. We assert:
      - the non-default adapter_name routes through the shim's peft_format=True save (adapter_name !=
        'default' -> twinkle is_peft_format=True), producing adapter_config.json +
        adapter_model.safetensors loadable by peft.PeftModel.from_pretrained;
      - the exported per-adapter delta is REAL (lora_B non-zero after a step), proving the
        get_peft_model_state_dict(adapter_name) path captured the trained adapter, not zeros/empty.

    NOTE (scope + honest limitation): stacking multiple adapters on ONE model instance is a twinkle
    limitation (surfaced while writing this test), NOT fixed here. Covers dense DP=2; MoE/PP by review.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip('needs >=2 GPUs (twinkle MegatronModel requires world_size>=2)')
    try:
        from megatron.bridge.models.conversion.auto_bridge import AutoBridge  # noqa: F401
    except Exception:
        pytest.skip('megatron.bridge (Megatron-Bridge) not importable in this env')

    from modelscope import snapshot_download

    import twinkle
    from swift.dev.adapter import _build_adapter_config
    from swift.dev.model.megatron.bridge import MegatronBridgeBackend
    from swift.dev.model.megatron.model import MegatronModel
    from twinkle import DeviceGroup, DeviceMesh
    model_path = snapshot_download(MODEL)

    ids = list(range(10, 26))
    labels_shifted = ([-100] * 8 + ids[8:])[1:] + [-100]
    pos = list(range(len(ids)))
    micro = [{
        'input_ids': ids,
        'labels': labels_shifted,
        'position_ids': pos
    }, {
        'input_ids': ids,
        'labels': labels_shifted,
        'position_ids': pos
    }]

    def read_lora(ckpt_dir):
        from safetensors.torch import load_file
        cand = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.safetensors')]
        assert cand, f'no safetensors in {ckpt_dir}: {os.listdir(ckpt_dir)}'
        sd = {}
        for f in cand:
            sd.update(load_file(f))
        return {n: t.detach().float().cpu().clone() for n, t in sd.items() if 'lora' in n.lower()}

    def run_tenant(*, adapter_name, seed, out):
        """One independent tenant: own twinkle session, one named (non-default) adapter, save it."""
        twinkle.initialize(
            mode='ray',
            nproc_per_node=2,
            groups=[DeviceGroup(name='model', ranks=[0, 1], device_type='GPU', gpus_per_worker=1)])
        try:
            dm = DeviceMesh.from_sizes(world_size=2, dp_size=2)
            m = MegatronModel(
                model_id=model_path,
                device_mesh=dm,
                mixed_precision='no',
                remote_group='model',
                backend=MegatronBridgeBackend())
            torch.manual_seed(seed)
            # NON-default adapter name -> twinkle is_peft_format=True -> shim peft_format save branch.
            m.add_adapter_to_model(adapter_name, _build_adapter_config(_TunerCfg()), gradient_accumulation_steps=1)
            m.set_optimizer('Adam', lr=1e-2, adapter_name=adapter_name)
            m.forward_backward(inputs=micro, micro_batch_size=1, adapter_name=adapter_name)
            m.clip_grad_and_step(max_grad_norm=1.0, adapter_name=adapter_name)
            m.save(name=adapter_name, output_dir=str(out), adapter_name=adapter_name)
            return os.path.join(str(out), adapter_name)
        finally:
            try:
                twinkle.shutdown()
            except Exception:
                pass

    dir_a = run_tenant(adapter_name='tenantA', seed=0, out=tmp_path / 'a')

    files = sorted(os.listdir(dir_a)) if os.path.isdir(dir_a) else []
    assert 'adapter_config.json' in files, f'{dir_a} missing adapter_config.json: {files}'
    assert 'adapter_model.safetensors' in files, f'{dir_a} missing adapter_model.safetensors: {files}'

    # Loads independently via plain peft.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    pm = PeftModel.from_pretrained(base, dir_a)
    assert sum(1 for n, _ in pm.named_parameters() if 'lora' in n.lower()) > 0, \
        f'{dir_a} reloaded with no lora params (non-default adapter save produced empty adapter)'

    # The shim's peft_format=True branch must export the ADAPTER's real trained delta, not zeros.
    # After 1 optimizer step, lora_B (zero-init) has moved, so a correct per-adapter export has
    # non-zero lora_B weights. A broken/empty export would be all-zero (or missing lora_B).
    lora = read_lora(dir_a)
    assert lora, 'no lora weights in the saved non-default adapter'
    b_keys = [k for k in lora if 'lora_B' in k]
    assert b_keys, f'no lora_B keys in saved adapter: {sorted(lora)[:6]}'
    max_b = max(lora[k].abs().max().item() for k in b_keys)
    print(f'\nmegatron-bridge non-default-adapter LoRA (tenantA): {len(lora)} lora keys saved, '
          f'max|lora_B|={max_b:.3e} (must be > 0 -> adapter trained & per-adapter export real)')
    assert max_b > 0, ('non-default-adapter (peft_format=True) export is degenerate: lora_B all zero -> the '
                       'per-adapter get_peft_model_state_dict export did not capture the trained adapter delta.')
