# Copyright (c) ModelScope Contributors. All rights reserved.
"""S19a/b — weight_sync quantization kernels + pre-flight guards + re-quantization (L1).

The quantization-aware rollout refit's numeric half: the blockwise BF16->FP8 kernel (triton and its
PyTorch fallback must agree bit-for-bit), the MXFP4 E2M1 packer (two nibbles per int8 byte, the
dequantized grid landing exactly on the E2M1 value set), the pre-flight guards refusing the
engine/training combinations the refit cannot serve, and ``requantize_checkpoint`` rewriting a real
on-disk BF16 export back into the source's packed format (read back from the written artifact).

Maps to ut_plan S19-01..09 (the vLLM refit itself, S19-12..15, needs a live quantized engine and
checkpoint -- environment-gated, not written here).
"""
import json
import os

import pytest
import torch
from safetensors.torch import load_file, save_file

from swift.dev.weight_sync import fp8_kernel as FK
from swift.dev.weight_sync import mxfp4_kernel as MX
from swift.dev.weight_sync.checks import check_quantized_weight_sync

BLOCK = (16, 16)


class TestFP8Kernel:

    def test_constant_block_round_trips_exactly(self):
        """S19-01: a constant 2.0 block quantizes to q*scale == 2.0 (the scale absorbs the
        magnitude, the fp8 value rides at the top of its range)."""
        w = torch.full((16, 16), 2.0, dtype=torch.bfloat16)
        q, s = FK.scaled_fp8_blockwise(w, BLOCK)
        assert q.dtype == torch.float8_e4m3fn
        assert float(q[0, 0]) == FK.FP8_MAX  # the max element rides at FP8_MAX
        assert float(q[0, 0]) * float(s[0, 0]) == pytest.approx(2.0, rel=1e-3)

    def test_saturation_is_lossless_for_the_block_max(self):
        """S19-03: a block of 1792 (= 4*448) saturates the fp8 value but the per-block scale
        carries it -- dequantize gives back exactly 1792."""
        w = torch.full((16, 16), 4 * FK.FP8_MAX, dtype=torch.bfloat16)
        q, s = FK.scaled_fp8_blockwise(w, BLOCK)
        assert float(q[0, 0]) * float(s[0, 0]) == pytest.approx(1792.0, rel=1e-3)

    def test_triton_and_pytorch_paths_agree(self):
        """S19-01/06: forcing the PyTorch fallback (_DISABLE_TRITON_FP8) must produce the SAME
        quantized bytes and scales as the triton path -- a silent divergence would make the refit
        non-deterministic across machines."""
        x = torch.randn(64, 64, dtype=torch.bfloat16)
        q_t, s_t = FK.scaled_fp8_blockwise(x, BLOCK)
        FK._DISABLE_TRITON_FP8 = True
        try:
            q_p, s_p = FK.scaled_fp8_blockwise(x, BLOCK)
        finally:
            FK._DISABLE_TRITON_FP8 = False
        assert torch.equal(q_t, q_p)
        assert torch.allclose(s_t, s_p)

    def test_ceil_div(self):
        """S19-02: block-count ceiling for non-multiple sizes."""
        assert FK.ceil_div(7, 3) == 3
        assert FK.ceil_div(6, 3) == 2

    def test_fp8_constants(self):
        """S19-03: the e4m3 value range."""
        assert FK.FP8_DTYPE == torch.float8_e4m3fn
        assert FK.FP8_MAX == 448.0 and FK.FP8_MIN == -448.0


class TestMXFP4Kernel:

    def test_packing_two_nibbles_per_byte(self):
        """S19-04/05: [R, 32] bf16 packs to [R, 16] int8 (two E2M1 nibbles per byte) with a
        per-row [R, 1] scale (MXFP4_BLOCK_SIZE=32 elements per scale)."""
        x = torch.randn(4, 32, dtype=torch.bfloat16)
        scale = torch.ones(4, 1, dtype=torch.bfloat16)
        packed, out_scale = MX.quantize_mxfp4_e2m1(x, scale)
        assert packed.shape == (4, 16) and packed.dtype == torch.int8
        assert out_scale.shape == (4, 1)
        assert MX.MXFP4_BLOCK_SIZE == 32

    def test_dequantized_values_land_on_the_e2m1_grid(self):
        """S19-04: dequantized / scale is exactly an E2M1 codepoint (nearest-neighbor on the
        {0, .5, 1, 1.5, 2, 3, 4, 6} x sign grid) -- a value off the grid is a packing bug. The
        output scale is recomputed per block as a power of two from the amax (the source scale
        fixes only geometry + dtype), so the oracle normalizes by it first."""
        x = torch.randn(8, 32, dtype=torch.bfloat16)
        packed, out_scale = MX.quantize_mxfp4_e2m1(x, torch.ones(8, 1, dtype=torch.bfloat16))
        deq = MX.dequantize_mxfp4_e2m1(packed, out_scale).float()
        normalized = deq / out_scale.float()
        grid = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
        for value in normalized.abs().flatten().tolist():
            assert min(abs(value - g) for g in grid) < 1e-3, f'{value} is off the E2M1 grid'

    def test_output_scale_is_a_power_of_two(self):
        """S19-04: the checkpoint declares ue8m0 scales -- only exact powers of two survive the
        e8m0 cast, so the kernel rounds the block scale up to one (never a perturbing rounding)."""
        x = torch.randn(4, 32, dtype=torch.bfloat16)
        _, out_scale = MX.quantize_mxfp4_e2m1(x, torch.ones(4, 1, dtype=torch.bfloat16))
        exponents = torch.log2(out_scale.float()).flatten().tolist()
        for e in exponents:
            assert abs(e - round(e)) < 1e-3, f'scale exponent {e} is not integral'

    def test_constant_block_hand_computed_oracle(self):
        """S19-04: an independent hand-computed oracle, not a quantize->dequantize round trip.
        A constant 0.9 block: amax/6 = 0.15, ceil(log2(0.15)) = -2 so scale = 0.25; the
        normalized 0.9/0.25 = 3.6 sits between the 3.5 and 5.0 bucket edges so every element
        takes code 6 (value 4.0) -> each byte packs two 6-nibbles = 0x66, and dequantizing
        gives 4.0 * 0.25 = 1.0 exactly."""
        x = torch.full((1, 32), 0.9, dtype=torch.bfloat16)
        packed, out_scale = MX.quantize_mxfp4_e2m1(x, torch.ones(1, 1, dtype=torch.bfloat16))
        assert packed.view(torch.uint8).tolist() == [[0x66] * 16]
        assert out_scale.item() == 0.25
        assert MX.dequantize_mxfp4_e2m1(packed, out_scale).unique().tolist() == [1.0]

    def test_geometry_guards(self):
        """S19-05: an odd column count or a mis-shaped scale is rejected with the geometry named
        (a silent reshape would pack the wrong nibbles together)."""
        with pytest.raises(RuntimeError, match='geometry'):
            MX.quantize_mxfp4_e2m1(torch.randn(2, 31, dtype=torch.bfloat16),
                                   torch.ones(2, 1, dtype=torch.bfloat16))
        with pytest.raises(RuntimeError, match='geometry'):
            MX.quantize_mxfp4_e2m1(torch.randn(4, 32, dtype=torch.bfloat16),
                                   torch.ones(3, 1, dtype=torch.bfloat16))


class _Cfg:

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _vllm():
    return _Cfg(rollout_sampler='vllm')


class TestCheckQuantizedWeightSync:
    """S19-07: the pre-flight guards -- safe to call unconditionally, refusing exactly the
    combinations the quantized refit cannot serve."""

    def test_non_vllm_backend_is_noop(self):
        check_quantized_weight_sync(_Cfg(model_type='deepseek_v4'), None,
                                    _Cfg(tuner='full'), _Cfg(rollout_sampler='sglang'))

    def test_non_deepseek_model_is_noop(self):
        check_quantized_weight_sync(_Cfg(model_type='qwen2'), None,
                                    _Cfg(tuner='full'), _vllm())

    def test_fp4_experts_with_sleep_level_2_rejected(self):
        """sleep_level=2 discards the rollout engine's parameters; the MXFP4 experts would wake as
        uninitialized memory and the rollout would emit garbage from the first token."""
        with pytest.raises(NotImplementedError, match='sleep_level=2'):
            check_quantized_weight_sync(_Cfg(model_type='deepseek_v4', expert_dtype='fp4'), None,
                                        _Cfg(tuner='full'), _Cfg(rollout_sampler='vllm', sleep_level=2))

    def test_sleep_level_1_is_fine(self):
        check_quantized_weight_sync(_Cfg(model_type='deepseek_v4', expert_dtype='fp4'), None,
                                    _Cfg(tuner='full'), _Cfg(rollout_sampler='vllm', sleep_level=1))

    def test_lora_on_routed_experts_of_quantized_engine_rejected(self):
        """A merged LoRA delta is far below the MXFP4 grid step -- re-packing would replace the
        adapter update with quantization noise."""
        hf = _Cfg(model_type='deepseek_v4',
                  quantization_config=_Cfg(quant_method='mxfp4'))
        with pytest.raises(NotImplementedError, match='routed experts'):
            check_quantized_weight_sync(hf, None, _Cfg(tuner='lora', target_modules=['all-linear']),
                                        _vllm())

    def test_lora_off_the_experts_is_fine(self):
        hf = _Cfg(model_type='deepseek_v4',
                  quantization_config=_Cfg(quant_method='mxfp4'))
        check_quantized_weight_sync(hf, None, _Cfg(tuner='lora', target_modules=['o_proj']), _vllm())

    def test_full_precision_engine_with_expert_lora_is_fine(self):
        """A BF16 engine loads the adapter stream as-is -- an adapter on the experts is fine there."""
        check_quantized_weight_sync(_Cfg(model_type='deepseek_v4'), None,
                                    _Cfg(tuner='lora', target_modules=['all-linear']), _vllm())


class TestRequantizeCheckpoint:
    """S19-09: rewrite a real on-disk BF16 export back into the source's packed format."""

    def _write(self, root, tensors, expert_dtype='fp4'):
        os.makedirs(root, exist_ok=True)
        save_file(tensors, os.path.join(root, 'model.safetensors'))
        with open(os.path.join(root, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump({'expert_dtype': expert_dtype}, f)

    def test_bf16_export_repacks_to_source_format(self, tmp_path):
        key = 'model.ffn.experts.0.weight'
        w = torch.randn(4, 32, dtype=torch.bfloat16)
        packed, out_scale = MX.quantize_mxfp4_e2m1(w, torch.ones(4, 1, dtype=torch.bfloat16))
        src = tmp_path / 'src'
        self._write(str(src), {key: packed, key.replace('.weight', '.scale'): out_scale})
        ckpt = tmp_path / 'ckpt'
        self._write(str(ckpt), {key: w})

        counts = requantize_checkpoint(str(ckpt), str(src))
        assert counts['mxfp4'] == 1 and counts['fp8'] == 0 and counts['kept'] == 0

        # read back from the WRITTEN artifact: the bf16 export became the packed form, byte-equal
        # to packing the same bf16 weights directly.
        re = load_file(str(ckpt / 'model.safetensors'))
        assert sorted(re) == sorted([key, key.replace('.weight', '.scale')])
        assert torch.equal(re[key], packed)
        assert torch.equal(re[key.replace('.weight', '.scale')], out_scale)

    def test_non_packed_source_returns_none(self, tmp_path):
        plain = tmp_path / 'plain'
        self._write(str(plain), {'a.weight': torch.randn(2, 2)})
        ckpt = tmp_path / 'ckpt'
        self._write(str(ckpt), {'a.weight': torch.randn(2, 2)})
        assert requantize_checkpoint(str(ckpt), str(plain)) is None


def requantize_checkpoint(*args, **kwargs):
    from swift.dev.weight_sync.requantize import requantize_checkpoint as fn

    return fn(*args, **kwargs)


class TestIsQuantizedSyncModel:
    """S19-08: whether weight sync into this engine has to go through quantization."""

    def test_none_config_is_not_quantized(self):
        from swift.dev.weight_sync.manager import is_quantized_sync_model

        assert is_quantized_sync_model(None) is False

    def test_deepseek_v4_hf_config_forces_quantized(self):
        """The MXFP4 refit is keyed off model_type, not just an Fp8Config -- a DeepSeek-V4
        engine without a block-FP8 quant_config still needs quantized sync."""
        from swift.dev.weight_sync.manager import is_quantized_sync_model

        cfg = _Cfg(quant_config=None,
                   model_config=_Cfg(hf_config=_Cfg(model_type='deepseek_v4')))
        assert is_quantized_sync_model(cfg) is True

    def test_plain_model_is_not_quantized(self):
        from swift.dev.weight_sync.manager import is_quantized_sync_model

        cfg = _Cfg(quant_config=None,
                   model_config=_Cfg(hf_config=_Cfg(model_type='qwen2')))
        assert is_quantized_sync_model(cfg) is False


class _Experts(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.float8_e4m3fn),
                                             requires_grad=False)
        self.w2_weight = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.float8_e4m3fn),
                                            requires_grad=False)


class _MoERunner(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.routed_experts = _Experts()


class _Layer(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mlp = _MoERunner()


class _MoEModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer()])


class TestParamNameResolution:
    """S19-10: HF name -> vLLM module resolution, via the model's own mapper and the reverse
    packed-modules mapping."""

    def test_walk_stops_at_the_fused_moe_block(self):
        """S19-11: traversal stops at the fused-MoE block and resolves to the module that owns
        the expert weights -- descending into the expert index would fail (it is not a real
        submodule; vLLM's fused weight loader consumes it)."""
        from swift.dev.weight_sync.manager import _walk_to_module

        model = _MoEModel()
        for name in ('layers.0.mlp.experts.3.w1.weight', 'layers.0.mlp.experts.7.w2.weight'):
            assert _walk_to_module(model, name) is model.layers[0].mlp.routed_experts

    def test_expert_module_duck_typing(self):
        """S19-11: expert-weight modules are recognized by the concrete classes when importable
        and by w13_weight/w2_weight duck typing otherwise."""
        from swift.dev.weight_sync.manager import _is_expert_weight_module, _is_fused_moe_block

        model = _MoEModel()
        experts, runner = model.layers[0].mlp.routed_experts, model.layers[0].mlp
        assert _is_expert_weight_module(experts)
        assert _is_fused_moe_block(runner)  # MoERunner: stops via routed_experts
        assert _is_fused_moe_block(experts)  # pre-0.24 FusedMoE shape: stops too
        assert not _is_expert_weight_module(torch.nn.Linear(2, 2))

    def test_packed_module_name_is_reversed(self):
        """``q_proj`` lives on ``qkv_proj`` in vLLM: the walk reaches it through the reverse
        of packed_modules_mapping."""
        from swift.dev.weight_sync.manager import _walk_to_module

        class _Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.qkv_proj = torch.nn.Linear(2, 2)

        _Model.packed_modules_mapping = {'qkv_proj': ['q_proj', 'k_proj', 'v_proj']}
        model = _Model()
        assert _walk_to_module(model, 'q_proj.weight') is model.qkv_proj

    def test_unresolvable_name_raises_key_error(self):
        """The NaN-logprobs regression anchor: a name that resolves to nothing must raise, not
        answer silently -- treating it as un-quantized would stream BF16 into a staged FP8
        parameter."""
        from swift.dev.weight_sync.manager import get_module_from_param_name

        with pytest.raises(KeyError, match='Refusing to guess'):
            get_module_from_param_name(_MoEModel(), 'layers.0.nothere.weight')

    def test_mapper_none_and_exception_fall_back(self):
        """A model without a mapper passes the name through; a mapper raising falls back to
        the raw name (private API); a mapper dropping the name means the engine never loads
        that weight, so resolution answers None."""
        from swift.dev.weight_sync.manager import _to_vllm_param_name, get_module_from_param_name

        class _Mapper:

            def _map_name(self, name):
                return None if 'drop' in name else name.replace('head.', 'lm_head.')

        class _Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.lm_head = torch.nn.Linear(2, 2)

        _Model.hf_to_vllm_mapper = _Mapper()
        model = _Model()
        assert _to_vllm_param_name(model, 'head.weight') == 'lm_head.weight'
        assert get_module_from_param_name(model, 'model.drop.weight') is None

        class _BadMapper:

            def _map_name(self, name):
                raise RuntimeError('boom')

        _Model.hf_to_vllm_mapper = _BadMapper()
        assert _to_vllm_param_name(model, 'fc.weight') == 'fc.weight'

        class _Plain(torch.nn.Module):
            pass

        assert _to_vllm_param_name(_Plain(), 'x.weight') == 'x.weight'


class TestIsFp8Weight:
    """S19-11: FP8 candidacy decided from the owning module's actual parameter dtype, with a
    leaf-name pre-screen and a per-name cache."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        from swift.dev.weight_sync import manager as M

        M._quant_state.seen_params.clear()
        M._quant_state.fp8_param_names.clear()
        yield
        M._quant_state.seen_params.clear()
        M._quant_state.fp8_param_names.clear()

    @staticmethod
    def _model_with(dtype):
        pytest.importorskip('vllm')
        from vllm.model_executor.layers.linear import LinearBase

        class _Fake(LinearBase):
            pass

        linear = object.__new__(_Fake)  # skip __init__: only the dtype is under test
        linear.weight = torch.zeros(2, 2, dtype=dtype)

        class _Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.fc = linear

        return _Model()

    def test_fp8_linear_is_candidate_and_cached(self):
        from swift.dev.weight_sync import manager as M

        model = self._model_with(torch.float8_e4m3fn)
        assert M.is_fp8_weight('fc.weight', model) is True
        assert 'fc.weight' in M._quant_state.fp8_param_names

    def test_bf16_linear_is_not(self):
        from swift.dev.weight_sync.manager import is_fp8_weight

        assert is_fp8_weight('fc.weight', self._model_with(torch.bfloat16)) is False

    def test_bias_leaf_is_prescreened_out(self):
        """The cheap pre-screen answers without any module walk: only weight leaves and the
        packed gate_up_proj/down_proj names can be FP8."""
        from swift.dev.weight_sync.manager import is_fp8_weight

        model = self._model_with(torch.float8_e4m3fn)
        assert is_fp8_weight('fc.bias', model) is False
        assert is_fp8_weight('fc.weight_scale_inv', model) is False

    def test_packed_projection_leaf_passes_the_prescreen(self):
        """A packed fused-MoE projection leaf (no .weight suffix) is not prescreened out; it
        walks to its module and the dtype decides (a bf16 module answers False)."""
        from swift.dev.weight_sync.manager import is_fp8_weight

        class _Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.mlp = torch.nn.Linear(2, 2, dtype=torch.bfloat16)

        assert is_fp8_weight('mlp.gate_up_proj', _Model()) is False
