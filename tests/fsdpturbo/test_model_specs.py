import importlib.util
import torch
import unittest
from torch import nn
from types import SimpleNamespace

from swift.fsdpturbo.model_specs import get_model_spec


class _Attention(nn.Module):

    def __init__(self, *, with_tp=True):
        super().__init__()
        if with_tp:
            self.q_proj = nn.Linear(2, 2, bias=False)
            self.k_proj = nn.Linear(2, 2, bias=False)
            self.v_proj = nn.Linear(2, 2, bias=False)
            self.o_proj = nn.Linear(2, 2, bias=False)


class _DecoderLayer(nn.Module):

    def __init__(self, *, with_tp=True, with_experts=True):
        super().__init__()
        self.self_attn = _Attention(with_tp=with_tp)
        self.linear_attn = nn.Module()
        self.linear_attn.A_log = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.linear_attn.norm = nn.LayerNorm(2, dtype=torch.float32)
        self.mlp = nn.Module()
        if with_experts:
            self.mlp.experts = nn.ModuleList([nn.Linear(2, 2, bias=False)])


class _SyntheticQwenMoe(nn.Module):

    def __init__(self, *, with_tp=True, with_experts=True):
        super().__init__()
        self.config = SimpleNamespace(model_type='qwen3_5_moe')
        self.model = nn.Module()
        self.model.visual = nn.Module()
        self.model.visual.blocks = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        self.model.visual.patch_embed = nn.Linear(2, 2, bias=False)
        self.model.visual.pos_embed = nn.Embedding(2, 2)
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([_DecoderLayer(with_tp=with_tp, with_experts=with_experts)])
        self.model.language_model.embed_tokens = nn.Embedding(2, 2)
        self.lm_head = nn.Linear(2, 2, bias=False)


@unittest.skipUnless(importlib.util.find_spec('fsdp_turbo'), 'requires the optional FSDPTurbo package')
class TestFSDPTurboModelSpecs(unittest.TestCase):

    def test_qwen_moe_patterns_match_exact_module_boundaries(self):
        model = _SyntheticQwenMoe()
        matches = get_model_spec(model).validate_model(model, require_tp=True, require_ep=True, require_recompute=True)
        expected_counts = {
            'fsdp_modules': 6,
            'fsdp_hook_modules': 1,
            'tp_colwise_modules': 3,
            'tp_rowwise_modules': 1,
            'ep_modules': 1,
            'efsdp_modules': 1,
            'recompute_modules': 2,
        }

        for group, count in expected_counts.items():
            with self.subTest(group=group):
                self.assertEqual(sum(len(names) for names in matches[group].values()), count)

        fsdp_names = {name for names in matches['fsdp_modules'].values() for name in names}
        expert_names = {name for names in matches['efsdp_modules'].values() for name in names}
        self.assertTrue(fsdp_names.isdisjoint(expert_names))

    def test_optional_parallel_groups_are_validated_only_when_requested(self):
        model = _SyntheticQwenMoe(with_tp=False, with_experts=False)
        spec = get_model_spec(model)

        matches = spec.validate_model(model)
        self.assertEqual(set(matches), {'fsdp_modules', 'fsdp_hook_modules'})
        with self.assertRaises(RuntimeError):
            spec.validate_model(model, require_tp=True)
        with self.assertRaises(RuntimeError):
            spec.validate_model(model, require_ep=True)

    def test_each_required_fsdp_pattern_must_match(self):
        model = _SyntheticQwenMoe()
        del model.lm_head

        with self.assertRaises(RuntimeError):
            get_model_spec(model).validate_model(model)


class TestFSDPTurboModelScope(unittest.TestCase):

    def test_old_deepseek_v4_has_no_backend_plan(self):
        with self.assertRaisesRegex(ValueError, 'No FSDPTurbo model spec'):
            get_model_spec('deepseek_v4')


if __name__ == '__main__':
    unittest.main()
