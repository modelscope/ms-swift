# Copyright (c) ModelScope Contributors. All rights reserved.
"""Key models x capability combinations, on 4-layer models shrunk from each family's real config.

Two tables and one policy function. The tables say what exists; ``CombinationPolicy.for_model`` says
which combinations get run, and it is deliberately the only place that decision lives -- the policy
is expected to change, the tables and the runner are not.

Why combinations do not explode: a slot has two GPUs, so ``tp * pp * cp * ep * dp <= 2`` and at most
ONE parallel dimension can be enabled at a time. That collapses the five parallel switches from 2^5
to the six ``PARALLEL`` entries below, leaving only the orthogonal switches to cross-product.

Assertions are about availability, not numbers: random weights make loss values meaningless, so a
combination passes when it completes its steps with finite loss and writes a checkpoint. Numerical
claims live in ``feature/sft/test_alignment.py``, which needs real weights.
"""
import os
import sys
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, FrozenSet, List, Tuple

import json
import pytest

from swift.dev.tests._runners import Runners
from swift.dev.tests.tiny import TinyData, TinyModel
from swift.dev.tests.tiny_loader import loader_builder


@dataclass(frozen=True)
class ModelSpec:
    """One key model: how to build it tiny, and which capabilities apply to it.

    ``model_type`` does double duty: it is what the run is given, and it is how the tiny checkpoint is
    built -- ``loader_builder`` inherits that family's registered loader, so this table never names a
    transformers config or model class. It has to be a name BOTH registries know (``qwen2``, not
    ``qwen2_5``): ``run_sft`` still asks legacy swift for the processor, and legacy lumps all of Qwen2.5
    under ``qwen2``. ``model_id`` is the real checkpoint the tokenizer and the starting ``config.json``
    are snapshotted from (config/tokenizer files only, never weights), which is what keeps the MoE
    geometry below a shrink of a real one rather than an invention.
    """

    name: str
    model_type: str
    model_id: str
    template: str
    supports: FrozenSet[str]
    extra: Dict[str, object] = field(default_factory=dict)


#: Exactly one of these may be on at a time -- their product must stay within a two-GPU slot.
PARALLEL: Dict[str, Dict[str, object]] = {
    'dp2': {},
    'tp2': {
        'tensor_model_parallel_size': 2
    },
    'pp2': {
        'pipeline_model_parallel_size': 2
    },
    'cp2': {
        'context_parallel_size': 2
    },
    'ep2': {
        'expert_model_parallel_size': 2
    },
    'sp': {
        'tensor_model_parallel_size': 2,
        'sequence_parallel': True
    },
}

#: Free to combine with each other and with one PARALLEL entry.
ORTHOGONAL: Dict[str, Dict[str, object]] = {
    'packing': {
        'packing': True
    },
    'padding_free': {
        'padding_free': True
    },
    'mtp': {
        'mtp_num_layers': 1
    },
}

#: Which Config each switch writes to, so a switch can be turned into config overrides generically.
TARGET = {
    'tensor_model_parallel_size': 'distributed',
    'pipeline_model_parallel_size': 'distributed',
    'context_parallel_size': 'distributed',
    'expert_model_parallel_size': 'distributed',
    'sequence_parallel': 'distributed',
    'packing': 'dataset',
    'padding_free': 'template',
    'mtp_num_layers': 'model',
}

MODELS = [
    ModelSpec(
        name='qwen2-dense',
        model_type='qwen2',
        model_id='Qwen/Qwen2.5-0.5B-Instruct',
        template='qwen2_5',
        supports=frozenset({'dp2', 'tp2', 'pp2', 'cp2', 'sp', 'packing', 'padding_free'}),
    ),
    ModelSpec(
        name='qwen3-moe',
        model_type='qwen3_moe',
        model_id='Qwen/Qwen3-30B-A3B',
        template='qwen3',
        supports=frozenset({'dp2', 'tp2', 'pp2', 'ep2', 'packing', 'padding_free'}),
        extra={
            'num_experts': 4,
            'num_experts_per_tok': 2,
            'moe_intermediate_size': 128
        },
    ),
    # qwen3 dense: the non-MoE Qwen3 line (thinking template), same parallel surface as qwen2-dense.
    ModelSpec(
        name='qwen3-dense',
        model_type='qwen3',
        model_id='Qwen/Qwen3-0.6B-Base',
        template='qwen3',
        supports=frozenset({'dp2', 'tp2', 'pp2', 'cp2', 'sp', 'packing', 'padding_free'}),
    ),
    # glm4: the GLM family's dense line.
    ModelSpec(
        name='glm4-dense',
        model_type='glm4',
        model_id='ZhipuAI/GLM-4-9B-0414',
        template='glm4',
        supports=frozenset({'dp2', 'tp2', 'pp2', 'cp2', 'sp', 'packing', 'padding_free'}),
    ),
    # deepseek-v2.5 MoE: MLA + routed experts shrunk from the real geometry. REGISTERED GAP
    # (upstream): transformers 5.20.1's DeepseekV2Moe references self.num_experts on the
    # group_limited_greedy path without assigning it, and the shrunk MLA geometry still trips a
    # shape mismatch inside sdpa on the DP2 training path (single-process forward is fine). The
    # spec stays in the table (marked xfail below) so the day upstream fixes land, these turn
    # red-for-real and the grid runs. deepseek_v3 needs the `kernels` fp8 package on this box.
    ModelSpec(
        name='deepseek-moe',
        model_type='deepseek_v2_5',
        model_id='deepseek-ai/DeepSeek-V2.5',
        template='deepseek_v2_5',
        supports=frozenset({'dp2', 'tp2', 'pp2', 'ep2', 'packing', 'padding_free'}),
        # MLA heads do not shrink like GQA's: kv projections are per-head-rank in MLA geometry,
        # so the whole MLA block is shrunk together (q_lora/kv_lora ranks and head dims), not by
        # the generic TinyModel.DIMS (num_key_value_heads=2 against MLA head dims is a shape
        # mismatch inside sdpa attention). topk_method stays 'greedy': the real V2.5 value
        # ('group_limited_greedy') hits an upstream transformers bug on the shrunk geometry
        # (DeepseekV2Moe.num_experts is referenced but never assigned on that path).
        extra={
            'hidden_size': 256,
            'num_attention_heads': 4,
            'num_key_value_heads': 128,
            'kv_lora_rank': 32,
            'q_lora_rank': 64,
            'qk_nope_head_dim': 16,
            'qk_rope_head_dim': 8,
            'v_head_dim': 16,
            'n_routed_experts': 4,
            'n_shared_experts': 1,
            'num_experts_per_tok': 2,
            'moe_intermediate_size': 128,
            'first_k_dense_replace': 1,
            'topk_method': 'greedy',
        },
    ),
    # minicpm-v 2.6: the MiniCPM vision line (vision tower + adaptor). REGISTERED PRODUCT GAP:
    # the family's model class lives in the checkpoint's auto_map (remote code), and dev's
    # model_cls import resolves `transformers:MiniCPMV` as a transformers attribute only, so the
    # run dies before the first step. Strict-xfail until remote-code model_cls resolution consumes
    # the config's auto_map.
    ModelSpec(
        name='minicpm-v',
        model_type='minicpmv2_6',
        model_id='openbmb/MiniCPM-V-2_6',
        template='minicpmv2_6',
        supports=frozenset({'dp2', 'packing'}),
    ),
    # internvl3 (HF layout): vision tower via the merged "internvl_hf" path. REGISTERED PRODUCT
    # GAP: on a text-only SFT batch the vision tower contributes nothing to the loss and DDP
    # refuses the unused parameters (find_unused_parameters is not enabled for this family), so
    # the grid is strict-xfail until that knob (or tower freezing) is wired for internvl.
    ModelSpec(
        name='internvl',
        model_type='internvl',
        model_id='OpenGVLab/InternVL3-1B-hf',
        template='internvl_hf',
        supports=frozenset({'dp2', 'packing'}),
    ),
    # qwen2.5-VL: the mainstream vision-language line.
    ModelSpec(
        name='qwen2-5-vl',
        model_type='qwen2_5_vl',
        model_id='Qwen/Qwen2.5-VL-7B-Instruct',
        template='qwen2_5_vl',
        supports=frozenset({'dp2', 'packing'}),
    ),
]


class CombinationPolicy:
    """The one place that decides how much of the matrix to run."""

    #: Two GPUs per slot, so the parallel dimensions multiply to at most this.
    GPUS = 2
    #: Cap on how many orthogonal switches may be on together. 1 = singles + pairs with a parallel
    #: dimension; raise it to widen coverage, set it to len(ORTHOGONAL) for the full cross product.
    MAX_ORTHOGONAL = 1

    @staticmethod
    def for_model(spec: ModelSpec) -> List[Tuple[str, ...]]:
        """Every combination to run for ``spec``, as tuples of switch names."""
        parallels = [name for name in PARALLEL if name in spec.supports]
        orthogonals = [name for name in ORTHOGONAL if name in spec.supports]

        widths = range(0, min(CombinationPolicy.MAX_ORTHOGONAL, len(orthogonals)) + 1)
        extras = [combo for width in widths for combo in combinations(orthogonals, width)]

        out = []
        for parallel in parallels:
            for extra in extras:
                if 'packing' in extra and 'padding_free' in extra:
                    continue  # mutually exclusive: packing already concatenates without padding
                out.append((parallel, ) + extra)
        return out


class CapabilityRun:
    """Launch one combination as a fresh two-process torchrun and read its verdict."""

    STEPS = 2

    @staticmethod
    def overrides(switches: Tuple[str, ...]) -> Dict[str, Dict[str, object]]:
        """Fold switch names into per-Config override dicts."""
        buckets: Dict[str, Dict[str, object]] = {'model': {}, 'template': {}, 'dataset': {}, 'distributed': {}}
        for switch in switches:
            for field_name, value in {**PARALLEL, **ORTHOGONAL}[switch].items():
                buckets[TARGET[field_name]][field_name] = value
        return buckets

    @staticmethod
    def launch(spec: ModelSpec, switches: Tuple[str, ...], tmp_path) -> List[float]:
        from swift.dev.model.loader import resolve_loader_cls
        if getattr(resolve_loader_cls(spec.model_type, None), 'is_multimodal', False):
            # A VL checkpoint survives neither the dense DIMS nor an AutoTokenizer-only save: the
            # tower/merger geometry is wired to the text sizes, so only the depth shrinks and the
            # full processor (image processor included) is saved beside the weights.
            from swift.dev.tests.tiny_loader import build_tiny_multimodal
            model_dir = build_tiny_multimodal(
                tmp_path / 'model', model_type=spec.model_type, model_id=spec.model_id)
        else:
            model_dir = TinyModel.build(
                tmp_path / 'model', tokenizer_id=spec.model_id, builder=loader_builder(spec.model_type),
                **spec.extra)
        data_path = TinyData.sft(tmp_path / 'data.jsonl')
        result_path = str(tmp_path / 'result.json')
        buckets = CapabilityRun.overrides(switches)

        payload = {
            'model': {
                'model': model_dir,
                'model_type': spec.model_type,
                'torch_dtype': 'bfloat16',
                **buckets['model']
            },
            'template': {
                'template': spec.template,
                'max_length': 256,
                **buckets['template']
            },
            'dataset': {
                'dataset': [data_path],
                **buckets['dataset']
            },
            'train': {
                'learning_rate': 1e-4,
                'per_device_train_batch_size': 2,
                'max_steps': CapabilityRun.STEPS
            },
            'distributed': buckets['distributed'],
            'output_dir': str(tmp_path / 'out'),
            'result_path': result_path,
        }

        cmd = [
            sys.executable, '-m', 'torch.distributed.run', f'--nproc_per_node={CombinationPolicy.GPUS}',
            *Runners.RENDEZVOUS,
            Runners.path('capability'),
            json.dumps(payload)
        ]
        proc = Runners.launch(cmd)
        assert proc.returncode == 0, \
            f'torchrun failed ({proc.returncode}):\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}'
        assert os.path.exists(result_path), f'runner wrote no result:\n{proc.stdout[-4000:]}'
        with open(result_path) as f:
            return json.load(f)['losses']


#: Specs whose grid is a REGISTERED GAP (upstream transformers bugs on the shrunk geometry);
#: strict xfail so a fix turning them green also turns this marker red and forces a revisit.
_XFAIL_SPECS = frozenset({'deepseek-moe', 'internvl', 'minicpm-v'})


def _case_id(case):
    spec, switches = case
    return f'{spec.name}-{"+".join(switches)}'


def _case_marks(case):
    spec, _ = case
    if spec.name in _XFAIL_SPECS:
        return [pytest.mark.xfail(
            reason='upstream transformers DeepseekV2Moe bugs on shrunk MLA/MoE geometry', strict=True)]
    return []


_CASES = [(spec, switches) for spec in MODELS for switches in CombinationPolicy.for_model(spec)]
CASES = [pytest.param(spec, switches, id=_case_id((spec, switches)), marks=_case_marks((spec, switches)))
         for spec, switches in _CASES]


@pytest.mark.slow
@pytest.mark.accel(2)
@pytest.mark.parametrize('spec,switches', CASES)
def test_combination_trains_and_saves(spec, switches, tmp_path):
    """The combination must complete its steps with finite loss and leave a checkpoint behind."""
    losses = CapabilityRun.launch(spec, switches, tmp_path)

    assert len(losses) == CapabilityRun.STEPS, f'expected {CapabilityRun.STEPS} steps, got {losses}'
    assert all(loss == loss and abs(loss) != float('inf') for loss in losses), f'non-finite loss: {losses}'
    ckpt = tmp_path / 'out' / 'checkpoint-final'
    assert ckpt.is_dir(), f'no checkpoint-final under {tmp_path / "out"}'


# --- L1: the tables and the policy, checked without any GPU --------------------------------


def test_every_spec_builds_and_supports_only_known_switches():
    """S14-11: each table row resolves to its family loader (a dead model_type key would make the
    whole row a silent no-op) and its supports set stays inside the known switch universe."""
    from swift.dev.model.loader import resolve_loader_cls

    for spec in MODELS:
        loader_cls = resolve_loader_cls(spec.model_type, None)
        assert loader_cls is not None, f'{spec.name}: no loader registered for {spec.model_type!r}'
        assert getattr(loader_cls, 'template', None) is not None
        unknown = spec.supports - set(PARALLEL) - set(ORTHOGONAL)
        assert not unknown, f'{spec.name}: supports names unknown switches {sorted(unknown)}'
        # a MoE-only switch (ep2) needs a MoE spec, and VL specs never carry mtp
        if 'ep2' in spec.supports:
            assert getattr(loader_cls, 'is_moe', False) or spec.extra.get('num_experts'), \
                f'{spec.name}: claims ep2 but is not a MoE'
        if getattr(loader_cls, 'is_multimodal', False):
            assert 'mtp' not in spec.supports, f'{spec.name}: VL spec must not carry mtp'


def test_combination_policy_generates_the_expected_grid():
    """S14-12: the policy's grid, on a spec that supports everything: every parallel dimension x
    every orthogonal width up to MAX_ORTHOGONAL, minus the packing/padding_free mutual exclusion."""
    spec = ModelSpec(name='all', model_type='qwen2', model_id='m', template='t',
                     supports=frozenset(PARALLEL) | set(ORTHOGONAL))
    combos = CombinationPolicy.for_model(spec)

    # MAX_ORTHOGONAL=1 -> singles + pairs-with-one-parallel; no combo mixes packing+padding_free.
    for combo in combos:
        assert 'packing' not in combo or 'padding_free' not in combo
        assert len([c for c in combo[1:] if c in ORTHOGONAL]) <= CombinationPolicy.MAX_ORTHOGONAL
    # every parallel dimension appears with the bare combination
    assert {c[0] for c in combos} == set(PARALLEL)
    # the mutual exclusion actually removed cells (dp2 + both switches is generated then dropped)
    assert ('dp2', 'packing', 'padding_free') not in combos
    # a spec supporting nothing parallel yields nothing at all
    empty = ModelSpec(name='none', model_type='qwen2', model_id='m', template='t', supports=frozenset())
    assert CombinationPolicy.for_model(empty) == []
    # the width cap bounds the grid: raising it to len(ORTHOGONAL) would include triples
    original = CombinationPolicy.MAX_ORTHOGONAL
    try:
        CombinationPolicy.MAX_ORTHOGONAL = 2
        wider = CombinationPolicy.for_model(spec)
        assert ('dp2', 'packing', 'mtp') in wider
    finally:
        CombinationPolicy.MAX_ORTHOGONAL = original


def test_switches_fold_into_the_right_config_bucket():
    """Every switch writes its knobs into the Config bucket TARGET names for it -- a switch landing
    in the wrong bucket would be a TypeError at run time, silently killing that combination."""
    buckets = CapabilityRun.overrides(('tp2', 'packing', 'padding_free', 'mtp'))
    assert buckets['distributed'] == {'tensor_model_parallel_size': 2}
    assert buckets['dataset'] == {'packing': True}
    assert buckets['template'] == {'padding_free': True}
    assert buckets['model'] == {'mtp_num_layers': 1}
