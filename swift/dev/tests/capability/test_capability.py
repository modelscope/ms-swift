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
        model_dir = TinyModel.build(
            tmp_path / 'model', tokenizer_id=spec.model_id, builder=loader_builder(spec.model_type), **spec.extra)
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


CASES = [(spec, switches) for spec in MODELS for switches in CombinationPolicy.for_model(spec)]


@pytest.mark.slow
@pytest.mark.accel(2)
@pytest.mark.parametrize('spec,switches', CASES, ids=[f'{s.name}-{"+".join(c)}' for s, c in CASES])
def test_combination_trains_and_saves(spec, switches, tmp_path):
    """The combination must complete its steps with finite loss and leave a checkpoint behind."""
    losses = CapabilityRun.launch(spec, switches, tmp_path)

    assert len(losses) == CapabilityRun.STEPS, f'expected {CapabilityRun.STEPS} steps, got {losses}'
    assert all(loss == loss and abs(loss) != float('inf') for loss in losses), f'non-finite loss: {losses}'
    ckpt = tmp_path / 'out' / 'checkpoint-final'
    assert ckpt.is_dir(), f'no checkpoint-final under {tmp_path / "out"}'
