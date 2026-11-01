# Copyright (c) ModelScope Contributors. All rights reserved.
"""Frontend preset contracts through the REAL presets.ts (S20 20h + S20-26).

The presets are transpiled by esbuild and dumped as v4 declarations by
``_presets_dump.mjs`` (replicating the editor's serializeGraph), then fed to
the REAL python graph validator -- the same wire the vite server sends. The
dump is skipped (not failed) where the webui toolchain is absent."""
import json
import subprocess
from pathlib import Path

import pytest

from swift.dev.webui.graph import Graph
from swift.dev.webui.placement import plan_device_groups
from swift.dev.webui.registry import REGISTRY

pytestmark = pytest.mark.requires_node

_WEBUI = Path(__file__).resolve().parents[4] / 'web-ui-v5' / 'webui'


@pytest.fixture(scope='module')
def declarations():
    """The 13 presets as v4 declaration dicts, off the real TS source."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        bundle = f'{tmp}/presets_bundle.mjs'
        subprocess.run([str(_WEBUI / 'node_modules' / '.bin' / 'esbuild'),
                        'src/pages/workflow/presets.ts', '--bundle', '--format=esm',
                        f'--outfile={bundle}'],
                       cwd=_WEBUI, check=True, capture_output=True, text=True)
        dump = subprocess.run(['node', str(Path(__file__).parent / '_presets_dump.mjs'), bundle],
                              check=True, capture_output=True, text=True)
        return json.loads(dump.stdout)


@pytest.fixture(scope='module')
def built(declarations):
    return {d['workflow_id']: Graph.from_declaration(d) for d in declarations}


class TestPresetContracts:
    """Every preset builds as a v4 graph and keeps the core wiring."""

    def test_thirteen_presets_dump(self, declarations):
        assert len(declarations) == 13

    def test_every_preset_builds_as_v4(self, built):  # S20-59
        assert len(built) == 13

    def test_grpo_prompt_stream_comes_from_the_dataloader(self, built):  # S20-60
        graph = built['grpo']
        prompts = graph.inputs_of('g_driver')['prompts']
        assert prompts.source == 'g_loader'

    def test_grpo_loss_and_scorer_are_wired(self, built):  # S20-61
        graph = built['grpo']
        assert graph.inputs_of('g_train')['loss'].source == 'g_loss'
        scorers = graph.inputs_of('g_score')['scorers']
        assert isinstance(scorers, list) and scorers[0].slot == 'format'

    @pytest.mark.parametrize('key,train_id', [
        ('grpo_full', 'g_train'), ('sft_full', 's_train'), ('opsd', 'o_train')])
    def test_derived_presets_keep_the_loss_edge(self, built, key, train_id):  # S20-62
        assert built[key].inputs_of(train_id)['loss'] is not None

    @pytest.mark.parametrize('key,score_id', [
        ('tcpo', 't_score'), ('psc', 's_score'), ('rsi', 'i_score')])
    def test_challenger_presets_score_through_scorers(self, built, key, score_id):  # S20-63
        scorers = built[key].inputs_of(score_id)['scorers']
        assert isinstance(scorers, list) and len(scorers) == 1

    def test_frontend_registry_matches_the_backend(self, declarations):  # S20-64
        registry = json.loads(
            (_WEBUI / 'src' / 'pages' / 'workflow' / 'registry.json').read_text())
        assert set(registry['components']) == set(REGISTRY)
        for type_key, descriptor in REGISTRY.items():
            frontend = registry['components'][type_key]
            for port in (*descriptor.ports_in, *descriptor.ports_out):
                side = 'ports_in' if port in descriptor.ports_in else 'ports_out'
                assert any(p['key'] == port.key and p['kind'] == port.kind
                           for p in frontend[side]), f'{type_key}.{port.key}'

    def test_every_preset_plans_disjoint_groups(self, declarations):  # S20-26
        for declaration in declarations:
            stocked = json.loads(json.dumps(declaration))
            stocked['resource'] = {'mode': 'local', 'cards': list(range(16))}
            for node in stocked['nodes']:
                if node['type'] in ('model', 'sampler'):
                    node['params']['gpu_count'] = 1
            plan_device_groups(Graph.from_declaration(stocked))  # raises on overlap
