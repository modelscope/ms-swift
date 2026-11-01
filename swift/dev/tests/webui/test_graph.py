# Copyright (c) ModelScope Contributors. All rights reserved.
"""Graph validation, declaration migration and scope resolution (S20 20a-20c).

Every negative case asserts the guard's own error message: the message IS the
product surface a user sees when the drag-and-drop editor refuses a wire.
The declaration dicts come from ``_fakes`` -- the test input is the oracle.
"""
import copy
import json

import pytest

from swift.dev.webui.graph import Graph, GraphUpgrade, GraphValidationError
from swift.dev.webui.registry import REGISTRY, validate_registry
from swift.dev.webui.schema import CHANNEL_SCHEMAS

from ._fakes import MODEL_ID, base_declaration, edge, node, v3_declaration, with_second_model


def refusal(declaration):
    """Build the declaration, returning the GraphValidationError it raises."""
    with pytest.raises(GraphValidationError) as excinfo:
        Graph.from_declaration(copy.deepcopy(declaration))
    return excinfo.value


class TestKindMatrix:
    """20a: edge legality against KIND_ALLOWS and the descriptor surface."""

    def test_endpoint_cannot_feed_scorers(self):  # S20-01
        declaration = base_declaration()
        declaration['nodes'].append(node('api', 'api_endpoint',
                                         {'base_url': 'http://x', 'model': 'm'}))
        declaration['edges'].append(edge('ea', 'api', 'endpoint', 'score', 'scorers', 'api'))
        error = refusal(declaration)
        message = str(error)
        assert 'ea' in message and 'ENDPOINT' in message
        assert 'score.scorers' in message and 'SCORER' in message

    def test_reward_cannot_feed_forward_traj(self):  # S20-02
        declaration = base_declaration()
        declaration['nodes'].append(node('fwd', 'model_forward',
                                         {'tokenizer': MODEL_ID, 'outputs': ['logps']}))
        declaration['edges'].append(edge('er', 'score', 'reward', 'fwd', 'traj'))
        message = str(refusal(declaration))
        assert 'er' in message and 'REWARD' in message
        assert 'fwd.traj' in message and 'TRAJ' in message

    def test_single_input_rejects_slot(self):  # S20-03
        declaration = copy.deepcopy(base_declaration())
        declaration['edges'] = [dict(e, slot='x') if e['id'] == 'e4' else e
                                for e in declaration['edges']]
        message = str(refusal(declaration))
        assert '单输入不能声明 slot' in message and 'e4' in message

    def test_missing_many_port_wire_refused(self):  # S20-04
        declaration = copy.deepcopy(base_declaration())
        declaration['edges'] = [e for e in declaration['edges'] if e['id'] != 'e_s']
        message = str(refusal(declaration))
        assert 'score.scorers' in message and '必需输入未接线' in message

    def test_undeclared_ctrl_cycle_refused_with_path(self):  # S20-05
        declaration = copy.deepcopy(base_declaration())
        for frame in declaration['frames']:
            if frame['id'] == 'loop':
                frame['feedback'] = []
        message = str(refusal(declaration))
        assert '非法同轮环' in message
        assert 'driver' in message and 'train' in message

    def test_feedback_must_return_to_the_frame_source(self):  # S20-06
        declaration = copy.deepcopy(base_declaration())
        for frame in declaration['frames']:
            if frame['id'] == 'loop':
                frame['feedback'] = ['e4']
        message = str(refusal(declaration))
        assert 'feedback' in message and 'loop' in message

    def test_descriptor_surface_matches_channels(self):  # S20-07
        validate_registry()  # raises on any matrix inconsistency
        for descriptor in REGISTRY.values():
            for port in (*descriptor.ports_in, *descriptor.ports_out):
                assert port.kind in CHANNEL_SCHEMAS, f'{descriptor.type}.{port.key}: {port.kind}'

    def test_unknown_node_type_refused(self):  # S20-08
        declaration = copy.deepcopy(base_declaration())
        declaration['nodes'].append(node('zz', 'transformer', {}))
        message = str(refusal(declaration))
        assert "未注册组件 'transformer'" in message and 'zz' in message

    def test_unsupported_version_refused(self):  # S20-09
        declaration = copy.deepcopy(base_declaration())
        declaration['version'] = 99
        message = str(refusal(declaration))
        assert 'version' in message and '仅运行 v4' in message

    def test_to_dict_is_field_lossless(self):  # S20-10
        """Serialization must not rewrite anything the user declared: every
        field in the input survives verbatim in ``to_dict`` (defaults the
        parser adds are additive), and the canonical form is idempotent."""
        declaration = base_declaration()
        graph = Graph.from_declaration(copy.deepcopy(declaration))
        result = json.loads(json.dumps(graph.to_dict()))
        declared = {n['id']: n for n in declaration['nodes']}
        for entry in result['nodes']:
            source = declared[entry['id']]
            assert entry['type'] == source['type']
            for key, value in source['params'].items():
                assert entry['params'][key] == value, f"{entry['id']}.{key}"
        assert result['resource']['mode'] == declaration['resource']['mode']
        assert result['resource']['cards'] == declaration['resource']['cards']
        assert [e for e in result['edges'] if 'slot' in e] == \
            [e for e in declaration['edges'] if 'slot' in e]
        again = Graph.from_declaration(copy.deepcopy(result)).to_dict()
        assert json.loads(json.dumps(again)) == result  # canonical form is stable


class TestDeclarationMigration:
    """20b: the v2/v3 -> v4 explicit upgrade path."""

    def test_v3_upgrade_lifts_mode_and_loss(self):  # S20-11
        preview = GraphUpgrade(v3_declaration()).preview()
        assert preview['can_confirm'], preview.get('validation_error')
        candidate = preview['candidate']
        upgraded_nodes = {n['id']: n for n in candidate['nodes']}
        # V4_LIFTED_PARAMS are consumed by the lift, never a KeyError.
        assert 'mode' not in upgraded_nodes['policy']['params']
        assert 'mode' not in upgraded_nodes['smp']['params']
        assert upgraded_nodes['train:loss']['type'] == 'loss'
        assert upgraded_nodes['train:loss']['params']['kind'] == 'grpo'

    def test_upgrade_takes_over_driver_batch_size(self):  # S20-12
        v3 = v3_declaration()
        for entry in v3['nodes']:
            if entry['id'] == 'driver':
                entry['params']['batch_size'] = 8
        preview = GraphUpgrade(v3).preview()
        driver = {n['id']: n for n in preview['candidate']['nodes']}['driver']
        assert 'batch_size' not in driver['params']

    def test_upgraded_graph_keeps_the_core_wiring(self):  # S20-13
        preview = GraphUpgrade(v3_declaration()).preview()
        upgraded = GraphUpgrade.confirm(preview)
        assert upgraded.inputs_of('driver')['prompts'].source == 'driver:loader'
        assert upgraded.inputs_of('train')['loss'].source == 'train:loss'

    def test_upgraded_graph_still_needs_a_resource(self):  # S20-14
        preview = GraphUpgrade(v3_declaration()).preview()  # v3 carries no resource section
        upgraded = GraphUpgrade.confirm(preview)
        from swift.dev.webui.placement import plan_device_groups
        with pytest.raises(GraphValidationError):
            plan_device_groups(upgraded)


class TestScopeResolution:
    """20c: the implicit variable passing contract -- three observable states."""

    def test_unique_candidate_resolves_implicitly(self):  # S20-15
        graph = Graph.from_declaration(base_declaration())
        assert graph.scope_of('sync') == {'model': ('policy', 'model')}

    def test_multiple_candidates_refused_by_name(self):  # S20-16
        error = refusal(with_second_model(base_declaration()))
        message = str(error)
        assert 'sync.model' in message
        assert 'policy' in message and 'policy2' in message

    def test_explicit_wire_disambiguates(self):  # S20-17
        disambiguated = with_second_model(base_declaration())
        disambiguated['edges'].append(edge('x4', 'policy2', 'model', 'sync', 'model'))
        resolved = Graph.from_declaration(disambiguated)
        assert resolved.scope_of('sync') == {}

    def test_scope_crosses_frame_boundaries(self):  # S20-18
        """A loop-external checkpoint still resolves the in-loop model WITHOUT
        a wire: the identity closure INCLUDES cross-frame edges (they are
        round boundaries, not identity boundaries)."""
        declaration = copy.deepcopy(base_declaration())
        declaration['nodes'].append(node('ckpt', 'checkpoint'))
        declaration['edges'].append(edge('xck', 'train', 'step', 'ckpt', 'step'))
        graph = Graph.from_declaration(declaration)
        assert graph.scope_of('ckpt') == {'model': ('policy', 'model')}
