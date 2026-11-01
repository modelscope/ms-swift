# Copyright (c) ModelScope Contributors. All rights reserved.
"""GraphSession command protocol and the preview surface (S20 20f).

The command loop and pagination are the REAL product surface; only the
bindings are fakes. Preview pagination is driven against a known 10-row lane
injected through ``record_output`` -- the row list is the oracle."""
import pytest

from ._fakes import base_declaration, build_session, wait_for


@pytest.fixture()
def running_session():
    """A session whose first click ('sync') has already completed."""
    session, fakes, events = build_session(base_declaration())
    reply = session.command({'op': 'run', 'request_id': 'r1',
                             'mode': 'component', 'target': 'sync'})
    assert wait_for(events, 'run_end')
    return session, fakes, events, reply


class TestRunProtocol:
    """Receipts are immediate; results ride events; nothing is materialised."""

    def test_run_receipt_is_immediate_and_lean(self, running_session):  # S20-39
        _session, _fakes, _events, reply = running_session
        assert reply['ok'] and reply['accepted'] and reply['request_id'] == 'r1'
        assert 'outputs' not in reply

    def test_run_end_carries_no_outputs(self, running_session):  # S20-40
        _session, _fakes, events, _reply = running_session
        run_end = next(payload for kind, payload in events if kind == 'run_end')
        assert 'outputs' not in run_end
        assert run_end['state'] == 'idle'
        result = run_end['result']
        assert set(result) >= {'mode', 'rounds', 'optimizer_steps', 'state'}
        assert result['optimizer_steps'] == 1

    def test_duplicate_request_id_is_idempotent(self, running_session):  # S20-48
        session, fakes, _events, _reply = running_session
        reply = session.command({'op': 'run', 'request_id': 'r1',
                                 'mode': 'component', 'target': 'sync'})
        assert reply['ok'] and reply.get('duplicate')
        assert fakes['driver'].takes == 1  # nothing re-executed


class TestPreview:
    """The one place references materialise: bounded, paginated, loud."""

    def _inject_lane(self, session, node_id, lane, rows):
        """Record a known-length lane as the node's output (the oracle)."""
        session._outputs[node_id] = {
            'node': node_id, 'type': 'score',
            'output': {lane: rows}}

    def test_nodes_listing_without_node_id(self, running_session):  # S20-41
        session, _fakes, _events, _reply = running_session
        reply = session.command({'op': 'preview'})
        assert reply['ok'] and 'nodes' in reply
        assert 'train' in reply['nodes'] and 'score' in reply['nodes']

    def test_lane_summaries_with_node_only(self, running_session):  # S20-42
        session, _fakes, _events, _reply = running_session
        self._inject_lane(session, 'score', 'reward', [f'row{i}' for i in range(10)])
        reply = session.command({'op': 'preview', 'node_id': 'score'})
        assert reply['ok']
        summary = reply['lanes']['reward']
        assert summary['total'] == 10 and len(summary['head']) <= 2

    def test_windowed_rows_with_offset_and_limit(self, running_session):  # S20-43
        session, _fakes, _events, _reply = running_session
        rows = [f'row{i}' for i in range(10)]
        self._inject_lane(session, 'score', 'reward', rows)
        reply = session.command({'op': 'preview', 'node_id': 'score',
                                 'lane': 'reward', 'offset': 3, 'limit': 4})
        assert reply['ok'] and reply['total'] == 10
        assert reply['rows'] == rows[3:7]

    def test_offset_beyond_total_is_an_empty_page(self, running_session):  # S20-44
        session, _fakes, _events, _reply = running_session
        self._inject_lane(session, 'score', 'reward', [f'row{i}' for i in range(10)])
        reply = session.command({'op': 'preview', 'node_id': 'score',
                                 'lane': 'reward', 'offset': 10, 'limit': 5})
        assert reply['ok'] and reply['rows'] == [] and reply['total'] == 10

    def test_unknown_lane_names_the_available_ones(self, running_session):  # S20-45
        session, _fakes, _events, _reply = running_session
        self._inject_lane(session, 'score', 'reward', [1])
        reply = session.command({'op': 'preview', 'node_id': 'score', 'lane': 'nope'})
        assert not reply['ok']
        assert 'reward' in reply['error']

    def test_unrecorded_node_says_run_it_first(self, running_session):  # S20-46
        session, _fakes, _events, _reply = running_session
        session._outputs.pop('score', None)
        reply = session.command({'op': 'preview', 'node_id': 'score'})
        assert not reply['ok'] and 'run the node first' in reply['error']

    def test_limit_above_the_channel_bound_refused(self, running_session):  # S20-47
        session, _fakes, _events, _reply = running_session
        self._inject_lane(session, 'score', 'reward', [f'row{i}' for i in range(10)])
        from swift.dev.webui.graph import GraphValidationError
        with pytest.raises(GraphValidationError):
            session.command({'op': 'preview', 'node_id': 'score',
                             'lane': 'reward', 'offset': 0, 'limit': 2000})


class TestUpdateAndRelease:
    """Edit-then-continue semantics and the only resource-ending path."""

    def test_update_invalidates_changed_residents(self, running_session):  # S20-49
        session, _fakes, _events, _reply = running_session
        declaration = session._declaration
        for entry in declaration['nodes']:
            if entry['id'] == 'adv':
                entry['params']['profile'] = 'token'
        reply = session.command({'op': 'update', 'declaration': declaration})
        assert reply['ok'] and reply['revision'] == 1
        assert 'adv' not in session._residents   # fingerprint changed -> dropped
        assert 'policy' in session._residents    # untouched resident survives

    def test_update_over_training_state_demands_reset(self, running_session):  # S20-49b
        session, _fakes, _events, _reply = running_session
        declaration = session._declaration
        for entry in declaration['nodes']:
            if entry['id'] == 'train':
                entry['params']['lr'] = 0.5
        session.command({'op': 'update', 'declaration': declaration})
        assert session.state == 'reset_required'
        assert 'train' in session._residents  # nothing silently rebuilt under training state

    def test_placement_change_demands_a_rebuild(self, running_session):  # S20-49c
        session, _fakes, _events, _reply = running_session
        declaration = session._declaration
        for entry in declaration['nodes']:
            if entry['id'] == 'policy':
                entry['params']['gpu_count'] = 2
        reply = session.command({'op': 'update', 'declaration': declaration})
        assert not reply['ok'] and reply['state'] == 'rebuild_required'

    def test_release_clears_everything_and_refuses_runs(self, running_session):  # S20-50
        session, fakes, _events, _reply = running_session
        assert session.command({'op': 'release'})['ok']
        assert session.state == 'released'
        assert not session._residents and not session._outputs and not session._sources
        reply = session.command({'op': 'run', 'request_id': 'r2',
                                 'mode': 'component', 'target': 'sync'})
        assert not reply['ok'] and 'released' in reply['error']
