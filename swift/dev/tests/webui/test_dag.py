# Copyright (c) ModelScope Contributors. All rights reserved.
"""DAGController click semantics over fake bindings (S20 20e).

Real scheduling, fake engines: every case asserts an observable the
controller owns -- receipts, source takes/commits, invocation counts and
wall-clock spans. Parallelism is proven by overlapping spans, never by
completion order."""
import copy
import time

from swift.dev.webui.dag import DAGError

from ._fakes import (base_declaration, build_controller, edge, node, overlapping,
                      run_in_thread, wait_for, with_checkpoint)


class TestFrameClicks:
    """One click on a frame member turns exactly one circle."""

    def test_first_click_turns_one_circle(self):  # S20-27
        controller, fakes, _events, _recorded = build_controller(base_declaration())
        receipt = controller.run_component('sync')
        assert receipt['rounds'] == 1 and receipt['optimizer_steps'] == 1
        assert receipt['state'] == 'completed'
        # The source took one batch and committed the real trained steps.
        assert fakes['driver'].takes == 1 and fakes['driver'].commits == [1]
        # Out-of-frame prerequisites really ran before the circle.
        assert fakes['policy'].built == 1 and fakes['smp'].built == 1
        # Every frame member ran exactly once.
        for stage in ('score', 'adv', 'train', 'sync'):
            assert len(fakes[stage].invocations) == 1

    def test_reclick_turns_one_more_circle_without_rebuilds(self):  # S20-32
        controller, fakes, _events, _recorded = build_controller(base_declaration())
        controller.run_component('sync')
        receipt = controller.run_component('sync')
        assert receipt['rounds'] == 1
        assert fakes['driver'].takes == 2 and fakes['driver'].commits == [1, 1]
        # Residents survive the re-click: nothing was rebuilt.
        assert fakes['policy'].built == 1 and fakes['smp'].built == 1
        assert fakes['policy_q'].built == 1
        # The cross-frame producer advances with every consuming round.
        assert fakes['chall'].takes == 2

    def test_steps_budget_turns_two_circles(self):  # S20-33
        controller, fakes, _events, _recorded = build_controller(base_declaration())
        receipt = controller.run_component('train', steps=2)
        assert receipt['rounds'] == 2 and receipt['optimizer_steps'] == 2
        assert fakes['driver'].takes == 2 and fakes['chall'].takes == 2

    def test_observe_leaves_the_batch_pending(self):  # S20-34
        controller, fakes, _events, _recorded = build_controller(base_declaration())
        controller.run_component('sync', observe=True)
        # Observation suppresses every side effect.
        assert len(fakes['train'].invocations) == 0
        assert len(fakes['sync'].invocations) == 0
        assert len(fakes['score'].invocations) == 1
        assert fakes['driver'].commits == []
        assert controller.sources['driver'].has_pending
        # The real click afterwards reuses the pending batch.
        controller.run_component('sync')
        assert fakes['driver'].takes == 1 and fakes['driver'].commits == [1]
        assert len(fakes['train'].invocations) == 1


class TestLoopFreeClicks:
    """Clicks on loop-free targets: prerequisites once, then the target alone."""

    def test_target_downstream_never_runs(self):  # S20-30
        controller, fakes, _events, _recorded = build_controller(
            with_checkpoint(base_declaration()))
        controller.run_component('train')
        # The click reached the training step but never its downstream.
        assert len(fakes['train'].invocations) == 1
        assert fakes['ckpt'].invocations == []

    def test_reclick_runs_the_target_alone(self):  # S20-31
        controller, fakes, _events, _recorded = build_controller(
            with_checkpoint(base_declaration()))
        controller.run_component('ckpt')
        assert fakes['driver'].takes == 1 and fakes['chall'].takes == 1
        controller.run_component('ckpt')
        assert len(fakes['ckpt'].invocations) == 2
        assert fakes['driver'].takes == 1 and fakes['chall'].takes == 1


class TestParallelism:
    """Independent branches overlap; dependent branches wait."""

    def _parallel_declaration(self):
        declaration = copy.deepcopy(base_declaration())
        declaration['nodes'] += [
            node('policy2', 'model', {'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'freeze': True}),
            node('fwd', 'model_forward', {'tokenizer': 'Qwen/Qwen2.5-0.5B-Instruct',
                                          'outputs': ['logps']})]
        declaration['edges'] += [
            edge('e17', 'policy2', 'model', 'fwd', 'model'),
            edge('e18', 'driver', 'samples', 'fwd', 'traj'),
            edge('e19', 'fwd', 'logps', 'train', 'arguments', slot='old_logps'),
            edge('x4', 'policy2', 'model', 'sync', 'model')]
        for frame in declaration['frames']:
            if frame['id'] == 'loop':
                frame['members'].append('fwd')
        return declaration

    def test_independent_branches_overlap(self):  # S20-28
        controller, fakes, _events, _recorded = build_controller(
            self._parallel_declaration(), delays={'score': 0.3, 'fwd': 0.3})
        controller.run_component('train')
        assert overlapping(fakes['score'].spans[0], fakes['fwd'].spans[0]), \
            'score and fwd must run concurrently'

    def test_dependent_branch_waits(self):  # S20-29
        controller, fakes, _events, _recorded = build_controller(
            self._parallel_declaration(), delays={'score': 0.3, 'fwd': 0.3})
        controller.run_component('train')
        assert fakes['adv'].spans[0][0] >= fakes['score'].spans[0][1] - 1e-6


class TestPauseResumeStop:
    """Control lands on node boundaries and never kills submitted work."""

    def test_pause_drains_then_resume_completes(self):  # S20-35 / S20-36
        controller, fakes, events, _recorded = build_controller(
            base_declaration(), delays={'score': 0.4, 'adv': 0.4, 'train': 0.4, 'sync': 0.4})
        result = {}

        worker = run_in_thread(lambda: result.update(receipt=controller.run_component('sync')))
        time.sleep(0.15)
        assert controller.pause()['ok']
        assert wait_for(events, 'paused')
        # The boundary held: nothing running, nothing later started.
        assert controller.get_running_nodes() == []
        assert len(fakes['train'].invocations) == 0
        controller.resume()
        worker.join(10)
        assert not worker.is_alive()
        assert result['receipt']['state'] == 'completed'
        assert fakes['driver'].commits == [1]

    def test_stop_blocks_new_nodes_only(self):  # S20-37
        controller, fakes, _events, _recorded = build_controller(
            base_declaration(), delays={'score': 0.3, 'adv': 0.3, 'train': 0.3})
        result = {}
        worker = run_in_thread(lambda: result.update(receipt=controller.run_component('sync')))
        time.sleep(0.1)
        assert controller.stop()['ok']
        worker.join(10)
        # The in-flight node finished; no later node started.
        assert len(fakes['score'].invocations) == 1
        assert len(fakes['train'].invocations) == 0
        assert result['receipt']['state'] == 'stopped'


class TestCrossFrameDependency:
    """A consuming round forces the producing loop to turn once more."""

    def test_producer_freshens_per_consuming_round(self):  # S20-38
        controller, fakes, _events, _recorded = build_controller(
            base_declaration(), batches=5)
        receipt = controller.run()
        assert receipt['exhausted']
        # The producer turned once per consuming round -- no stale reads.
        assert fakes['driver'].takes == 5 and fakes['chall'].takes == 5
        assert fakes['driver'].commits == [1] * 5
