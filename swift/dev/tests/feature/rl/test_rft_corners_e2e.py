# Copyright (c) ModelScope Contributors. All rights reserved.
"""S7 rows 05 / 33 / 34 (L2): RFT's selector variants, round-atomic checkpointing and its
empty-selection fail-loudly contract.

RFT's rejection filter (``RFTSelector._select_keep``) has three modes; ``test_algorithms_e2e``
covers ``best_of_n`` (always keeps one, so a round can never come up empty). The two modes that
CAN come up empty -- ``threshold`` (keeps rewards >= cutoff) and ``top_k`` (keeps the k best) --
are pinned here, including the corner that makes the empty case load-bearing: a threshold no
completion can reach must fail loudly with NO checkpoint, because saving one would hand back an
untrained adapter that looks like a finished run.

The checkpoint contract is RFT-specific and only observable from the written artifact: generated
and selected samples are never checkpointed, so a save taken mid-round would carry that round's
optimizer steps but not its completions. ``run_rft`` therefore commits ``next_round`` and saves
together at the round boundary, and persists it as ``component_states['rft']`` in
``trainer_state.json``. Row 33 reads that key back from disk and proves a resume SKIPS the
already-committed round rather than replaying it.
"""
import json
import os

import pytest

from swift.dev.tests.feature.rl.conftest import rl_configs, run_rl

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]


def _component_state(out_dir: str, ckpt_name: str) -> dict:
    """``component_states`` read back from a checkpoint's ``trainer_state.json`` (artifact, not memory)."""
    path = os.path.join(out_dir, ckpt_name, 'trainer_state.json')
    assert os.path.exists(path), f'no trainer_state.json at {path}'
    with open(path, encoding='utf-8') as f:
        return json.load(f).get('component_states', {})


@pytest.mark.parametrize('select,extra', [
    ('top_k', {'rft_top_k': 1}),
    ('threshold', {'rft_threshold': 0.0}),
])
def test_rft_selector_variants_train(select, extra, tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """S7-05: ``top_k`` and ``threshold`` selectors each keep a non-empty set and train.

    On a tiny random policy the ``accuracy`` rule reward is 0.0 for every completion, so
    ``threshold=0.0`` keeps every sample (the comparison is ``>=``) and ``top_k=1`` keeps the best
    of the group -- both distinct code paths from ``best_of_n``'s ``order[:1]``.
    """
    out_dir = str(tmp_path / f'out_rft_{select}')
    configs = rl_configs(
        rl_type='rft',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.rft_math(n=4),
        out_dir=out_dir,
        max_steps=2,
        reward_over={'orm': ['accuracy']},
        rlhf_over={
            'rft_num_samples': 2,
            'rft_select': select,
            'rft_iterations': 1,
            **extra
        },
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    assert_rl_trained(history, configs, f'rft/{select}', max_loss=20.0)


def test_rft_unreachable_threshold_fails_loudly_without_checkpoint(tiny_qwen2_5, rl_data, tmp_path):
    """S7-34: a threshold no completion can reach keeps nothing, so the run takes 0 optimizer
    steps -- which must RAISE and write no checkpoint (a silent 0-step save would look finished)."""
    out_dir = str(tmp_path / 'out_rft_empty')
    configs = rl_configs(
        rl_type='rft',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.rft_math(n=4),
        out_dir=out_dir,
        max_steps=2,
        reward_over={'orm': ['accuracy']},
        rlhf_over={
            'rft_num_samples': 2,
            'rft_select': 'threshold',
            # The accuracy rule scores 0.0..1.0; 1e9 is unreachable, so every group keeps nothing.
            'rft_threshold': 1e9,
            'rft_iterations': 1,
        },
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    with pytest.raises(RuntimeError, match='took 0 optimizer steps'):
        run_rl(configs)
    # No checkpoint at all: neither a periodic one nor the final save (the raise precedes it).
    leftovers = [
        name for name in os.listdir(out_dir) if name.startswith('checkpoint-')
    ] if os.path.isdir(out_dir) else []
    assert not leftovers, f'an empty RFT run must not persist a checkpoint, found {leftovers}'


def test_rft_checkpoint_is_round_atomic_and_resume_skips_committed_rounds(tiny_qwen2_5, rl_data, tmp_path,
                                                                          assert_rl_trained):
    """S7-33: the round counter is committed WITH the checkpoint, and a resume starts at the next
    uncommitted round.

    Phase A runs ONE round (``rft_iterations=1``, ``max_steps=2``) and lands ``checkpoint-final``
    carrying ``component_states['rft']['next_round'] == 1``. Phase B resumes from it with
    ``rft_iterations=2`` and ``max_steps=4``: with correct seeding only round 1 remains, so the run
    adds exactly two more steps ([3, 4]) and commits ``next_round == 2``.

    Load-bearing: a resume that ignored ``component_states`` would start at round 0, and with
    ``max_steps=4`` capping the run it would commit ``next_round == 1`` (round 0 replayed) -- the
    read-back assertion goes red. The step-counter continuation is the second guard (same contract
    the offline-family resume test pins).
    """
    data = rl_data.rft_math(n=4)
    generation = {'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16}

    out_a = str(tmp_path / 'phase_a')
    configs_a = rl_configs(
        rl_type='rft',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=data,
        out_dir=out_a,
        max_steps=2,
        reward_over={'orm': ['accuracy']}, rlhf_over={'rft_num_samples': 2, 'rft_select': 'best_of_n', 'rft_iterations': 1},
        generation_over=generation,
    )
    history_a = run_rl(configs_a)
    assert_rl_trained(history_a, configs_a, 'rft/phase_a', max_loss=20.0)
    ckpt_a = os.path.join(configs_a['checkpoint_config'].output_dir, 'checkpoint-final')
    assert os.path.isdir(ckpt_a), f'phase A wrote no final checkpoint at {ckpt_a}'
    state_a = _component_state(configs_a['checkpoint_config'].output_dir, 'checkpoint-final')
    assert state_a.get('rft', {}).get('next_round') == 1, \
        f'phase A must commit the one round it ran: component_states={state_a}'

    out_b = str(tmp_path / 'phase_b')
    configs_b = rl_configs(
        rl_type='rft',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=data,
        out_dir=out_b,
        max_steps=4,
        ckpt_over={'resume_from_checkpoint': ckpt_a},
        reward_over={'orm': ['accuracy']}, rlhf_over={'rft_num_samples': 2, 'rft_select': 'best_of_n', 'rft_iterations': 2},
        generation_over=generation,
    )
    history_b = run_rl(configs_b)
    assert [r['step'] for r in history_b] == [3, 4], \
        f'resume did not continue the step counter: {[r["step"] for r in history_b]}'
    state_b = _component_state(configs_b['checkpoint_config'].output_dir, 'checkpoint-final')
    assert state_b.get('rft', {}).get('next_round') == 2, (
        f'resume replayed an already-committed round instead of running the next one: '
        f'component_states={state_b} (expected next_round=2)')
