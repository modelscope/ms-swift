# Copyright (c) ModelScope Contributors. All rights reserved.
"""Row-shaping (emit) tests: how sampled candidates become output rows.

Covers the three storage shapes the pipeline can emit -- the plain/eval 'all' row (one row per prompt
carrying every candidate, the SFT/eval shape), the best-of-n 'dpo' chosen/rejected pairs, and the
scored group that a GRPO-style consumer reads -- plus the ranking rules (reward_threshold,
easy_query_threshold, n_best_to_keep), nan-candidate exclusion, and multi-turn trajectory preservation.

All pure: synthetic ``_Candidate`` objects, no sampler, no model.
"""
from swift.dev.config import InferConfig
from swift.dev.recipe.run_infer import _dpo_row, _emit_all, _emit_dpo
from swift.dev.tests.infer.conftest import make_candidate

# A prompt-only trajectory (what split_prompt_and_reference hands the emit layer) and its dataset row.
TRAJECTORY = {'messages': [{'role': 'user', 'content': 'What is 2+2?'}]}
ROW = {'messages': [{'role': 'user', 'content': 'What is 2+2?'}], 'solution': '4', 'extra': 7}


def test_emit_all_single_turn_shape():
    """'all' -> one row: response=first, responses=all, messages=prompt+first answer, labels=reference,
    and every non-messages dataset column rides along."""
    candidates = [make_candidate('4'), make_candidate('5')]
    out = _emit_all(ROW, TRAJECTORY, '4', candidates, None, InferConfig())
    assert len(out) == 1
    row = out[0]
    assert row['response'] == '4'
    assert row['responses'] == ['4', '5']
    assert row['labels'] == '4'
    assert row['solution'] == '4' and row['extra'] == 7  # dataset columns preserved
    # messages is the prompt plus the FIRST candidate as an assistant turn (not the raw dataset messages)
    assert row['messages'] == [{'role': 'user', 'content': 'What is 2+2?'},
                               {'role': 'assistant', 'content': '4'}]
    assert 'scores' not in row  # unscored run carries no scores key


def test_emit_all_with_scores_nulls_nan():
    """A judge that returned nothing numeric scores nan; it must be nulled so the row stays json-safe."""
    candidates = [make_candidate('a'), make_candidate('b')]
    out = _emit_all(ROW, TRAJECTORY, None, candidates, [0.9, float('nan')], InferConfig())
    assert out[0]['scores'][0] == 0.9
    assert out[0]['scores'][1] is None
    assert 'labels' not in out[0]  # ground_truth None -> no labels key


def test_emit_all_multi_turn_stores_every_trajectory():
    """Multi-turn candidates are distinct trajectories; 'all' keeps each one under all_messages."""
    c0 = make_candidate('a', multi_turn=True, messages=[{'role': 'user', 'content': 'q'},
                                                        {'role': 'assistant', 'content': 'a'}])
    c1 = make_candidate('b', multi_turn=True, messages=[{'role': 'user', 'content': 'q'},
                                                        {'role': 'assistant', 'content': 'b'}])
    out = _emit_all(ROW, TRAJECTORY, None, [c0, c1], None, InferConfig())
    assert len(out[0]['all_messages']) == 2
    assert out[0]['all_messages'][1][-1]['content'] == 'b'


def test_emit_all_carries_logprobs_when_present():
    candidates = [make_candidate('a', rollout_logprobs=[-0.1, -0.2])]
    out = _emit_all(ROW, TRAJECTORY, None, candidates, None, InferConfig())
    assert out[0]['logprobs'] == [-0.1, -0.2]


def test_emit_dpo_without_scores_emits_all_positives():
    """No reward func -> no ranking, so every candidate is a positive with no rejected turn."""
    candidates = [make_candidate('a'), make_candidate('b')]
    rows = _emit_dpo(ROW, TRAJECTORY, None, candidates, None, InferConfig())
    assert len(rows) == 2
    assert all('rejected_response' not in r for r in rows)
    assert {r['messages'][-1]['content'] for r in rows} == {'a', 'b'}


def test_emit_dpo_best_of_n_pairs_top_against_worst():
    """With scores, the top n_best_to_keep become positives, each paired against the lowest scorer."""
    candidates = [make_candidate('lo'), make_candidate('mid'), make_candidate('hi')]
    scores = [0.1, 0.5, 0.9]
    cfg = InferConfig(output_format='dpo', num_return_sequences=3, n_best_to_keep=2)
    rows = _emit_dpo(ROW, TRAJECTORY, None, candidates, scores, cfg)
    # negatives is the lowest scorer ('lo'); positives are the top 2 ('hi','mid') minus the negative
    assert len(rows) == 2
    assert all(r['rejected_response'] == 'lo' for r in rows)
    assert {r['messages'][-1]['content'] for r in rows} == {'hi', 'mid'}


def test_emit_dpo_reward_threshold_drops_low_scorers():
    candidates = [make_candidate('a'), make_candidate('b'), make_candidate('c')]
    scores = [0.2, 0.4, 0.9]
    cfg = InferConfig(output_format='dpo', num_return_sequences=3, n_best_to_keep=2, reward_threshold=0.3)
    rows = _emit_dpo(ROW, TRAJECTORY, None, candidates, scores, cfg)
    # 'a' (0.2) is below threshold and dropped from the keep pool; negative is still the lowest scorer
    kept = {r['messages'][-1]['content'] for r in rows}
    assert 'a' not in kept


def test_emit_dpo_easy_query_threshold_skips_whole_prompt():
    """When almost every candidate passes, the prompt teaches nothing and is skipped whole."""
    candidates = [make_candidate('a'), make_candidate('b'), make_candidate('c'), make_candidate('d')]
    scores = [0.9, 0.9, 0.9, 0.1]  # 3 of 4 above threshold 0.5 -> 0.75 >= easy_query_threshold
    cfg = InferConfig(output_format='dpo', num_return_sequences=4, n_best_to_keep=2,
                      reward_threshold=0.5, easy_query_threshold=0.75)
    assert _emit_dpo(ROW, TRAJECTORY, None, candidates, scores, cfg) == []


def test_emit_dpo_excludes_nan_from_ranking():
    """A nan-scored candidate can be neither positive nor negative; it is dropped from the pool."""
    candidates = [make_candidate('good'), make_candidate('unscored'), make_candidate('bad')]
    scores = [0.9, float('nan'), 0.1]
    cfg = InferConfig(output_format='dpo', num_return_sequences=3, n_best_to_keep=1)
    rows = _emit_dpo(ROW, TRAJECTORY, None, candidates, scores, cfg)
    assert len(rows) == 1
    assert rows[0]['messages'][-1]['content'] == 'good'
    assert rows[0]['rejected_response'] == 'bad'


def test_emit_dpo_all_nan_returns_empty():
    candidates = [make_candidate('a'), make_candidate('b')]
    cfg = InferConfig(output_format='dpo', num_return_sequences=2, n_best_to_keep=1)
    assert _emit_dpo(ROW, TRAJECTORY, None, candidates, [float('nan'), float('nan')], cfg) == []


def test_emit_dpo_multi_turn_preserves_full_trajectories():
    """A multi-turn pair keeps chosen AND rejected as full message trajectories, not flat strings."""
    chosen_msgs = [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'good'}]
    rejected_msgs = [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'bad'}]
    chosen = make_candidate('good', multi_turn=True, messages=chosen_msgs,
                            rollout_infos={'turns': 2}, truncated=False)
    rejected = make_candidate('bad', multi_turn=True, messages=rejected_msgs, truncated=True)
    cfg = InferConfig(output_format='dpo', num_return_sequences=2, n_best_to_keep=1)
    rows = _emit_dpo(ROW, TRAJECTORY, None, [chosen, rejected], [0.9, 0.1], cfg)
    assert len(rows) == 1
    row = rows[0]
    assert row['multi_turn'] is True
    assert row['rejected_messages'] == rejected_msgs  # full trajectory, not rejected_response
    assert 'rejected_response' not in row
    assert row['rollout_infos'] == {'turns': 2}
    assert row['truncated'] is False and row['rejected_truncated'] is True


def test_dpo_row_id_is_shared_within_a_group_and_carries_ground_truth():
    """Every row from one prompt shares an id (so positives trace back to their group)."""
    pos = make_candidate('a')
    neg = make_candidate('b')
    r1 = _dpo_row(ROW, TRAJECTORY, pos, neg, '4')
    r2 = _dpo_row(ROW, TRAJECTORY, neg, pos, '4')
    assert r1['id'] == r2['id']
    assert r1['ground_truth'] == '4'
    assert r1['rejected_response'] == 'b'
    # generated keys from the dataset row are stripped, other columns kept
    assert 'extra' in r1 and r1['extra'] == 7
    assert 'rejected_messages' not in r1  # single-turn -> rejected_response, not rejected_messages
