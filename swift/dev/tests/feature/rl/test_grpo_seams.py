"""S8b — dev2 GRPO worker/runner seams (L1).

Two dev-side functions the v5 snapshot could not express because they post-date it: the teacher-view
token splicer in the reward worker, and CHORD's expert-row loader in the runner. Both are pure Python
with an injected template / monkeypatched loader, so they run without Ray, a model, or a dataset.

Maps to ut_plan S8-37 (_messages_with_token_ids) and S8-41 (_load_chord_features).
"""
import pytest


# ----------------------------------------------------------------------
# S8-37 — _messages_with_token_ids: splice each turn's exact sampled tokens back into the messages
# ----------------------------------------------------------------------
def test_messages_with_token_ids_splices_newest_to_newest():
    """S8-37: turn_token_spans splits the flat response into one span per assistant turn; the newest span
    maps to the newest assistant message, each carrying its ids verbatim (re-encoding text could drift)."""
    from swift.dev.rewards.worker import _messages_with_token_ids

    # labels arrive in OUTPUT/shifted order (twinkle unrolls them before finding trainable runs); this
    # shifted form unrolls to trainable input positions [2,3] and [5,6] -> two assistant turns of 2 tokens.
    encoded = {'input_ids': [0, 0, 10, 11, 0, 12, 13], 'labels': [-100, 10, 11, -100, 12, 13, -100]}
    response_ids = [10, 11, 12, 13]  # the flat trainable tokens (turn_token_spans -> [(0,2),(2,4)])
    messages = [
        {'role': 'user', 'content': 'q0'},
        {'role': 'assistant', 'content': 'a0'},  # turn 0 -> span [0,2) -> [10,11]
        {'role': 'user', 'content': 'obs'},
        {'role': 'assistant', 'content': 'a1'},  # turn 1 -> span [2,4) -> [12,13]
    ]
    out = _messages_with_token_ids(messages, response_ids, encoded)
    assert out[1]['content'] == {'token_ids': [10, 11], 'loss_scale': [1, 1]}
    assert out[3]['content'] == {'token_ids': [12, 13], 'loss_scale': [1, 1]}
    # non-assistant turns are untouched, and the caller's messages are copied, not mutated
    assert out[0]['content'] == 'q0' and out[2]['content'] == 'obs'
    assert messages[1]['content'] == 'a0'


def test_messages_with_token_ids_guards_turn_run_mismatch():
    """S8-37 guard: a run-splitting mismatch (assistant-turn count != trainable-run count) is loud, not a
    silent misalignment -- here one assistant turn but two trainable runs."""
    from swift.dev.rewards.worker import _messages_with_token_ids

    encoded = {'input_ids': [0, 0, 10, 11, 0, 12, 13], 'labels': [-100, 10, 11, -100, 12, 13, -100]}
    with pytest.raises(RuntimeError, match='trainable token runs'):
        _messages_with_token_ids(
            [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}], [10, 11, 12, 13], encoded)


# ----------------------------------------------------------------------
# S8-41 — _load_chord_features: no dataset -> empty; dataset -> encoded rows; no rows -> raise
# ----------------------------------------------------------------------
def test_load_chord_features_three_states(monkeypatch):
    """S8-41: CHORD's expert-SFT rows are encoded once (empty when no chord dataset; a hard error when the
    dataset yields nothing encodable). The loader is monkeypatched so no real dataset is touched."""
    import importlib

    import swift.dev.builders as builders
    from swift.dev.config import DatasetConfig, RLConfig

    run_grpo = importlib.import_module('swift.dev.recipe.grpo.run_grpo')

    class _Tpl:

        def encode(self, row):
            return {'input_ids': row['ids'], 'labels': list(row['ids'])}

    # 1. no chord dataset -> [] (the template/loader are never touched)
    assert run_grpo._load_chord_features(RLConfig(rl_type='grpo'), DatasetConfig(), template=None) == []

    chord_cfg = RLConfig(rl_type='grpo', chord_sft_dataset=['dummy.jsonl'])
    # 2. a chord dataset -> one encoded feature per prompt-only row
    monkeypatch.setattr(builders, 'load_prompt_only_rows', lambda *a, **k: [{'ids': [1, 2]}, {'ids': [3]}])
    features = run_grpo._load_chord_features(chord_cfg, DatasetConfig(), template=_Tpl())
    assert features == [{'input_ids': [1, 2], 'labels': [1, 2]}, {'input_ids': [3], 'labels': [3]}]

    # 3. rows that encode to nothing -> ValueError (a silent empty CHORD set would train on nothing)
    monkeypatch.setattr(builders, 'load_prompt_only_rows', lambda *a, **k: [])
    with pytest.raises(ValueError, match='no encodable rows'):
        run_grpo._load_chord_features(chord_cfg, DatasetConfig(), template=_Tpl())
