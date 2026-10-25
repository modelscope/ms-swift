"""S8b — GRPO recipe seams: multi-turn token splice + CHORD feature loading (L1).

Two seams between the GRPO recipe and the reward/template layers:
``_messages_with_token_ids`` splices each assistant turn's exact sampled token ids back into the
messages (the teacher view must score the tokens the policy PRODUCED, not a re-encode of their
text), and ``_load_chord_features`` encodes CHORD's expert SFT rows once at recipe start. The
latter drives the REAL ``load_prompt_only_rows`` -> ``swift.dev.dataset.load_dataset`` path on a
real jsonl -- it is also the regression anchor for the stale ``trust_remote_code`` kwarg that
crashed every dataset load through the builders' kwargs.

Maps to ut_plan S8-37 and S8-41.
"""
import importlib
import json

import pytest


def _chord_module():
    """The run_grpo MODULE (the package __init__ re-exports the run_grpo FUNCTION under the same
    name, shadowing the submodule -- importlib goes through sys.modules instead)."""
    return importlib.import_module('swift.dev.recipe.grpo.run_grpo')


class TestMessagesWithTokenIds:

    ENCODED = {'labels': [-100, 10, 11, -100, 12, 13, -100]}
    #: turn_token_spans unrolls the output-order labels -> trainable runs [10,11] and [12,13] ->
    #: response spans (0,2),(2,4) over the flat response ids.

    def test_spans_map_to_assistant_turns_newest_to_newest(self):
        """S8-37: each assistant turn's content becomes its exact sampled ids ({token_ids,
        loss_scale}); the OLDEST turn gets the FIRST span, the newest the LAST; non-assistant
        messages pass through untouched."""
        from swift.dev.rewards.worker import _messages_with_token_ids

        messages = [
            {'role': 'user', 'content': 'q1'},
            {'role': 'assistant', 'content': 't1'},
            {'role': 'user', 'content': 'q2'},
            {'role': 'assistant', 'content': 't2'},
        ]
        out = _messages_with_token_ids(messages, [101, 102, 103, 104], self.ENCODED)
        assert out[0] == {'role': 'user', 'content': 'q1'}  # untouched
        assert out[2] == {'role': 'user', 'content': 'q2'}
        assert out[1]['content'] == {'token_ids': [101, 102], 'loss_scale': [1, 1]}  # oldest turn
        assert out[3]['content'] == {'token_ids': [103, 104], 'loss_scale': [1, 1]}  # newest turn

    def test_run_count_mismatch_is_a_hard_error(self):
        """S8-37 guard: more trainable token runs than assistant turns means a turn produced no
        trainable token or a mask split one turn in two -- raise, never silently mis-assign ids."""
        from swift.dev.rewards.worker import _messages_with_token_ids

        messages = [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}]
        with pytest.raises(RuntimeError, match='trainable token runs'):
            _messages_with_token_ids(messages, [101, 102, 103, 104], self.ENCODED)  # 2 runs, 1 turn


class TestLoadChordFeatures:

    def test_no_chord_dataset_is_empty(self):
        """S8-41: with no --chord_sft_dataset the recipe mixes in no expert features."""
        from swift.dev.config import DatasetConfig, RLConfig

        rl = RLConfig(rl_type='grpo')
        assert _chord_module()._load_chord_features(rl, DatasetConfig(dataset=['/tmp/x.jsonl']),
                                                    None) == []

    def test_chord_dataset_encodes_rows_once(self, tmp_path):
        """S8-41: with a chord dataset the expert rows load through the REAL dataset path
        (load_prompt_only_rows -> swift.dev.dataset.load_dataset) and encode via the template.
        REGRESSION ANCHOR: the builders' stale ``trust_remote_code`` kwarg crashed this exact call
        with TypeError before it was removed."""
        from swift.dev.config import DatasetConfig, RLConfig

        path = tmp_path / 'chord.jsonl'
        path.write_text(json.dumps({'messages': [{'role': 'user', 'content': 'hi'}]}) + '\n',
                        encoding='utf-8')

        class _StubTemplate:

            def encode(self, row):
                return {'input_ids': [1, 2], 'labels': [-100, 2]}

        rl = RLConfig(rl_type='grpo', chord_sft_dataset=[str(path)])
        features = _chord_module()._load_chord_features(rl, DatasetConfig(dataset=[str(path)]),
                                                        _StubTemplate())
        assert features == [{'input_ids': [1, 2], 'labels': [-100, 2]}]
