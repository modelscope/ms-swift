"""S18a/b — template_contract encoding + output kernel ground truth (L1).

The encoding kernels' literal contracts (SHIFTED_FIELDS mapping, shift/unshift round-trip identity,
refresh's derived fields, TOKEN_FILL) and the output kernels' boundaries (strip_special_tokens,
split_reasoning including the unclosed case) -- the processor suite covers these only through the
Template surface; these tests pin the kernels directly.

Maps to ut_plan S18-01/06/09/14/15.
"""
from swift.dev.template_contract import encoding as E
from swift.dev.template_contract import output as O

OPEN, CLOSE = chr(60) + 'think' + chr(62), chr(60) + '/think' + chr(62)

ROW = {
    'input_ids': [1, 2, 3],
    'labels': [-100, 2, 3],
    'loss_scale': [0.0, 1.0, 1.0],
    'completion_mask': [0, 1, 1],
    'token_type_ids': [0, 0, 0],
}


class TestShiftedFieldsContract:

    def test_shifted_fields_mapping(self):
        """S18-01: exactly the three label-class fields shift, each with its own fill value."""
        assert dict(E.SHIFTED_FIELDS) == {'labels': -100, 'loss_scale': 0.0, 'completion_mask': 0}

    def test_token_fill_mapping(self):
        """S18-09: the two token-type fields fill with 0 when a splice extends them."""
        assert dict(E.TOKEN_FILL) == {'token_type_ids': 0, 'mm_token_type_ids': 0}


class TestShiftUnshift:

    def test_shift_rolls_labels_left_by_one(self):
        """shift moves each label-class field to next-token alignment: out[i] = in[i+1], the tail
        takes the field's fill."""
        shifted = E.shift(ROW)
        assert shifted['labels'] == [2, 3, -100]
        assert shifted['loss_scale'] == [1.0, 1.0, 0.0]
        assert shifted['completion_mask'] == [1, 1, 0]

    def test_token_fields_do_not_shift(self):
        """S18-03 (kernel half): input_ids and token_type_ids are positional token fields -- they
        must pass through untouched (shifting them would misalign every embedding)."""
        shifted = E.shift(ROW)
        assert shifted['input_ids'] == [1, 2, 3]
        assert shifted['token_type_ids'] == [0, 0, 0]

    def test_round_trip_is_identity(self):
        """S18-02: unshift(shift(row)) == row for every field -- a lossy shift would corrupt every
        multi-turn splice downstream."""
        assert E.unshift(E.shift(ROW)) == ROW

    def test_shift_copies_the_row(self):
        """The caller's row is never mutated in place."""
        original = dict(ROW)
        E.shift(ROW)
        assert ROW == original

    def test_missing_shifted_fields_pass_through(self):
        """A row without label fields (pure inference input) shifts to itself."""
        assert E.shift({'input_ids': [1, 2]}) == {'input_ids': [1, 2]}


class TestRefresh:

    def test_refresh_recomputes_derived_fields_and_shifts(self):
        """S18-06: refresh rebuilds attention_mask/length from input_ids and applies the shift --
        a stale derived field (wrong length after a splice) is exactly what it kills."""
        refreshed = E.refresh(ROW)
        assert refreshed['attention_mask'] == [1, 1, 1]
        assert refreshed['length'] == 3
        assert refreshed['labels'] == [2, 3, -100]  # shifted

    def test_refresh_without_input_ids_only_shifts(self):
        out = E.refresh({'labels': [-100, 5]})
        assert out['labels'] == [5, -100]
        assert 'attention_mask' not in out and 'length' not in out


class TestStripSpecialTokens:

    def test_strips_each_token(self):
        """S18-14: each listed special token is removed from the text."""
        assert O.strip_special_tokens('a <|x|> b <|y|>', ['<|x|>', '<|y|>']) == 'a  b '

    def test_no_tokens_present_is_unchanged(self):
        assert O.strip_special_tokens('abc', ['<|x|>']) == 'abc'


class TestSplitReasoning:

    def test_closed_block_splits(self):
        """A closed think block splits into (reasoning, answer)."""
        assert O.split_reasoning(OPEN + 'r' + CLOSE + 'answer') == ('r', 'answer')

    def test_unclosed_block_is_all_reasoning(self):
        """S18-15: a stream cut mid-thought yields the whole text as reasoning with an empty
        answer -- not a crash and not answer leakage."""
        assert O.split_reasoning(OPEN + 'r still thinking') == ('r still thinking', '')

    def test_no_block_is_all_answer(self):
        assert O.split_reasoning('plain') == (None, 'plain')
