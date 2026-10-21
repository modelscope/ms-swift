# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple

from swift.dev.utils.plugin import PluginRegistry, SwiftPlugin

__all__ = ['PRMScorer', 'DelimiterPRMScorer', 'TurnPRMScorer', 'MultiTurnContext', 'prm_scorers', 'PRM_STEP_SCORER']

#: One reasoning step as a half-open ``[start, end)`` span of response-token indices. Spans a scorer
#: returns must tile the whole response (see :meth:`PRMScorer._checked_spans`).
StepSpan = Tuple[int, int]


@dataclass
class MultiTurnContext:
    """Turn structure of a multi-turn rollout, so a PRM step never spans two assistant turns.

    Built from a RolloutSample whose ``messages`` carry the full conversation and whose encoded
    feature yields the per-turn spans; ``None`` for single-turn. ``turn_spans``: ``[start, end)``
    per assistant turn over the flat response tokens; ``assistant_message_indices[k]``: the messages
    index of turn k's assistant message (the slice before it is that turn's true history).
    """
    messages: List[dict]
    turn_spans: List[StepSpan]
    assistant_message_indices: List[int]

    def turn_index_for(self, position: int) -> int:
        """Which turn's span contains ``position`` (a response-token index); ``-1`` if none does."""
        for index, (start, end) in enumerate(self.turn_spans):
            if start <= position < end:
                return index
        return -1

    def history_before(self, turn_index: int) -> List[dict]:
        """A deep copy of the real conversation preceding turn ``turn_index`` (prior turns + observations)."""
        return copy.deepcopy(self.messages[:self.assistant_message_indices[turn_index]])


class PRMScorer(SwiftPlugin):
    """Turn one response into a per-response-token process reward by scoring its reasoning steps.

    ``__call__`` is a fixed three-stage template: ``segment`` (the subclass-owned stage) splits the
    response into step spans; every step is scored by its prefix (the true history under multi-turn)
    in ONE ``score_steps`` batch; ``broadcast`` maps step scores onto tokens (default per-step;
    override for interpolation). Blending the result with the ORM outcome is the loop's job.
    """

    #: Registry key; :class:`DelimiterPRMScorer` (single-turn default) and :class:`TurnPRMScorer`
    #: (multi-turn default) set it. Which one a run defaults to when ``RLConfig.prm_scorer`` is unset is
    #: decided by the rollout's shape in ``run_grpo._build_prm_channel``, not by this base class.
    name: ClassVar[Optional[str]] = None

    def __call__(self,
                 prompt_messages: Sequence[Dict[str, Any]],
                 response_token_ids: Sequence[int],
                 decoded: str,
                 score_steps: Callable[[List[Dict[str, Any]]], Sequence[float]],
                 decode: Callable[[Sequence[int]], str],
                 columns: Optional[Dict[str, Any]] = None,
                 turn_context: Optional[MultiTurnContext] = None) -> List[float]:
        """One process reward per response token (length ``len(response_token_ids)``).

        ``prompt_messages``: prompt turns. ``response_token_ids``: completion tokens. ``decoded``:
        the text ``segment`` splits. ``score_steps``: scores the step rows with the frozen PRM
        (injected; the scorer owns no model). ``decode``: the tokenizer's decode. ``turn_context``:
        multi-turn structure, else single-turn (a step scores against ``prompt + response-so-far``).
        """
        response_token_ids = list(response_token_ids)
        if not response_token_ids:
            return []
        spans = self._checked_spans(
            self.segment(response_token_ids, decoded or '', decode, turn_context=turn_context),
            len(response_token_ids),
            turn_context=turn_context)
        columns = columns or {}
        rows: List[Dict[str, Any]] = []
        for start, end in spans:
            if turn_context is None:
                messages = copy.deepcopy(list(prompt_messages))
                messages.append({'role': 'assistant', 'content': decode(response_token_ids[:end])})
            else:
                turn_index = turn_context.turn_index_for(start)
                if turn_index < 0:
                    raise RuntimeError(
                        f'PRM step span [{start}, {end}) does not fall within any assistant turn of the '
                        'multi-turn context; _checked_spans should have rejected a turn-crossing span first.')
                turn_start = turn_context.turn_spans[turn_index][0]
                messages = turn_context.history_before(turn_index)
                messages.append({'role': 'assistant', 'content': decode(response_token_ids[turn_start:end])})
            rows.append({'messages': messages, **copy.deepcopy(columns)})
        step_scores = list(score_steps(rows))
        if len(step_scores) != len(spans):
            raise ValueError(f'PRM step scoring returned {len(step_scores)} scores for {len(spans)} steps; '
                             'score_steps must return exactly one score per step row, in order.')
        return self.broadcast(step_scores, spans, len(response_token_ids))

    def segment(self, response_token_ids: Sequence[int], decoded: str,
                decode: Callable[[Sequence[int]], str],
                turn_context: Optional[MultiTurnContext] = None) -> List[StepSpan]:
        """Split one response into contiguous ``[start, end)`` response-token spans, one per step.

        The single subclass-owned stage. Spans must tile ``[0, len(response_token_ids))`` with no
        gaps or overlaps; under ``turn_context`` no span may cross an assistant-turn boundary
        (:meth:`_checked_spans` enforces both regardless of the subclass).
        """
        raise NotImplementedError

    @staticmethod
    def broadcast(step_scores: Sequence[float], spans: Sequence[StepSpan], length: int) -> List[float]:
        """Per-step broadcast: every response token in a step's span inherits that step's score.

        This is the default PRM->token mapping. A per-token interpolation scheme overrides it; the spans
        already carry the step boundaries either way.
        """
        per_token = [0.0] * length
        for score, (start, end) in zip(step_scores, spans):
            value = float(score)
            for index in range(start, end):
                per_token[index] = value
        return per_token

    @staticmethod
    def _checked_spans(spans: Sequence[StepSpan], length: int,
                       turn_context: Optional[MultiTurnContext] = None) -> List[StepSpan]:
        """Validate that ``spans`` tiles ``[0, length)`` exactly; a segmentation bug must be loud.

        A gap leaves a token unscored and an overlap scores one twice -- both corrupt the per-token
        advantage silently, so reject here. With ``turn_context``, also reject a span crossing an
        assistant-turn boundary: a step must live entirely within one turn.
        """
        checked = list(spans)
        if not checked:
            raise ValueError('PRM segmentation produced no steps for a non-empty response.')
        cursor = 0
        for start, end in checked:
            if start != cursor or end <= start or end > length:
                raise ValueError(f'PRM step spans {checked!r} do not tile [0, {length}) contiguously '
                                 f'(bad span [{start}, {end}) at cursor {cursor}); a scorer must cover every '
                                 'response token exactly once.')
            cursor = end
        if cursor != length:
            raise ValueError(f'PRM step spans {checked!r} cover only {cursor} of {length} response tokens.')
        if turn_context is not None:
            for start, end in checked:
                start_turn = turn_context.turn_index_for(start)
                end_turn = turn_context.turn_index_for(end - 1)
                if start_turn < 0 or start_turn != end_turn:
                    raise ValueError(
                        f'PRM step span [{start}, {end}) crosses an assistant-turn boundary (turn '
                        f'{start_turn} at its start vs {end_turn} at its end); a multi-turn step must live '
                        'entirely within one turn.')
        return checked


class DelimiterPRMScorer(PRMScorer):
    """One reasoning step per delimiter-separated segment of the response.

    Steps are cut in TOKEN space where the accumulated per-token text contains the delimiter
    (``prm_step_delimiter``; a newline by default, one step per line); the trailing segment is its
    own step, and under multi-turn the scan runs separately within each assistant turn. Isolated
    per-token decodes may differ at BPE boundaries, which only moves a line break, never the spans.
    """

    name = 'delimiter'

    def __init__(self, args: Optional[Any] = None, **kwargs):
        super().__init__(args, **kwargs)
        #: Step boundary in the decoded text. Read off the run's RLConfig (``args``) so it is a plain
        #: ``--prm_step_delimiter`` knob; a newline is the default one-step-per-line segmentation.
        self.delimiter = getattr(args, 'prm_step_delimiter', None) or '\n'

    def segment(self, response_token_ids: Sequence[int], decoded: str,
                decode: Callable[[Sequence[int]], str],
                turn_context: Optional[MultiTurnContext] = None) -> List[StepSpan]:
        if turn_context is None:
            return self._segment_range(response_token_ids, 0, len(response_token_ids), decode)
        spans: List[StepSpan] = []
        for turn_start, turn_end in turn_context.turn_spans:
            spans.extend(self._segment_range(response_token_ids, turn_start, turn_end, decode))
        return spans

    def _segment_range(self, response_token_ids: Sequence[int], range_start: int, range_end: int,
                       decode: Callable[[Sequence[int]], str]) -> List[StepSpan]:
        """The delimiter scan over ``[range_start, range_end)``, the whole response or one turn's slice."""
        spans: List[StepSpan] = []
        cursor = range_start
        buffer = ''
        for index in range(range_start, range_end):
            buffer += decode([response_token_ids[index]])
            if self.delimiter in buffer:
                spans.append((cursor, index + 1))
                cursor = index + 1
                buffer = ''
        if cursor < range_end:
            spans.append((cursor, range_end))
        return spans or [(range_start, range_end)]


class TurnPRMScorer(PRMScorer):
    """The multi-turn default: one assistant turn (thought + action) is one reasoning step.

    Agentic-RL PRM research scores a tool-using trajectory one assistant turn at a time -- a turn's
    thought+action is the unit a PRM judges, and the tool observation it triggers is context for the
    NEXT turn. The turn spans ``turn_token_spans`` recovers ARE the steps; requires ``turn_context``
    (single-turn has no turn structure -- use ``delimiter`` there).
    """

    name = 'turn'

    def segment(self, response_token_ids: Sequence[int], decoded: str,
                decode: Callable[[Sequence[int]], str],
                turn_context: Optional[MultiTurnContext] = None) -> List[StepSpan]:
        if turn_context is None:
            raise ValueError(
                "TurnPRMScorer requires a multi-turn rollout (turn_context); a single-turn response has no "
                'turn structure to segment. Use --prm_scorer delimiter for single-turn runs.')
        return list(turn_context.turn_spans)


#: name -> scorer. The 'prm_scorer' extension point adopts this dict as its registry, so
#: ``@PluginRegistry.register('prm_scorer', ...)`` and ``prm_scorers['my'] = MyScorer`` write to one place.
prm_scorers = {
    'delimiter': DelimiterPRMScorer,
    'turn': TurnPRMScorer,
}

#: The 'prm_scorer' extension point: a run picks a scorer with ``RLConfig.prm_scorer`` (a registered name,
#: a class, or a callable), resolved through :meth:`PluginRegistry.resolve` and defaulted to ``'delimiter'``.
PRM_STEP_SCORER = PluginRegistry.register_kind(
    'prm_scorer', PRMScorer, config_field='prm_scorer', entries=prm_scorers)
