# Copyright (c) ModelScope Contributors. All rights reserved.
"""Process Reward Model (PRM) step scorers for swift.dev -- the built-in ``prm_scorer`` plugins.

A PRM scores *how* a solution reasons, not only whether its final answer is right, so training against one
needs a reward for each reasoning step rather than a single scalar for the whole response. This module owns
that "segment + score" process behind one extension point (:data:`PRM_STEP_SCORER`), exactly the way
:mod:`swift.dev.rewards.orm` owns outcome rewards:

  - :class:`PRMScorer` is the general base -- a template method that segments a response into steps, scores
    each step with an injected PRM callable, and broadcasts the per-step scores into a per-response-token
    process reward. A subclass owns only *how a response is split into steps*; building the scoring rows,
    calling the PRM and the per-step broadcast (the mapping the training loop consumes) are shared.
  - :class:`DelimiterPRMScorer` splits the decoded response on a text delimiter
    (``RLConfig.prm_step_delimiter``, one step per line by default); under a multi-turn rollout it splits
    WITHIN each assistant turn separately, never across a turn boundary.
  - :class:`TurnPRMScorer` is the multi-turn default: one assistant turn (thought + action) is one step,
    matching how agentic-RL PRM research (PAIR, AgentPRM) scores a tool-using trajectory -- the tool
    observation a turn triggers is context for the NEXT turn, not part of this one.

A scorer never owns a model. The training loop injects ``score_steps`` (a batch of step rows -> one PRM
score per row, wired to the frozen PRM) and ``decode`` (the policy tokenizer's decode), so a scorer is pure
text/token logic and is exercisable without a GPU or a checkpoint. A multi-turn rollout also injects a
:class:`MultiTurnContext` (the real conversation plus its per-turn token spans) so a step is scored against
the true history -- prior turns AND their tool observations -- rather than a flat splice of trainable tokens.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple

from swift.dev.utils.plugin import PluginRegistry, SwiftPlugin

__all__ = ['PRMScorer', 'DelimiterPRMScorer', 'TurnPRMScorer', 'MultiTurnContext', 'prm_scorers', 'PRM_STEP_SCORER']

#: One reasoning step as a half-open ``[start, end)`` span of response-token indices. The spans a scorer
#: returns must be contiguous and cover the whole response, so every response token belongs to exactly one
#: step (see :meth:`PRMScorer._checked_spans`).
StepSpan = Tuple[int, int]


@dataclass
class MultiTurnContext:
    """Turn structure of a multi-turn rollout, so a PRM step never spans two assistant turns.

    Built by the GRPO assembler (:meth:`swift.dev.recipe.grpo.assembler.GRPOAssembler.score_sample`) from a
    ``RolloutSample`` whose ``messages`` carries the full conversation and whose ``encoded`` feature
    yields the per-turn token spans (:func:`swift.dev.rollout.multi_turn.turn_token_spans`). ``None``
    is passed instead for a single-turn rollout, which has no turn structure to respect.

    Fields:
        messages: the full conversation exactly as the rollout recorded it -- assistant turns AND the
            tool/observation turns between them.
        turn_spans: ``[start, end)`` spans, one per assistant turn, indexing the flat
            ``response_token_ids`` array :meth:`PRMScorer.__call__` receives.
        assistant_message_indices: for the k-th ``turn_spans`` entry, the index into ``messages`` of that
            turn's assistant message, so ``messages[:assistant_message_indices[k]]`` is the true history
            (prior assistant turns AND their tool observations) preceding turn k.
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

    ``__call__`` is a template method fixed to three stages so every PRM scorer is interchangeable:

    1. :meth:`segment` -- the one subclass-owned stage -- splits the response into step spans.
    2. Each step is scored by its *response prefix*: for a single-turn rollout that is
       ``prompt + assistant(response[:end])``; for a multi-turn rollout (``turn_context`` given) it is the
       real conversation history preceding the step's turn -- prior assistant turns AND their tool
       observations -- plus that turn's own text up to the step, so a scalar PRM judges the solution as far
       as that step with the context it actually had. All steps of a response are
       scored in ONE ``score_steps`` batch so the frozen PRM sees a full batch, not one row at a time.
    3. :meth:`broadcast` maps the per-step scores onto tokens. The default is the *per-step broadcast*
       (every token in a step inherits that step's score); a scorer wanting per-token interpolation
       overrides this one method, leaving segmentation and scoring untouched.

    The result is a plain per-token process reward. Combining it with the outcome (ORM) reward -- the
    training loop's ``prm_orm_mix`` convex blend of this per-token process reward with the group-relative
    outcome advantage -- is the training loop's job, not the scorer's, so a scorer stays reusable across
    advantage schemes.
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

        Args:
            prompt_messages: the prompt turns; a single-turn step is scored against
                ``prompt + the response so far``.
            response_token_ids: the policy-produced completion tokens -- the frame the output aligns to.
            decoded: the decoded completion text (what :meth:`segment` splits).
            score_steps: scores a batch of step rows (each ``{'messages': [...], **columns}``) with the
                frozen PRM, returning one scalar per row in order. Injected so a scorer owns no model.
            decode: the policy tokenizer's decode, ``List[int] -> str``. Used to render each step's response
                prefix as text and (in the default scorer) to locate step boundaries in token space.
            columns: dataset passthrough columns (``solution`` / ...) the PRM may need; copied onto every
                step row.
            turn_context: the multi-turn rollout's real conversation and per-turn token spans (see
                :class:`MultiTurnContext`); ``None`` for a single-turn rollout. When present, a step is
                scored against the true history preceding its turn (prior assistant turns AND their tool
                observations) plus that turn's own text up to the step, and :meth:`segment` must not span
                two turns.
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
        """Split one response into contiguous ``[start, end)`` response-token spans, one per reasoning step.

        The single stage a subclass owns. Spans are in token space (so the broadcast is exact) and must
        cover ``[0, len(response_token_ids))`` without gaps or overlaps. When ``turn_context`` is given
        (a multi-turn rollout), no span may cross an assistant-turn boundary --
        :meth:`PRMScorer._checked_spans` enforces this regardless of the subclass.
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
        """Validate that ``spans`` tiles ``[0, length)`` exactly; a segmentation bug must be loud, not silent.

        A gap would leave a response token with no process reward, and an overlap would score one token
        twice -- both would corrupt the per-token advantage without any visible error, so reject them here.
        When ``turn_context`` is given, also reject a span that crosses an assistant-turn boundary: a step
        must live entirely within one turn so its scored prefix is that turn's own text, not a splice of two.
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

    Steps are located in TOKEN space by walking the response one token at a time and cutting a step where
    the accumulated per-token text contains the delimiter (``RLConfig.prm_step_delimiter``, a newline by
    default, i.e. one step per line -- the shape most chain-of-thought answers emit). Cutting on a token
    boundary keeps the spans exact; the trailing segment after the last delimiter is its own step, and a
    response with no delimiter is a single step covering all its tokens.

    Under a multi-turn rollout (``turn_context`` given) the same scan runs separately WITHIN each assistant
    turn's span, so a step never crosses a turn boundary -- a line break at the very end of one turn and
    the very start of the next is a turn boundary, not a step boundary.

    Each step is then scored by its response prefix (:meth:`PRMScorer.__call__`), so a step's score is the
    PRM's judgement of the solution up to and including that step.

    The per-token scan decodes each token in isolation to find delimiters, which can differ from a
    whole-sequence decode at BPE/space boundaries; that only affects *where* a line break is detected, never
    the exactness of the token spans, and the text actually scored is a proper ``decode(response[:end])``
    prefix. A delimiter that is merged into a larger token is still detected because the check is a substring
    test on the accumulated pieces.
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

    Agentic-RL PRM research (PAIR, AgentPRM) scores a tool-using trajectory one assistant turn at a time --
    a turn's thought+action is the unit a PRM judges, and the tool observation it triggers is context for
    the NEXT turn, not part of this one. This scorer therefore needs no delimiter and no text scan: the turn
    spans :func:`swift.dev.rollout.multi_turn.turn_token_spans` recovers from the trajectory's labels ARE
    the steps. Requires ``turn_context`` (a multi-turn rollout); a single-turn response has no turn
    structure to segment and must use :class:`DelimiterPRMScorer` instead.
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
