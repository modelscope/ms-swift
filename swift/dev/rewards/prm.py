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
  - :class:`DelimiterPRMScorer` is the default: it splits the decoded response on a text delimiter
    (``RLHFConfig.prm_step_delimiter``, one step per line by default) and scores each step by its response
    prefix through the scalar PRM channel (``compute_reward_model_scores``).

A scorer never owns a model. The training loop injects ``score_steps`` (a batch of step rows -> one PRM
score per row, wired to the frozen PRM) and ``decode`` (the policy tokenizer's decode), so a scorer is pure
text/token logic and is exercisable without a GPU or a checkpoint.
"""
from __future__ import annotations
import copy
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple

from swift.dev.plugin import PluginRegistry, SwiftPlugin

__all__ = ['PRMScorer', 'DelimiterPRMScorer', 'prm_scorers', 'PRM_STEP_SCORER']

#: One reasoning step as a half-open ``[start, end)`` span of response-token indices. The spans a scorer
#: returns must be contiguous and cover the whole response, so every response token belongs to exactly one
#: step (see :meth:`PRMScorer._checked_spans`).
StepSpan = Tuple[int, int]


class PRMScorer(SwiftPlugin):
    """Turn one response into a per-response-token process reward by scoring its reasoning steps.

    ``__call__`` is a template method fixed to three stages so every PRM scorer is interchangeable:

    1. :meth:`segment` -- the one subclass-owned stage -- splits the response into step spans.
    2. Each step is scored by its *response prefix* (``prompt + assistant(response[:end])``): a scalar PRM
       judges the solution as far as that step, which is the per-step score. All steps of a response are
       scored in ONE ``score_steps`` batch so the frozen PRM sees a full batch, not one row at a time.
    3. :meth:`broadcast` maps the per-step scores onto tokens. The default is the *per-step broadcast*
       (every token in a step inherits that step's score); a scorer wanting per-token interpolation
       overrides this one method, leaving segmentation and scoring untouched.

    The result is a plain per-token process reward. Combining it with the outcome (ORM) reward -- placing
    process rewards on the intermediate tokens and the terminal reward on the last one -- is the training
    loop's job, not the scorer's, so a scorer stays reusable across advantage schemes.
    """

    #: Registry key; :class:`DelimiterPRMScorer` sets it and is the default when none is configured.
    name: ClassVar[Optional[str]] = None

    def __call__(self,
                 prompt_messages: Sequence[Dict[str, Any]],
                 response_token_ids: Sequence[int],
                 decoded: str,
                 score_steps: Callable[[List[Dict[str, Any]]], Sequence[float]],
                 decode: Callable[[Sequence[int]], str],
                 columns: Optional[Dict[str, Any]] = None) -> List[float]:
        """One process reward per response token (length ``len(response_token_ids)``).

        Args:
            prompt_messages: the prompt turns; a step is scored against ``prompt + the response so far``.
            response_token_ids: the policy-produced completion tokens -- the frame the output aligns to.
            decoded: the decoded completion text (what :meth:`segment` splits).
            score_steps: scores a batch of step rows (each ``{'messages': [...], **columns}``) with the
                frozen PRM, returning one scalar per row in order. Injected so a scorer owns no model.
            decode: the policy tokenizer's decode, ``List[int] -> str``. Used to render each step's response
                prefix as text and (in the default scorer) to locate step boundaries in token space.
            columns: dataset passthrough columns (``solution`` / ...) the PRM may need; copied onto every
                step row.
        """
        response_token_ids = list(response_token_ids)
        if not response_token_ids:
            return []
        spans = self._checked_spans(self.segment(response_token_ids, decoded or '', decode),
                                    len(response_token_ids))
        columns = columns or {}
        rows: List[Dict[str, Any]] = []
        for _start, end in spans:
            messages = copy.deepcopy(list(prompt_messages))
            messages.append({'role': 'assistant', 'content': decode(response_token_ids[:end])})
            rows.append({'messages': messages, **copy.deepcopy(columns)})
        step_scores = list(score_steps(rows))
        if len(step_scores) != len(spans):
            raise ValueError(f'PRM step scoring returned {len(step_scores)} scores for {len(spans)} steps; '
                             'score_steps must return exactly one score per step row, in order.')
        return self.broadcast(step_scores, spans, len(response_token_ids))

    def segment(self, response_token_ids: Sequence[int], decoded: str,
                decode: Callable[[Sequence[int]], str]) -> List[StepSpan]:
        """Split one response into contiguous ``[start, end)`` response-token spans, one per reasoning step.

        The single stage a subclass owns. Spans are in token space (so the broadcast is exact) and must
        cover ``[0, len(response_token_ids))`` without gaps or overlaps.
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
    def _checked_spans(spans: Sequence[StepSpan], length: int) -> List[StepSpan]:
        """Validate that ``spans`` tiles ``[0, length)`` exactly; a segmentation bug must be loud, not silent.

        A gap would leave a response token with no process reward, and an overlap would score one token
        twice -- both would corrupt the per-token advantage without any visible error, so reject them here.
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
        return checked


class DelimiterPRMScorer(PRMScorer):
    """The default PRM scorer: one reasoning step per delimiter-separated segment of the response.

    Steps are located in TOKEN space by walking the response one token at a time and cutting a step where
    the accumulated per-token text contains the delimiter (``RLHFConfig.prm_step_delimiter``, a newline by
    default, i.e. one step per line -- the shape most chain-of-thought answers emit). Cutting on a token
    boundary keeps the spans exact; the trailing segment after the last delimiter is its own step, and a
    response with no delimiter is a single step covering all its tokens.

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
        #: Step boundary in the decoded text. Read off the run's RLHFConfig (``args``) so it is a plain
        #: ``--prm_step_delimiter`` knob; a newline is the default one-step-per-line segmentation.
        self.delimiter = getattr(args, 'prm_step_delimiter', None) or '\n'

    def segment(self, response_token_ids: Sequence[int], decoded: str,
                decode: Callable[[Sequence[int]], str]) -> List[StepSpan]:
        length = len(response_token_ids)
        spans: List[StepSpan] = []
        start = 0
        buffer = ''
        for index, token in enumerate(response_token_ids):
            buffer += decode([token])
            if self.delimiter in buffer:
                spans.append((start, index + 1))
                start = index + 1
                buffer = ''
        if start < length:
            spans.append((start, length))
        return spans or [(0, length)]


#: name -> scorer. The 'prm_scorer' extension point adopts this dict as its registry, so
#: ``@PluginRegistry.register('prm_scorer', ...)`` and ``prm_scorers['my'] = MyScorer`` write to one place.
prm_scorers = {
    'delimiter': DelimiterPRMScorer,
}

#: The 'prm_scorer' extension point: a run picks a scorer with ``RLHFConfig.prm_scorer`` (a registered name,
#: a class, or a callable), resolved through :meth:`PluginRegistry.resolve` and defaulted to ``'delimiter'``.
PRM_STEP_SCORER = PluginRegistry.register_kind(
    'prm_scorer', PRMScorer, config_field='prm_scorer', entries=prm_scorers)
