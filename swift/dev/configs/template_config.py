"""Chat template, formatting, truncation, and loss configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class TemplateConfig:
    """Template selection, sequence control, and thinking mode settings."""

    template: Optional[str] = None
    system: Optional[str] = None
    max_length: Optional[int] = None
    truncation_strategy: Literal['delete', 'left', 'right', 'split', None] = None
    max_pixels: Optional[int] = None
    agent_template: Optional[str] = None
    norm_bbox: Literal['norm1000', 'none', None] = None
    use_chat_template: Optional[bool] = None
    padding_side: Literal['left', 'right'] = 'right'
    padding_free: bool = False
    loss_scale: str = 'default'
    sequence_parallel_size: int = 1
    is_binary_loss_scale: Optional[bool] = None
    template_backend: Literal['swift', 'jinja'] = 'swift'
    response_prefix: Optional[str] = None
    enable_thinking: Optional[bool] = None
    preserve_thinking: Optional[bool] = None
    add_non_thinking_prefix: bool = True
    disable_ignore_empty_think: bool = False

    # Which template implementation encodes: legacy's (True) or dev's own rewrite (False). dev-only,
    # no legacy counterpart. It selects a CLASS -- build_template either derives from the legacy
    # template class, keeping all of its behaviour and adding only dev's next-token label shift, or
    # re-classes into dev's Template. The label convention (contract 1) is the same either way.
    #
    # A Config field rather than the env var it started as because it changes what a run computes: a
    # value that decides the tokenization must be visible in the Config the run reports, and reachable
    # from a test.
    #
    # DEFAULTS TO True, i.e. dev's own encode is opt-in, for two reasons:
    #   1. dev cannot express some templates yet. Qwen3.5's chat template injects an empty
    #      '<think>\n\n</think>\n\n' block on a FULL encode but not when re-encoding a message prefix,
    #      which breaks TokenizeByRound's append-only assumption (it locates assistant spans by
    #      prefix-length diffs) -- dev fails fast in _assert_prefix_append_only, so that model family
    #      can only train with this on. legacy assembles the spans itself and is immune.
    #   2. Every dev-vs-legacy loss comparison was measured with legacy's assembly, so the other
    #      default would have made the unvalidated path the default one.
    # Turn it off only to work on dev's encode itself.
    legacy_encode: bool = True
