# Copyright (c) ModelScope Contributors. All rights reserved.
"""The controlled vocabulary for model / dataset ``tags``.

Tags are documentation-and-filtering labels only -- no code branches on them -- but an open-ended
string field drifts: the same concept accumulates spelling variants (``cot``/``CoT``, ``qa``/``QA``,
``multilingual``/``multi-lingual``, ``coding``/``code``, ``classification``/``cls``) that fragment the
generated docs table and make tag filtering unreliable. :data:`CONTROLLED_TAGS` is the single source
of truth for the allowed set; ``register_model`` / ``register_dataset`` validate against it, so
introducing a tag is a deliberate edit here (canonical, lower-case) rather than an ad-hoc string
coined at a registration site.
"""
from typing import Sequence

# Canonical lower-case spellings, grouped by theme so a new tag lands next to its kin.
CONTROLLED_TAGS = frozenset({
    # modality / capability
    'audio', 'bert', 'caption', 'caption-grounding', 'coding', 'docqa', 'embedding', 'grounding',
    'math', 'multi-modal', 'ner', 'nl2sql', 'ocr', 'ocr-vqa', 'omni', 'poetry', 'rerank',
    'reranker', 'science', 'similarity', 'sql', 'text-generation', 'tts', 'video', 'vision', 'vqa',
    # task / training paradigm
    'agent', 'bench', 'classification', 'cot', 'distill', 'dpo', 'em', 'grpo', 'hfrl', 'kto',
    'multi-agent', 'multi-domain', 'multi-round', 'multi-task', 'multilabel', 'o1', 'orpo',
    'pairwise', 'pretrain', 'qa', 'r1', 'rl', 'rlhf', 'rlvr', 'sft', 'thinking',
    # language / region
    'en', 'multilingual', 'zh', 'zhihu',
    # domain / topic
    'character-dialogue', 'financial', 'general', 'law', 'long', 'long-sequence', 'medical',
    'novel', 'play-ground', 'poison-management', 'quality', 'role-play', 'self-cognition',
    # provenance / marker
    'awq', 'chat', 'common', 'gpt4', 'human-annotated', 'youtube', '🔥',
})


def validate_tags(tags: Sequence[str], *, context: str) -> None:
    """Fail loudly on any tag outside :data:`CONTROLLED_TAGS`.

    ``context`` names the offending family / dataset so the error points back at the registration
    site instead of surfacing as an anonymous vocabulary violation.
    """
    unknown = [tag for tag in tags if tag not in CONTROLLED_TAGS]
    if unknown:
        raise ValueError(
            f'{context}: unknown tag(s) {sorted(unknown)}. Tags must come from the controlled '
            'vocabulary in swift/dev/tags.py -- add a genuinely new concept there (canonical, '
            'lower-case) rather than coining a spelling at the registration site.')
