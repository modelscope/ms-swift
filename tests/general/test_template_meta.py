# Copyright (c) ModelScope Contributors. All rights reserved.
from swift.template import TemplateMeta


def test_replace_system_preserves_non_string_elements():
    """_replace_system must not drop list elements like ['bos_token_id'].

    Templates such as ziya, bluelm and emu3_chat use
    ``prefix=[['bos_token_id'], '{{SYSTEM}}']``.  When no system message is
    provided the prefix is produced by _replace_system, which should keep every
    non-string element intact and only strip the placeholder from strings.
    """
    meta = TemplateMeta(
        template_type='_test_replace_system_bug',
        prefix=[['bos_token_id'], '{{SYSTEM}}'],
        prompt=['{{QUERY}}'],
        chat_sep=['\n'],
    )
    # __post_init__ moves prefix to system_prefix and builds a no-system prefix
    # via _replace_system.  The list element must survive.
    assert any(isinstance(p, list) for p in meta.prefix), (
        f'_replace_system dropped the bos_token_id list; '
        f'meta.prefix={meta.prefix!r}'
    )
