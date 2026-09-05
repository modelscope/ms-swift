# Copyright (c) ModelScope Contributors. All rights reserved.
"""Every local link in the repo's own markdown must land somewhere -- the file *and* the anchor.

Docs rot invisibly: a section gets renamed and the twenty links pointing at it still look like links.
The anchor is the half nobody checks -- the script this test replaces (``scripts/utils/
test_link_valid.py``) split the ``#fragment`` off and threw it away, and only ever logged, so it could
not fail a build. Ten dead anchors and four dead paths had accumulated behind it; this test found them.

Deliberately offline, so it can run on every PR in under a second: only the local half of a link is
resolved. HTTP reachability is *not* asserted -- someone else's 404 is not a defect in this repo, and a
network hiccup would turn the suite into a liar. Files come from ``git ls-files``, which also excludes
the vendored checkouts (``peft/``, ``transformers/``, ``twinkle/``) that carry their own ``.git``.

Anchors follow GitHub's slugger: lowercase, punctuation dropped, whitespace to ``-``, repeated headings
suffixed ``-1``, ``-2``. Letters outside ASCII survive, so ``## vLLM参数`` is reachable as ``#vllm参数``
-- while ``### qwen3_vl, qwen3_5`` is *not* reachable as ``#qwen3_vl,qwen3_5``, which is exactly the
near-miss that lets a dead link keep looking like a link.

These docs are read through two renderers that disagree about ``_``: GitHub keeps it, while the
published Sphinx/MyST build turns it into ``-`` (measured: the heading above is ``#qwen3_vl-qwen3_5`` on
github.com and ``#qwen3-vl-qwen3-5`` on swift.readthedocs.io). A fragment is therefore accepted if
*either* renderer resolves it -- demanding both would outlaw linking to any heading containing an
underscore, and every anchor link in the repo today satisfies both anyway.
"""
import re
import subprocess
import unicodedata
from collections import Counter
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
from typing import Iterator, List, NamedTuple, Set, Tuple
from urllib.parse import unquote, urlparse

REPO = Path(__file__).resolve().parents[4]


class LocalLink(NamedTuple):
    """A link whose target this repo is responsible for."""

    source: Path
    lineno: int
    target: str
    dest: Path
    fragment: str

    def where(self) -> str:
        return f'{self.source.relative_to(REPO)}:{self.lineno} -> {self.target}'


class Markdown:
    """Read markdown the way a link resolver has to: fenced code is not prose."""

    FENCE = re.compile(r'^\s*(```|~~~)')
    #: ``[text](target)``, ``![alt](target)``, with an optional ``<...>`` wrapper and title.
    LINK = re.compile(r'!?\[(?:[^\]\[]|\[[^\]]*\])*\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\s*\)')
    #: Raw HTML is how the READMEs do their banners and their language switch.
    HTML_LINK = re.compile(r'(?:src|href)\s*=\s*["\']([^"\']+)["\']', re.I)
    HEADING = re.compile(r'^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$')
    HTML_ANCHOR = re.compile(r'<a\s[^>]*(?:name|id)\s*=\s*["\']([^"\']+)["\']', re.I)

    @staticmethod
    def files() -> List[Path]:
        """The repo's own markdown, per git -- so scratch files and vendored subrepos stay out."""
        listed = subprocess.run(['git', 'ls-files', '*.md'], cwd=REPO, capture_output=True, text=True, check=True)
        return [REPO / line for line in listed.stdout.splitlines()]

    @staticmethod
    def body(path: Path) -> List[Tuple[int, str]]:
        """``(lineno, text)`` for every line outside a fenced code block.

        Fences have to go first: the shell samples in these docs are full of ``# comments`` that would
        otherwise register as headings, and of quoted paths that would register as links.
        """
        out, fence = [], None
        for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            opening = Markdown.FENCE.match(line)
            if fence is None:
                if opening:
                    fence = opening.group(1)
                else:
                    out.append((lineno, line))
            elif line.strip().startswith(fence):
                fence = None
        return out

    @staticmethod
    def slug(text: str) -> str:
        """GitHub's heading-to-anchor transform, including its treatment of non-ASCII letters.

        ``_`` survives: it is dropped from *emphasis* markup, but model types are full of intraword
        underscores (``qwen3_vl``) which CommonMark leaves alone and the slugger keeps.
        """
        text = re.sub(r'`([^`]*)`', r'\1', text)
        text = re.sub(r'!?\[([^\]]*)\]\([^)]*\)', r'\1', text)
        text = re.sub(r'<[^>]+>|[*~]', '', text)
        keep = [
            char if char.isalnum() or char in '-_' or unicodedata.category(char).startswith('L') else
            '-' if char in ' \t' else '' for char in text.strip().lower()
        ]
        return ''.join(keep)

    @staticmethod
    @lru_cache(maxsize=None)
    def anchors(path: Path) -> Set[str]:
        """Every fragment that resolves inside ``path``: heading slugs plus explicit HTML anchors."""
        seen: Counter = Counter()
        found = set()
        for _, line in Markdown.body(path):
            heading = Markdown.HEADING.match(line)
            if heading:
                base = Markdown.slug(heading.group(2))
                # A repeated heading is reachable as slug, slug-1, slug-2 ... in document order.
                slug = base if not seen[base] else f'{base}-{seen[base]}'
                seen[base] += 1
                # Both dialects of the same heading: GitHub's, and the Sphinx build's '_' -> '-'.
                found.update({slug, slug.replace('_', '-')})
            found.update(name.lower() for name in Markdown.HTML_ANCHOR.findall(line))
        return found

    @staticmethod
    def links(path: Path) -> Iterator[Tuple[int, str]]:
        for lineno, line in Markdown.body(path):
            for target in Markdown.LINK.findall(line) + Markdown.HTML_LINK.findall(line):
                yield lineno, target

    @staticmethod
    @lru_cache(maxsize=1)
    def local_links() -> Tuple[LocalLink, ...]:
        """Every link in the repo that resolves to a path here -- ``http(s)`` and other schemes are out."""
        out = []
        for source in Markdown.files():
            for lineno, target in Markdown.links(source):
                if urlparse(target).scheme:
                    continue
                path, _, fragment = target.partition('#')
                path = unquote(path)
                # A leading '/' is repo-absolute on GitHub, not filesystem-absolute.
                dest = (REPO / path.lstrip('/')) if path.startswith('/') else (source.parent / path)
                out.append(LocalLink(source, lineno, target, dest if path else source, unquote(fragment)))
        return tuple(out)


def test_markdown_is_being_read():
    """Guard the two tests below against a silently empty file list, which would make them vacuous."""
    assert len(Markdown.files()) > 50, 'git listed almost no markdown -- is this a git checkout?'
    assert len(Markdown.local_links()) > 100, 'no local links were extracted -- the link regexes broke'


def test_link_targets_exist():
    """Every local target -- file, directory or image -- must be on disk."""
    missing = [link.where() for link in Markdown.local_links() if not link.dest.exists()]
    assert not missing, 'markdown links point at paths that do not exist:\n  ' + '\n  '.join(missing)


def test_anchors_exist():
    """Every ``#fragment`` into a markdown file must match a heading there."""
    missing = []
    for link in Markdown.local_links():
        if not link.fragment or link.dest.suffix != '.md' or not link.dest.exists():
            continue
        anchors = Markdown.anchors(link.dest)
        if {Markdown.slug(link.fragment), link.fragment.lower()} & anchors:
            continue
        near = get_close_matches(Markdown.slug(link.fragment), anchors, n=3, cutoff=0.6)
        missing.append(f'{link.where()}{"  (did you mean: " + ", ".join(near) + ")" if near else ""}')
    assert not missing, 'markdown links point at anchors that do not exist:\n  ' + '\n  '.join(missing)
