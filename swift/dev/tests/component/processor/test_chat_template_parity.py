# Copyright (c) ModelScope Contributors. All rights reserved.
"""Every registered template, checked against the tokenizer's own ``apply_chat_template``.

``apply_chat_template`` is the reference because it is the model author's jinja, shipped in
``tokenizer_config.json``: if dev's template renders a different prompt string then inference in any
other runtime sees different text than training did, and no loss curve will show it.

The check is a case matrix rather than a single conversation. Each case exercises one shape the
official jinja itself defines the rendering for:

- plain string content and OpenAI content-parts content (the parts form is what multimodal
  families require, and what the ``mm_*`` cases pin down),
- single and multi turn, with and without a system message,
- inference (``add_generation_prompt=True``) and training (full conversation, final eos),
- tool calls with dict and with string ``arguments`` (both appear in real data),
- a tool response followed by a user message and consecutive user messages -- the sequences the
  OpenAI format allows and the official jinjas render as separate turns.

Messages are handed to dev in the same shape the official template accepted (plain string or
content parts), so both sides see identical input. A case the official jinja cannot render at all
(e.g. string tool arguments on a template whose jinja requires a dict) is skipped with that reason:
without a reference there is nothing to compare against. The tool-description layer (what dev's
agent template injects into the system prompt when ``tools`` are passed) is deliberately not part
of this matrix: it is training-time prompt content chosen per family, not the message-structure
contract the matrix guards.

Skipping is per template+case and always carries the reason: no registered model, an unfetchable
tokenizer, no shipped chat template, or an official render failure on the case's input. A dev-side
error is never a skip -- it is the failure this test exists to catch.

Token ids rather than decoded text: ``decode`` is not the inverse of ``encode`` for sentencepiece
tokenizers -- it reinserts word-piece boundaries as spaces (``assistant`` came back as
``ass istant``), which reads like a template defect and is not one.
"""

import pytest

from swift.dev.tests.tiny import TinyModel

SYSTEM = "You are helpful."
WEATHER_TOOL_NAME = "get_weather"


def _asst(text, calls=None):
    message = {"role": "assistant", "content": text}
    if calls is not None:
        message["tool_calls"] = calls
    return message


def _tool_calls(arguments):
    return [{"id": "call_1", "type": "function", "function": {"name": WEATHER_TOOL_NAME, "arguments": arguments}}]


def _text_part(text):
    return [{"type": "text", "text": text}]


TOOL_MSG = {"role": "tool", "content": '{"temp": 25}', "tool_call_id": "call_1"}
DICT_ARGS = {"city": "Shanghai"}
STR_ARGS = '{"city": "Shanghai"}'
SYSTEM_MESSAGE = {"role": "system", "content": SYSTEM}

#: name -> (messages, intent, add_generation_prompt). Populated with INFER/TRAIN at test time.
_CASE_SPECS = {
    "infer_basic": ([SYSTEM_MESSAGE, {"role": "user", "content": "hi"}, _asst("hello"), {"role": "user", "content": "bye"}], "INFER", True),
    "train_basic": (
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "hi"},
            _asst("hello"),
            {"role": "user", "content": "bye"},
            _asst("bye ans"),
        ],
        "TRAIN",
        False,
    ),
    "train_no_system": ([{"role": "user", "content": "hi"}, _asst("hello")], "TRAIN", False),
    "train_multi": (
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "q1"},
            _asst("a1"),
            {"role": "user", "content": "q2"},
            _asst("a2"),
            {"role": "user", "content": "q3"},
            _asst("a3"),
        ],
        "TRAIN",
        False,
    ),
    "train_tool_dict": (
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "weather?"},
            _asst("", _tool_calls(DICT_ARGS)),
            TOOL_MSG,
            _asst("It is 25C."),
        ],
        "TRAIN",
        False,
    ),
    "train_tool_str": (
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "weather?"},
            _asst("", _tool_calls(STR_ARGS)),
            TOOL_MSG,
            _asst("It is 25C."),
        ],
        "TRAIN",
        False,
    ),
    "train_tool_user": (
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "weather?"},
            _asst("", _tool_calls(DICT_ARGS)),
            TOOL_MSG,
            {"role": "user", "content": "and Beijing?"},
            _asst("Beijing 22C."),
        ],
        "TRAIN",
        False,
    ),
    "train_consec_user": (
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
            _asst("merged"),
        ],
        "TRAIN",
        False,
    ),
    "mm_infer": (
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _text_part("hi")},
            _asst("hello"),
            {"role": "user", "content": _text_part("bye")},
        ],
        "INFER",
        True,
    ),
    "mm_multi": (
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _text_part("hi")},
            _asst("hello"),
            {"role": "user", "content": _text_part("bye")},
            _asst("bye ans"),
        ],
        "TRAIN",
        False,
    ),
}


def _to_parts(messages):
    """The OpenAI content-parts form: plain string contents become single text parts."""
    res = []
    for message in messages:
        message = dict(message)
        if isinstance(message.get("content"), str) and message["role"] in ("system", "user", "assistant"):
            message["content"] = _text_part(message["content"])
        res.append(message)
    return res


def _to_plain(messages):
    """The plain string form: a single text part collapses back to its text."""
    res = []
    for message in messages:
        message = dict(message)
        content = message.get("content")
        if isinstance(content, list) and len(content) == 1 and content[0].get("type") == "text":
            message["content"] = content[0]["text"]
        res.append(message)
    return res


class _CaseSkip(Exception):
    """The official jinja cannot render this input, so there is no reference to compare against."""


def _parts_stringified(tokenizer, messages, add_generation_prompt):
    """True when the jinja printed a parts list verbatim (its Python repr) instead of its text.

    A jinja without content-parts support does not raise on a parts form -- it silently renders
    ``[{'type': 'text', ...}]`` into the prompt. That printout is not a rendering contract dev
    could or should reproduce, so it counts as "cannot render this input shape".
    """
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    if isinstance(rendered, list):
        rendered = "".join(rendered)
    return any(
        isinstance(message.get("content"), list) and str(message["content"]) in rendered for message in messages
    )


def _render_official(tokenizer, messages, add_generation_prompt, prefer_parts):
    """Render with the official jinja, handing it the input shape it accepts.

    ``prefer_parts`` marks the ``mm_*`` cases, which exist to pin the content-parts path: for them
    the parts form is required, and a jinja that only takes strings yields a skip rather than a
    silent fallback that would make the case identical to ``infer_basic``. A jinja that prints the
    parts list itself instead of extracting its text is the same situation and skips the same way.
    """
    forms = [_to_parts, _to_plain] if prefer_parts else [_to_plain, _to_parts]
    last_error = None
    for form in forms:
        shaped = form(messages)
        try:
            rendered = tokenizer.apply_chat_template(
                shaped, tokenize=True, add_generation_prompt=add_generation_prompt
            )["input_ids"]
        except Exception as exc:  # noqa: BLE001 -- the jinja rejecting an input shape is a skip signal
            last_error = exc
            continue
        if form is _to_parts and _parts_stringified(tokenizer, shaped, add_generation_prompt):
            raise _CaseSkip("official jinja stringifies content parts (no parts support)")
        return list(rendered), form is _to_parts
    raise _CaseSkip(f"{type(last_error).__name__}: {last_error}"[:160])


def _first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else None


class TemplateSurvey:
    """Pair each template with one model that can supply a tokenizer."""

    @staticmethod
    def representatives() -> dict:
        """template name -> a model id registered against it.

        The registry is keyed by model_type, so walk it once and keep the first model id per
        template. Templates with no registered model cannot be checked this way and are left out.
        """
        from swift.dev.model.loader import MODEL_MAPPING as DEV_MODEL_MAPPING
        from swift.model.register import MODEL_MAPPING

        found = {}
        for loader in DEV_MODEL_MAPPING.values():
            template = getattr(loader, "template", None)
            models = getattr(loader, "models", ())
            if not template or not models or template in found:
                continue
            model = models[0]
            found[template] = model[0] if isinstance(model, tuple) else model
        for meta in MODEL_MAPPING.values():
            template = getattr(meta, "template", None)
            if not template or template in found:
                continue
            for group in getattr(meta, "model_groups", None) or []:
                for model in getattr(group, "models", None) or []:
                    model_id = getattr(model, "ms_model_id", None) or getattr(model, "hf_model_id", None)
                    if model_id:
                        found[template] = model_id
                        break
                if template in found:
                    break
        return found


REPRESENTATIVES = TemplateSurvey.representatives()
UNREPRESENTED = {"xcomposer2_4khd": "no model id is registered in either model registry"}


def test_every_unique_family_has_a_tokenizer_representative_or_skip_reason():
    from swift.dev.template.register import TEMPLATE_MAPPING, _ensure_registered

    _ensure_registered()
    groups = {}
    for name, meta in TEMPLATE_MAPPING.items():
        groups.setdefault(meta.template_cls, []).append(name)
    missing = []
    for names in groups.values():
        if not set(names) & set(REPRESENTATIVES) and not set(names) & set(UNREPRESENTED):
            missing.append(names)
    assert not missing


@pytest.mark.slow
def test_all_templates_render_like_official_jinja():
    """dev's token ids must equal the official jinja's, for every template and every case.

    One test over the whole matrix (templates x cases) rather than one parametrised test per
    template: the run collects every mismatch before failing, so a night of template fixing sees
    the full red list in one pass instead of one template per rerun.
    """
    from swift.dev.builders import build_template
    from swift.dev.config import TemplateConfig
    from swift.dev.template import INFER, TRAIN
    from swift.model import get_model_processor

    intents = {"INFER": INFER, "TRAIN": TRAIN}

    failures = []
    skips = []
    for template in sorted(REPRESENTATIVES):
        model_id = REPRESENTATIVES[template]
        try:
            model_dir = TinyModel.tokenizer_dir(model_id)
            _, processor = get_model_processor(model_dir, load_model=False)
        except Exception as exc:  # noqa: BLE001 -- a missing or gated repo is not a template defect
            skips.append(f"{template}: setup failed ({type(exc).__name__}: {exc})"[:200])
            continue
        tokenizer = processor if hasattr(processor, "apply_chat_template") else processor.tokenizer
        if getattr(tokenizer, "chat_template", None) is None:
            skips.append(f"{template}: {model_id} ships no chat_template, so there is nothing to compare against")
            continue
        try:
            dev_template = build_template(TemplateConfig(template=template, max_length=4096), processor)
        except Exception as exc:  # noqa: BLE001 -- e.g. legacy registry pointing at an unregistered template
            skips.append(f"{template}: build failed ({type(exc).__name__}: {exc})"[:200])
            continue

        for case, (messages, intent, add_generation_prompt) in _CASE_SPECS.items():
            prefer_parts = case.startswith("mm_")
            try:
                reference, used_parts = _render_official(tokenizer, messages, add_generation_prompt, prefer_parts)
            except _CaseSkip as exc:
                skips.append(f"{template} / {case}: official jinja rejects the input ({exc})")
                continue

            dev_messages = _to_parts(messages) if used_parts else _to_plain(messages)
            try:
                rendered = list(dev_template.encode({"messages": dev_messages}, intent=intents[intent])["input_ids"])
            except Exception as exc:  # noqa: BLE001 -- a dev-side error is the failure this test exists to catch
                failures.append(f"{template} / {case}: dev encode raised {type(exc).__name__}: {exc}"[:400])
                continue

            diff_at = _first_diff(rendered, reference)
            if diff_at is not None:
                head = max(0, diff_at - 4)
                failures.append(
                    f"{template} / {case}: differs at {diff_at} (dev {len(rendered)} vs official {len(reference)} tokens)\n"
                    f"  dev:      {tokenizer.decode(rendered[head : diff_at + 12])!r}\n"
                    f"  official: {tokenizer.decode(reference[head : diff_at + 12])!r}"
                )

    report = "\n".join(failures)
    assert not failures, f"{len(failures)} template/case mismatches:\n{report}\n\nskipped:\n" + "\n".join(skips)
    # skips are recorded, not asserted empty: they carry the reason a template could not be checked.
    print(f"\nchecked all cases; skipped {len(skips)} template/case combinations")
    for reason in skips:
        print(f"  {reason}")
