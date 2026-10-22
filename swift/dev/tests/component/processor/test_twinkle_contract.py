"""dev's side of the twinkle contract: the consumers dev relies on, plus the template shape it supplies.

Two directions, one file, because they only make sense together:

1. TWINKLE_SURFACE -- what dev's runtime *consumes* from twinkle (``Template`` methods, ``InputProcessor``
   members, the tool-parser registry). If twinkle's ``dev-swift-v5`` renames, moves or drops one of these,
   dev breaks at runtime on some code path; this test names the consumer so the failure is actionable.
   Update the pin together with a deliberate dev-side adaptation -- never silently.

2. TEMPLATE_MUST_HAVE -- what dev must *provide* to be handed to twinkle (``set_template`` / any
   sampler): twinkle calls ``encode_prompt`` / ``append_completion`` / ``decode_response`` /
   ``parse_tool_call`` (vllm_sampler.py:325/294, sampler/base.py:113, sampler/generation.py:53) and
   ``batch_encode`` from ``Model.forward``'s ``_not_encoded`` fallback. Asserted against the *twinkle*
   base class's attribute set, so it also proves the signature family dev promises still exists.

Surface twinkle has but dev deliberately does NOT provide is enumerated in KNOWN_DIVERGENCE with its
reason -- a future reader can tell "not needed yet" from "forgotten".

Run (against a twinkle checkout of the matching line):

    PYTHONPATH="<twinkle>/src:<repo>" python -m pytest -m "not slow" \
        swift/dev/tests/component/processor/test_twinkle_contract.py
"""

import pytest

#: ``(class, member, dev consumer)`` for every twinkle symbol dev's runtime touches.
TWINKLE_SURFACE = (
    # -- Template: the encode entry points twinkle's Model and samplers call --------------------------
    ("twinkle.template.Template", "encode", "sampler/generation.py:53, vllm_sampler.py:217, sglang:154"),
    ("twinkle.template.Template", "batch_encode", "model/transformers.py:669 (_not_encoded fallback)"),
    ("twinkle.template.Template", "pre_forward_hook", "model/transformers.py:540"),
    ("twinkle.template.Template", "is_multimodal", "model/transformers.py:_resolve_mm_graph_context"),
    ("twinkle.template.Template", "encode_prompt", "vllm_sampler.py/sglang_sampler.py trajectory path"),
    ("twinkle.template.Template", "append_completion", "sampler generation/response attachment"),
    ("twinkle.template.Template", "decode_response", "sampler/base.py:decode_response"),
    ("twinkle.template.Template", "decode_message", "partial_rollout.py:merged response"),
    # -- InputProcessor: the base class dev's processor builds on -------------------------------------
    ("twinkle.processor.InputProcessor", "padding_map", "processor/base.py:51-55"),
    ("twinkle.processor.InputProcessor", "VLM_CONCAT_FIELDS", "processor/base.py (media concat boundary)"),
    ("twinkle.processor.InputProcessor", "prepare_inputs", "processor/base.py:100 (super())"),
    ("twinkle.processor.InputProcessor", "collate_fn", "processor/base.py:203 (super())"),
    ("twinkle.processor.InputProcessor", "to_transformers_dict", "processor/base.py:240 (super())"),
    ("twinkle.processor.InputProcessor", "pad_cp", "processor/base.py:266 (super())"),
    ("twinkle.processor.InputProcessor", "split_cp", "processor/base.py:269 (super())"),
    ("twinkle.processor.InputProcessor", "_get_packed_seq_params", "processor/base.py:63 (super())"),
)

#: What dev's Template must provide for twinkle to drive it.
TEMPLATE_MUST_HAVE = (
    "encode",
    "batch_encode",
    "pre_forward_hook",
    "is_multimodal",
    "encode_prompt",
    "append_completion",
    "finalize_appended_features",
    "decode_response",
    "decode_message",
)

#: Exists on twinkle's Template but absent by design on dev's -- the consumer is a twinkle path dev
#: does not enter today. Keep the pins so "absent" stays an explicit decision.
KNOWN_DIVERGENCE = {
    "check": "twinkle dataset/lazy_dataset.py:217; dev drives its own dataset/loader, never twinkle Dataset",
    "batch_check": "twinkle dataset/base.py:174; same reason as check",
    "map_col_to_row": "twinkle Dataset columnar input; dev passes row lists only",
    "map_row_to_col": "twinkle Dataset columnar output; same reason",
    "_add_attention_fields": "twinkle server/common/datum.py:54; dev does not serve through that path",
}


#: Members dev's InputProcessor overrides that twinkle no longer defines (or calls). Each entry must
#: stay DEAD -- if twinkle brings the name back, this test fails so the override can be re-evaluated
#: (and removed if it turns out redundant with postprocess_tensor_cp / postprocess_tensor_sp).
DEAD_OVERRIDES = {
    "postprocess_tensor_gather": "no caller nor definition anywhere in twinkle dev-swift-v5 (grepped src/); "
    "twinkle reshapes via postprocess_tensor_cp / postprocess_tensor_sp",
}


def _resolve(dotted: str):
    import importlib

    module, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(module), name)


def test_dev_template_provides_required_surface():
    """Every member twinkle calls on a stored template exists on dev's Template -- or is a pinned divergence."""
    from swift.dev.template import Template

    missing_impl, missing_twinkle, diverged = [], [], []
    for member in TEMPLATE_MUST_HAVE:
        if not hasattr(Template, member):
            missing_impl.append(member)
    assert not missing_impl, (
        f"dev Template lacks {missing_impl}, which twinkle calls on whatever set_template stored "
        "(e.g. vllm_sampler.py -> append_completion)"
    )
    for member, reason in KNOWN_DIVERGENCE.items():
        from twinkle.template import Template as TwinkleTemplate

        if not hasattr(TwinkleTemplate, member):
            missing_twinkle.append(member)
        if hasattr(Template, member):
            diverged.append(f"{member}: {reason}")
    assert not missing_twinkle, (
        f"{missing_twinkle} are pinned as known divergences but twinkle no longer defines them; "
        "drop them from KNOWN_DIVERGENCE"
    )
    assert not diverged, (
        "these were pinned as deliberate gaps but now exist on dev Template; drop them from "
        "KNOWN_DIVERGENCE (or keep and re-justify):\n" + "\n".join(diverged)
    )


def test_twinkle_surface_exists():
    """Every symbol dev reaches for must exist wherever twinkle puts it today."""
    missing = []
    for dotted, member, consumer in TWINKLE_SURFACE:
        try:
            target = _resolve(dotted)
        except Exception as exc:  # noqa: BLE001 -- report the import failure, not the traceback
            missing.append(f"{dotted}: cannot import ({exc}) [consumer: {consumer}]")
            continue
        if not hasattr(target, member):
            missing.append(f"{dotted}.{member} is gone [consumer: {consumer}]")
    assert not missing, (
        "twinkle dev-swift-v5 surface drifted; either update the pin or adapt the consumer:\n" + "\n".join(missing)
    )


def test_processor_contract_members_match_twinkle():
    """The members dev's InputProcessor overrides must still exist upstream, or the override is dead code."""
    from twinkle.processor import InputProcessor as TwinkleIP

    from swift.dev.processor import InputProcessor

    overridden = {"prepare_inputs", "collate_fn", "to_transformers_dict", "pad_cp", "split_cp"}
    for name in overridden:
        assert hasattr(TwinkleIP, name), f"twinkle InputProcessor lost {name}, which dev overrides"
        assert name in vars(InputProcessor), f"dev InputProcessor must override {name} to stay in contract"
    assert "_get_packed_seq_params" not in vars(InputProcessor)
    assert InputProcessor._get_packed_seq_params is TwinkleIP._get_packed_seq_params


@pytest.mark.parametrize(
    "method",
    [
        "encode_prompt",
        "append_completion",
        "render_delta",
        "decode_message",
        "decode_response",
        "finalize_appended_features",
    ],
)
def test_phase3_role_method_signatures_match(method):
    import inspect

    from twinkle.template import Template as TwinkleTemplate

    from swift.dev.template import Template as SwiftTemplate

    swift_params = list(inspect.signature(getattr(SwiftTemplate, method)).parameters)
    twinkle_params = list(inspect.signature(getattr(TwinkleTemplate, method)).parameters)
    assert swift_params == twinkle_params


def test_templates_satisfy_shared_runtime_protocol():
    from twinkle.template import Template as TwinkleTemplate

    from swift.dev.template import Template as SwiftTemplate
    from swift.dev.template_contract import TemplateProtocol

    assert isinstance(object.__new__(SwiftTemplate), TemplateProtocol)
    assert isinstance(object.__new__(TwinkleTemplate), TemplateProtocol)


def test_twinkle_template_encoder_hook_signature():
    """twinkle's batch_encode routes through ``_encode_messages``; dev's injection point must match it."""
    import inspect

    from twinkle.template import Template as TwinkleTemplate

    params = list(inspect.signature(TwinkleTemplate._encode_messages).parameters)
    assert params == ["self", "trajectory", "add_generation_prompt", "kwargs"], (
        f"twinkle changed _encode_messages signature to {params}; update dev Template.batch_encode, which "
        "feeds it via the same (trajectory, add_generation_prompt) call shape"
    )
