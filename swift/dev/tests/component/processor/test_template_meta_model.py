"""Offline builder regressions: real family rendering and HF models built from config on meta.

No checkpoint, tokenizer download, legacy registry, or mock model factory is needed.
"""

import copy
import pickle
import socket
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import AutoConfig, PreTrainedModel, Qwen2_5_VLConfig

from swift.dev.builders.template import build_template
from swift.dev.config import TemplateConfig
from swift.dev.model.loader import LoaderModelInfo, ModelLoader, get_model_loader
from swift.dev.template import INFER


class _Tokenizer:
    """A picklable character tokenizer; the native family/Jinja renderers are not mocked."""

    eos_token = "<eos>"
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(char) + 16 for char in text]

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text)}

    def decode(self, ids, **kwargs):
        return "".join(self.eos_token if token == self.eos_token_id else chr(token - 16) for token in ids)

    def convert_tokens_to_ids(self, token):
        return {"<|image_pad|>": 5, "<|video_pad|>": 6, "<|image|>": 5, "<|video|>": 6}.get(token)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        from jinja2 import Template

        text = Template(
            "JINJA|thinking={{ enable_thinking }}|preserve={{ preserve_thinking }}|"
            "{% for m in messages %}{{ m.role }}={{ m.content }}|{% endfor %}"
            "{% if add_generation_prompt %}assistant={% endif %}"
        ).render(messages=messages, add_generation_prompt=add_generation_prompt, **kwargs)
        return self.encode(text) if tokenize else text


class _Unpicklable:
    def __reduce__(self):
        raise AssertionError("process-local model leaked into the serialized template")


def _processor(config=None, model_type="qwen2_5_vl", *, multimodal=False):
    return SimpleNamespace(
        tokenizer=_Tokenizer(),
        model_info=LoaderModelInfo(
            model_type=model_type,
            model_dir="/nonexistent/native-builder-test",
            config=config,
            task_type="causal_lm",
            max_model_len=1024,
            torch_dtype=torch.float32,
        ),
        model_meta=SimpleNamespace(model_type=model_type, is_multimodal=multimodal, template="qwen2_5"),
    )


def _tiny_vl_config():
    # Keep head_dim=8 and mRoPE sections summing to head_dim/2. Both HF4 (flat text
    # config) and HF5 (nested text_config) consume the same explicitly tiny dimensions.
    text = {
        "vocab_size": 256,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    vision = {
        "depth": 1,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 4,
        "in_channels": 3,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": 32,
        "fullatt_block_indexes": [0],
        "window_size": 8,
    }
    return Qwen2_5_VLConfig(
        text_config=text,
        vision_config=vision,
        **text,
        image_token_id=5,
        video_token_id=6,
        vision_start_token_id=3,
        vision_end_token_id=4,
        architectures=["Qwen2_5_VLForConditionalGeneration"],
    )


@pytest.fixture(autouse=True)
def _no_checkpoint_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("template construction must not load checkpoints or access the network")

    monkeypatch.setattr(PreTrainedModel, "from_pretrained", forbidden)
    monkeypatch.setattr(AutoConfig, "from_pretrained", forbidden)
    monkeypatch.setattr(ModelLoader, "build_model", forbidden)
    monkeypatch.setattr(ModelLoader, "build_config", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def test_multimodal_identity_has_one_explicit_source():
    text = build_template(TemplateConfig(template="qwen2_5"), _processor())
    vision = build_template(TemplateConfig(template="qwen2_5_vl"), _processor(_tiny_vl_config(), multimodal=True))
    assert text.media_types == frozenset() and not text.is_multimodal
    assert vision.media_types == frozenset({"image"}) and vision.is_multimodal


def test_builder_forwards_non_default_config():
    config = TemplateConfig(
        template="qwen3",
        agent_template="qwen3_coder",
        system="Configured system",
        max_length=512,
        truncation_strategy="left",
        max_pixels=100,
        norm_bbox="none",
        use_chat_template=True,
        padding_side="left",
        padding_free=True,
        loss_scale="last_round",
        sequence_parallel_size=2,
        is_binary_loss_scale=False,
        template_backend="jinja",
        response_prefix="CUSTOM",
        enable_thinking=False,
        preserve_thinking=False,
        add_non_thinking_prefix=False,
        disable_ignore_empty_think=True,
    )
    template = build_template(config, _processor())
    assert template.template_meta.template_type == "qwen3"
    assert template.template_meta.default_system == config.system
    for name in (
        "max_length",
        "truncation_strategy",
        "max_pixels",
        "norm_bbox",
        "use_chat_template",
        "padding_side",
        "padding_free",
        "sequence_parallel_size",
        "is_binary_loss_scale",
        "template_backend",
        "response_prefix",
        "enable_thinking",
        "preserve_thinking",
        "add_non_thinking_prefix",
    ):
        assert getattr(template, name) == getattr(config, name), name
    assert template._loss_scale == config.loss_scale
    assert template._agent_template == config.agent_template
    assert template.spec.agent_template == config.agent_template
    assert template.model is template.dummy_model is None
    assert callable(template.model_factory)


def test_template_spec_pickle_and_rebuild_with_supplied_processor():
    from swift.dev.builders import build_template_from_spec

    processor = _processor()
    template = build_template(TemplateConfig(template="qwen2_5", agent_template="hermes"), processor)
    spec = pickle.loads(pickle.dumps(template.spec))
    rebuilt = build_template_from_spec(spec, processor=_processor())
    sample = {"messages": [{"role": "user", "content": "hi"}]}
    assert rebuilt.encode(sample, intent=INFER) == template.encode(sample, intent=INFER)
    assert type(template).from_spec(spec, processor=_processor()).encode(sample, intent=INFER) == template.encode(
        sample, intent=INFER
    )


@pytest.mark.parametrize("use_chat_template", [None, True, False])
def test_use_chat_template_none_keeps_factory_default_and_false_renders_plain(use_chat_template):
    template = build_template(TemplateConfig(template="qwen2_5", use_chat_template=use_chat_template), _processor())
    encoded = template.encode({"messages": [{"role": "user", "content": "hi"}]}, intent=INFER)
    text = template.tokenizer.decode(encoded["input_ids"])
    if use_chat_template is False:
        assert template.use_chat_template is False
        assert text == "hi"
    else:
        assert template.use_chat_template is True
        assert "<|im_start|>user\nhi" in text
        assert text.endswith("<|im_start|>assistant\n")


def test_jinja_receives_system_and_non_default_thinking_settings():
    template = build_template(
        TemplateConfig(
            template="qwen3", system="custom", template_backend="jinja", enable_thinking=False, preserve_thinking=False
        ),
        _processor(),
    )
    encoded = template.encode({"messages": [{"role": "user", "content": "hi"}]}, intent=INFER)
    assert template.tokenizer.decode(encoded["input_ids"]) == (
        "JINJA|thinking=False|preserve=False|system=custom|user=hi|assistant="
    )


@pytest.mark.parametrize("preserve", [False, True])
def test_response_prefix_and_preserve_thinking_change_rendering(preserve):
    template = build_template(
        TemplateConfig(
            template="qwen3",
            response_prefix="CUSTOM",
            preserve_thinking=preserve,
            enable_thinking=False,
            add_non_thinking_prefix=False,
        ),
        _processor(),
    )
    sample = {
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "<think>secret reasoning</think>answer"},
            {"role": "user", "content": "next"},
        ]
    }
    text = template.tokenizer.decode(template.encode(sample, intent=INFER)["input_ids"])
    assert ("secret reasoning" in text) is preserve
    assert text.endswith("CUSTOM")


@pytest.mark.parametrize("add_prefix", [False, True])
def test_non_thinking_prefix_and_binary_loss_scale_affect_encoding(add_prefix):
    template = build_template(
        TemplateConfig(
            template="qwen3",
            add_non_thinking_prefix=add_prefix,
            is_binary_loss_scale=False,
            disable_ignore_empty_think=True,
        ),
        _processor(),
    )
    encoded = template.encode(
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "answer"},
            ]
        }
    )
    assert ("<think>" in template.tokenizer.decode(encoded["input_ids"])) is add_prefix
    assert len(encoded["loss_scale"]) == len(encoded["input_ids"])
    assert encoded["loss_scale"][-1] == 0


@pytest.mark.parametrize("template_type", ["qwen2_5", "deepseek_r1"])
def test_ignore_empty_think_is_not_added_without_hybrid_prefix(template_type):
    template = build_template(TemplateConfig(template=template_type), _processor())
    assert template._loss_scale == "default"


def test_disable_ignore_empty_think_controls_automatic_loss_scale_only():
    sample = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "<think>\n\n</think>\n\nanswer"},
        ]
    }
    results = []
    for disable in (False, True):
        template = build_template(TemplateConfig(template="qwen3", disable_ignore_empty_think=disable), _processor())
        encoded = template.encode(copy.deepcopy(sample))
        supervised = template.tokenizer.decode([token for token in encoded["labels"] if token != -100])
        assert ("<think>" in supervised) is disable
        assert "answer" in supervised
        results.append(encoded["input_ids"])
    assert results[0] == results[1]
    for disable in (False, True):
        explicit = build_template(
            TemplateConfig(
                template="qwen3", loss_scale="last_round+ignore_empty_think", disable_ignore_empty_think=disable
            ),
            _processor(),
        )
        assert explicit._loss_scale == "last_round+ignore_empty_think"


def _position_row(with_image):
    if with_image:
        return {"input_ids": [10, 3, 5, 5, 5, 5, 4, 11], "image_grid_thw": [[1, 4, 4]]}
    return {"input_ids": [10, 11, 12, 13]}


def _prepare(template, with_image):
    row = _position_row(with_image)
    if with_image and template.requires_mm_token_type_ids:
        row["mm_token_type_ids"] = [0, 0, 1, 1, 1, 1, 0, 0]
    result = template.prepare_row(row)
    assert result["position_ids"].shape == (4, 1, len(row["input_ids"]))
    assert result["position_ids"].device.type == "cpu"
    assert result["position_ids"].dtype == torch.long
    torch.testing.assert_close(result["position_ids"][0, 0], torch.arange(len(row["input_ids"])))
    if not with_image:
        for axis in result["position_ids"]:
            torch.testing.assert_close(axis[0], torch.arange(len(row["input_ids"])))
    return result["position_ids"]


@pytest.mark.parametrize("identity_source", ["model_info", "model_meta", "architectures", "config_type"])
@pytest.mark.parametrize("with_image", [False, True])
def test_real_qwen_meta_factory_is_lazy_cached_and_pickle_rebuilds(identity_source, with_image):
    config = _tiny_vl_config()
    processor = _processor(config, multimodal=True)
    if identity_source != "model_info":
        processor.model_info.model_type = None
    if identity_source in ("architectures", "config_type"):
        processor.model_meta.model_type = None
    if identity_source == "config_type":
        config.architectures = None
    config_before = copy.deepcopy(config.to_dict())
    loader_cls = get_model_loader("qwen2_5_vl")
    with patch.object(loader_cls, "resolve_model_cls", wraps=loader_cls.resolve_model_cls) as resolve:
        template = build_template(TemplateConfig(template="qwen2_5_vl"), processor)
        assert resolve.call_count == 0
        expected = _prepare(template, with_image)
        meta_model = template.dummy_model
        assert type(meta_model).__name__ == "Qwen2_5_VLForConditionalGeneration"
        assert all(parameter.is_meta for parameter in meta_model.parameters())
        assert template.model is None
        torch.testing.assert_close(_prepare(template, with_image), expected)
        assert template.dummy_model is meta_model
        assert resolve.call_count == 1

        # Both live slots must be dropped; the partial may capture neither of these modules.
        meta_model.serialization_guard = _Unpicklable()
        real_model = torch.nn.Linear(2, 2)
        real_model.serialization_guard = _Unpicklable()
        template.bind_model(real_model)
        assert template._get_model() is real_model
        restored = pickle.loads(pickle.dumps(template))
        assert restored.model is restored.dummy_model is None
        torch.testing.assert_close(_prepare(restored, with_image), expected)
        torch.testing.assert_close(_prepare(restored, with_image), expected)
        assert resolve.call_count == 2
        assert restored.dummy_model is not meta_model
        assert all(parameter.is_meta for parameter in restored.dummy_model.parameters())
        assert template.model is real_model
    assert config.to_dict() == config_before


def test_meta_factory_supports_glm_auto_class_and_survives_direct_pickle():
    from transformers import Glm4Config

    config = Glm4Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    template = build_template(TemplateConfig(template="glm4"), _processor(config, "glm4"))
    factory = pickle.loads(pickle.dumps(template.model_factory))
    model = factory()  # Also meta when called outside the family's device context.
    assert type(model).__name__ == "Glm4ForCausalLM"
    assert all(parameter.is_meta for parameter in model.parameters())


def test_meta_factory_rebuilds_in_fresh_worker():
    import subprocess
    import sys

    import json

    import swift.dev.tests.component.processor.test_template_meta_model as helpers

    template = build_template(
        TemplateConfig(template="qwen2_5_vl"), helpers._processor(helpers._tiny_vl_config(), multimodal=True)
    )
    expected = helpers._prepare(template, True)
    template.dummy_model.serialization_guard = _Unpicklable()
    script = """import json, pickle, socket, sys, torch
from transformers import AutoConfig, PreTrainedModel
from swift.dev.tests.component.processor.test_template_meta_model import _prepare
def forbidden(*args, **kwargs):
    raise AssertionError("worker must not load weights or access the network")
socket.socket.connect = forbidden
AutoConfig.from_pretrained = forbidden
PreTrainedModel.from_pretrained = forbidden
template = pickle.loads(sys.stdin.buffer.read())
assert template.model is template.dummy_model is None
positions = _prepare(template, True)
assert all(p.is_meta for p in template.dummy_model.parameters())
print(json.dumps(positions.tolist()))
"""
    child = subprocess.run(
        [sys.executable, "-B", "-c", script], input=pickle.dumps(template), capture_output=True, timeout=90
    )
    assert child.returncode == 0, child.stderr.decode()[-3000:]
    assert json.loads(child.stdout.decode().strip().splitlines()[-1]) == expected.tolist()


def test_forward_hook_uses_its_real_model_not_the_meta_factory():
    template = build_template(TemplateConfig(template="qwen2_5"), _processor())
    real_model = torch.nn.Linear(2, 2)
    real_model.device = torch.device("cpu")
    with patch.object(template, "_post_encode", return_value={}) as post_encode:
        with patch.object(template, "model_factory", side_effect=AssertionError("not a forward model")):
            template.pre_forward_hook(real_model, (), {"input_ids": torch.tensor([[1, 2]])})
    assert post_encode.call_args.args[0] is real_model
    assert template.model is template.dummy_model is None
