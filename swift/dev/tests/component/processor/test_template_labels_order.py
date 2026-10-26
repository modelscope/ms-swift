import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from swift.dev.template.base import Template
from swift.dev.template.families.deepseek import UnlimitedOCR
from swift.dev.template.families.gemma import DiffusionGemmaTemplate
from swift.dev.template.families.internlm import InternLMXComposer2_5Template
from swift.dev.template.families.qwen import Ovis1_6Template


def test_shift_calls_stay_on_the_nominal_row_boundaries():  # noqa: C901
    dev = Path(__file__).parents[3]
    # The twinkle checkout may sit either nested inside ms-swift (<repo>/twinkle) or as a sibling
    # checkout directory; walk up until a `twinkle/src/twinkle` tree is found.
    twinkle = None
    for ancestor in dev.parents:
        candidate = ancestor / "twinkle" / "src" / "twinkle"
        if candidate.is_dir():
            twinkle = candidate
            break
    assert twinkle is not None, 'twinkle/src/twinkle not found above swift/dev'
    names = {"shift", "unshift", "shift_values", "unshift_values"}
    calls = Counter()
    for repository, root in (("swift", dev), ("twinkle", twinkle)):
        for path in root.rglob("*.py"):
            if "tests" in path.parts or "template_contract/encoding.py" in path.as_posix():
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            aliases = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name in names:
                            aliases[alias.asname or alias.name] = alias.name
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        called = aliases.get(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        called = node.func.attr if node.func.attr in names else None
                    else:
                        called = None
                    if called is not None:
                        calls[(repository, str(path.relative_to(root)), called)] += 1
    assert calls == Counter(
        {
            ("swift", "template/base.py", "shift"): 2,
            ("swift", "template/base.py", "unshift"): 1,
            ("swift", "template/families/gemma.py", "unshift_values"): 1,
            ("swift", "template/families/deepseek.py", "unshift_values"): 1,
            ("twinkle", "template/base.py", "shift_values"): 2,
        }
    )


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    @property
    def device(self):
        return self.anchor.device

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None, **kwargs):
        return inputs_embeds


class _ExpandingHookTemplate(Template):
    def __init__(self):
        self.task_type = "causal_lm"
        self.seen_labels = None

    def _post_encode(self, model, inputs):
        self.seen_labels = inputs["labels"].clone()
        labels = torch.cat(
            (inputs["labels"][:, :2], inputs["labels"].new_full((1, 1), -100), inputs["labels"][:, 2:]), dim=1
        )
        return {"inputs_embeds": torch.zeros((1, labels.shape[1], 2)), "labels": labels}

    @staticmethod
    def get_base_model(model):
        return model


def test_pre_forward_hook_exposes_input_order_and_restores_output_order():
    template = _ExpandingHookTemplate()
    model = _Model()
    labels = torch.tensor([[11, 12, 13, -100]])
    kwargs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "labels": labels,
    }

    _, result = template.pre_forward_hook(model, (), kwargs)

    assert template.seen_labels.tolist() == [[-100, 11, 12, 13]]
    assert result["labels"].tolist() == [[11, -100, 12, 13, -100]]
    assert kwargs["labels"] is labels


def test_ovis_hook_receives_input_order_labels():
    template = object.__new__(Ovis1_6Template)
    template.task_type = "causal_lm"
    template.padding_side = "right"
    template.max_length = None
    template.get_base_model = lambda model: model
    model = _Model()
    model.config = SimpleNamespace()
    seen = {}

    def merge_multimodal(**kwargs):
        seen["labels"] = kwargs["text_labels"].clone()
        labels = torch.cat(
            (kwargs["text_labels"][:, :2], kwargs["text_labels"].new_full((1, 1), -100), kwargs["text_labels"][:, 2:]),
            dim=1,
        )
        embeds = torch.zeros((1, labels.shape[1], 2))
        return None, embeds, labels, torch.ones_like(labels)

    model.merge_multimodal = merge_multimodal
    kwargs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[21, 22, -100]]),
        "pixel_values": [torch.zeros((1, 3, 2, 2))],
    }

    _, result = template.pre_forward_hook(model, (), kwargs)

    assert seen["labels"].tolist() == [[-100, 21, 22]]
    assert result["labels"].tolist() == [[21, -100, 22, -100]]


def test_internlm_xcomposer_hook_receives_input_order_labels():
    template = object.__new__(InternLMXComposer2_5Template)
    template.task_type = "causal_lm"
    template.padding_side = "right"
    template.padding_free = False
    template.use_megatron = False
    template.sequence_parallel_size = 1
    template.processor = SimpleNamespace(pad_token_id=0)
    template.get_base_model = lambda model: model
    model = _Model()
    model.model = SimpleNamespace(tok_embeddings=nn.Embedding(100, 2))
    model.img2emb = lambda image: (torch.ones((1, 2, 2)), None)
    kwargs = {
        "input_ids": torch.tensor([[10, 2, 20]]),
        "labels": torch.tensor([[2, 20, -100]]),
        "images": [[torch.zeros((3, 2, 2))]],
        "length": [3],
    }

    _, result = template.pre_forward_hook(model, (), kwargs)

    assert result["labels"].tolist() == [[-100, -100, 20, -100]]
    assert result["inputs_embeds"].shape == (1, 4, 2)


def test_diffusion_gemma_finds_response_boundary_in_input_order():
    template = object.__new__(DiffusionGemmaTemplate)
    template.config = SimpleNamespace(canvas_length=5, text_config=SimpleNamespace(vocab_size=100))
    template.processor = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    inputs = {
        "input_ids": torch.tensor([[10, 11, 12]]),
        "labels": torch.tensor([[11, 12, -100]]),
    }

    result = template._post_collate(inputs)

    assert result["input_ids"].tolist() == [[10]]
    assert result["labels"].tolist() == [[11, 12, 2, -100, -100]]
    assert inputs["labels"].tolist() == [[11, 12, -100]]


def test_unlimited_ocr_reads_input_order_without_replacing_labels():
    template = object.__new__(UnlimitedOCR)
    template._rswa_window = 2
    template.model_info = SimpleNamespace(torch_dtype=torch.float32)
    labels = torch.tensor([[41, 42, -100]])
    attention_mask = torch.ones((1, 3), dtype=torch.long)
    inputs = {"labels": labels, "attention_mask": attention_mask}

    result = template._post_collate(inputs)
    expected = template._build_rswa_attention_mask(torch.tensor([[-100, 41, 42]]), attention_mask, 2, torch.float32)

    assert result["labels"] is labels
    assert torch.equal(result["attention_mask"], expected)
