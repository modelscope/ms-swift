"""The builder resolves a dev-native family and exposes it through the stable dev Template API."""

import copy
import pickle
import subprocess
import sys

import pytest

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MSGS = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there"}]


@pytest.fixture(scope="module")
def processor():
    try:
        from swift.model import get_model_processor

        return get_model_processor(MODEL, load_model=False)[1]
    except Exception as exc:  # noqa: BLE001 -- an unimportable legacy stack / unfetchable tokenizer is not a builder defect
        pytest.skip(f"processor not available ({type(exc).__name__}: {exc})")


@pytest.fixture(scope="module")
def template(processor):
    from swift.dev.builders import build_template
    from swift.dev.config import TemplateConfig

    return build_template(TemplateConfig(template="qwen2_5", max_length=256), processor)


def _legacy(processor, mode="train"):
    from swift.template import get_template

    legacy = get_template(processor, template_type="qwen2_5", max_length=256)
    legacy.set_mode(mode)
    return legacy


def test_train_encode_is_legacy_plus_next_token_shift(template, processor):
    """dev's whole delta over legacy for TRAIN: the same ids, labels moved one position left."""
    from swift.dev.template.base import Template as DevFamilyBase

    assert isinstance(template, DevFamilyBase)
    assert type(template).__module__.startswith("swift.dev.template.")
    encoded = template.encode({"messages": copy.deepcopy(MSGS)})
    legacy_encoded = _legacy(processor).encode({"messages": copy.deepcopy(MSGS)})
    assert list(encoded["input_ids"]) == list(legacy_encoded["input_ids"])
    assert list(encoded["labels"]) == list(legacy_encoded["labels"])[1:] + [-100]


def test_intents_map_to_legacy_modes_without_mutating_the_template(template, processor):
    """Intent is per call: an INFER encode in between must not change what TRAIN produces."""
    from swift.dev.template import INFER, SCORE

    before = template.encode({"messages": copy.deepcopy(MSGS)})
    prompt = template.encode({"messages": copy.deepcopy(MSGS[:1])}, intent=INFER)
    legacy_prompt = _legacy(processor, "transformers").encode({"messages": copy.deepcopy(MSGS[:1])})
    assert list(prompt["input_ids"]) == list(legacy_prompt["input_ids"])
    assert prompt.get("labels") is None
    scored = template.encode({"messages": copy.deepcopy(MSGS)}, intent=SCORE)
    assert scored.get("labels") is None
    after = template.encode({"messages": copy.deepcopy(MSGS)})
    assert before == after
    assert template.mode == "train"


def test_sampler_api_returns_canonical_prompt_and_one_completion_result(template):
    prompt = template.encode_prompt({"messages": copy.deepcopy(MSGS[:1])}, backend="vllm")
    assert prompt.state.feature["input_ids"]
    assert "messages" not in prompt.state.feature and "tools" not in prompt.state.feature
    assert list(prompt.state.engine_input_ids) == prompt.state.feature["input_ids"]
    assert not prompt.state.media

    tokens = template.tokenizer.encode("done", add_special_tokens=False)
    completion = template.append_completion(prompt.state, tokens)
    assert completion.state.feature["input_ids"][-len(tokens) :] == tokens
    assert completion.response.content == "done"
    assert completion.state.messages[-1] == {"role": "assistant", "content": "done"}


def test_add_generation_prompt_is_honoured_or_rejected(template):
    """twinkle's flag selects INFER; a flag that contradicts the messages raises instead of being ignored."""
    prompt = template.encode({"messages": copy.deepcopy(MSGS[:1])}, add_generation_prompt=True)
    assert prompt.get("labels") is None
    with pytest.raises(ValueError, match="already ends with an assistant turn"):
        template.encode({"messages": copy.deepcopy(MSGS)}, add_generation_prompt=True)
    with pytest.raises(ValueError, match="ends with a 'user' turn"):
        template.encode({"messages": copy.deepcopy(MSGS[:1])}, add_generation_prompt=False)


def test_encode_routes_preference_by_type(template):
    """Only explicit runtime types select preference encoding; raw dicts stay SFT rows."""
    from swift.dev.template import INFER, KTOSample, PreferenceSample, TemplateInputs

    row = {"messages": copy.deepcopy(MSGS), "rejected_response": "go away"}
    flat = template.encode(copy.deepcopy(row), return_length=True)
    assert flat == template.encode({"messages": copy.deepcopy(MSGS)}, return_length=True)
    assert template.encode(TemplateInputs.from_dict(copy.deepcopy(row)), return_length=True) == flat

    preference = PreferenceSample(
        chosen={"messages": copy.deepcopy(MSGS)},
        rejected={"messages": copy.deepcopy(MSGS[:-1]) + [{"role": "assistant", "content": "go away"}]},
    )
    pair = template.encode(preference)
    assert pair == template.encode_preference(copy.deepcopy(row))
    assert pair["chosen"]["input_ids"] != pair["rejected"]["input_ids"]
    with pytest.raises(ValueError, match="labelled intent"):
        template.encode(preference, intent=INFER)

    kto = template.encode(KTOSample(conversation={"messages": copy.deepcopy(MSGS)}, label=True))
    assert kto["chosen"] is not None and kto["rejected"] is None and kto["label"] is True
    with pytest.raises(ValueError, match="requires a boolean label"):
        template.encode_preference({"messages": copy.deepcopy(MSGS)}, kto=True)


def test_encode_accepts_message_list_and_std_inputs(template):
    from swift.dev.template import StdTemplateInputs

    expected = template.encode({"messages": copy.deepcopy(MSGS)})
    assert template.encode(copy.deepcopy(MSGS)) == expected
    assert template.encode(StdTemplateInputs.from_dict({"messages": copy.deepcopy(MSGS)})) == expected
    with pytest.raises(TypeError, match="Cannot encode"):
        template.encode("hi")


def test_template_pickles_in_a_fresh_process(template):
    """datasets.map workers are fresh interpreters: the template must load there by plain import."""
    import subprocess
    import sys

    blob = pickle.dumps(template)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pickle, sys;"
            "t = pickle.loads(sys.stdin.buffer.read());"
            'm = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there"}];'
            'print(t.encode({"messages": m})["input_ids"])',
        ],
        input=blob,
        capture_output=True,
        timeout=600,
    )
    assert child.returncode == 0, f"a fresh process could not use the pickled template:\n{child.stderr.decode()[-800:]}"
    expected = template.encode({"messages": copy.deepcopy(MSGS)})["input_ids"]
    assert child.stdout.decode().strip().splitlines()[-1] == str(list(expected))


def test_replace_derives_without_touching_the_original(template):
    derived = template.replace(max_length=None)
    assert derived.max_length is None
    assert template.max_length == 256
    with pytest.raises(TypeError):
        template.replace(padding_free=True)


def test_dev_registry_covers_every_legacy_template():
    from swift.dev.template.register import TEMPLATE_MAPPING as DEV_MAPPING
    from swift.dev.template.register import _ensure_registered
    from swift.template import TEMPLATE_MAPPING as LEGACY_MAPPING

    _ensure_registered()
    assert set(DEV_MAPPING) == set(LEGACY_MAPPING)
    assert all(meta.template_cls.__module__.startswith("swift.dev.template.") for meta in DEV_MAPPING.values())


def test_one_template_class_backs_every_family():
    import swift.dev.template as package
    from swift.dev.template.base import Template
    from swift.dev.template.register import TEMPLATE_MAPPING, _ensure_registered

    _ensure_registered()
    assert package.Template is Template
    assert all(issubclass(meta.template_cls, Template) for meta in TEMPLATE_MAPPING.values())


def test_dev_template_package_import_boundary():  # noqa: C901
    import ast
    from pathlib import Path

    import swift.dev.template as package

    #: The one package the template may reach for by import spec instead of a static import: a
    #: fresh process has no builder until ``from_spec`` resolves it, and a static edge would make the
    #: template package depend on builders. Every other string import is a violation.
    deferred = {"swift.dev.builders.template"}
    used = set()

    class RuntimeImportVisitor(ast.NodeVisitor):
        def __init__(self, path, root):
            self.path = path
            self.root = root
            self.violations = []

        @staticmethod
        def _is_type_checking(node):
            return (isinstance(node, ast.Name) and node.id == "TYPE_CHECKING") or (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "typing"
                and node.attr == "TYPE_CHECKING"
            )

        def visit_If(self, node):
            if self._is_type_checking(node.test):
                for statement in node.orelse:
                    self.visit(statement)
                return
            self.generic_visit(node)

        def _check(self, module, lineno):
            if module in deferred:
                used.add(module)
                return
            if (
                module == "swift.dev.template"
                or module.startswith("swift.dev.template.")
                or module == "swift.dev.template_contract"
                or module.startswith("swift.dev.template_contract.")
            ):
                return
            if module == "swift" or module.startswith("swift."):
                relative = self.path.relative_to(self.root)
                self.violations.append(f"{relative}:{lineno}: {module}")

        def visit_Import(self, node):
            for alias in node.names:
                self._check(alias.name, node.lineno)

        def visit_ImportFrom(self, node):
            if node.level == 0 and node.module:
                self._check(node.module, node.lineno)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "import_module" and node.args:
                module = node.args[0]
                if isinstance(module, ast.Constant) and isinstance(module.value, str):
                    self._check(module.value, node.lineno)
            self.generic_visit(node)

    root = Path(package.__file__).parent
    violations = []
    for path in root.rglob("*.py"):
        visitor = RuntimeImportVisitor(path, root)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        violations.extend(visitor.violations)
    assert not violations, "\n".join(violations)
    assert used == deferred, f"deferred-import allow-list out of sync: used={sorted(used)}"


@pytest.mark.slow
def test_template_from_spec_works_in_fresh_process_without_prior_build():
    code = (
        "from swift.dev.builders import build_template_spec; "
        "from swift.dev.config import ModelConfig, TemplateConfig; "
        "from swift.dev.template import Template; "
        "spec = build_template_spec(TemplateConfig(template='qwen2_5'), "
        "ModelConfig(model='Qwen/Qwen2.5-0.5B-Instruct')); "
        "assert Template.from_spec(spec).template_meta.template_type == 'qwen2_5'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.slow
def test_template_spec_rebuilds_processor_through_dev_loader(processor):
    from swift.dev.builders import build_template, build_template_from_spec
    from swift.dev.config import TemplateConfig
    from swift.dev.template import INFER

    template = build_template(TemplateConfig(template="qwen2_5"), processor)
    rebuilt = build_template_from_spec(template.spec)
    sample = {"messages": [{"role": "user", "content": "hi"}]}
    assert rebuilt.encode(sample, intent=INFER)["input_ids"] == template.encode(sample, intent=INFER)["input_ids"]


@pytest.mark.slow
def test_qwen35_template_spec_rebuilds_without_processor_state_kwargs():
    from swift.dev.builders import build_template, build_template_from_spec
    from swift.dev.config import TemplateConfig
    from swift.model import get_model_processor

    _, processor = get_model_processor("Qwen/Qwen3.5-0.8B", load_model=False)
    template = build_template(TemplateConfig(template="qwen3_5"), processor)
    assert dict(template.spec.model.processor_kwargs) == {}
    assert build_template_from_spec(template.spec).template_meta.template_type == "qwen3_5"


def test_unwired_family_capabilities_are_explicit():
    from swift.dev.template import TemplateCapability
    from swift.dev.template.families.baai import Emu3GenTemplate
    from swift.dev.template.families.baidu import ERNIE_VLTemplate
    from swift.dev.template.families.deepseek import DeepseekJanus, UnlimitedOCR
    from swift.dev.template.families.gemma import DiffusionGemmaTemplate
    from swift.dev.template.families.molmo import MolmoTemplate
    from swift.dev.template.families.qwen import (
        Qwen2_5OmniTemplate,
        Qwen3RerankerTemplate,
        Qwen3TTSTemplate,
        QwenPRMTemplate,
    )

    expected = {
        DeepseekJanus: {TemplateCapability.HF_GENERATE},
        UnlimitedOCR: {TemplateCapability.HF_GENERATE, TemplateCapability.OUTPUT_POSTPROCESS},
        ERNIE_VLTemplate: {TemplateCapability.HF_GENERATE},
        Qwen2_5OmniTemplate: {TemplateCapability.HF_GENERATE},
        MolmoTemplate: {TemplateCapability.HF_GENERATE},
        Qwen3TTSTemplate: {TemplateCapability.CUSTOM_COLLATOR, TemplateCapability.CUSTOM_SFT_LOSS},
        DiffusionGemmaTemplate: {TemplateCapability.CUSTOM_SFT_LOSS},
        QwenPRMTemplate: {TemplateCapability.PRM_DECODE},
        Emu3GenTemplate: {TemplateCapability.GENERATE_KWARGS},
        Qwen3RerankerTemplate: {TemplateCapability.ENGINE_KWARGS},
    }
    assert sum(map(len, expected.values())) == 12
    for template_cls, capabilities in expected.items():
        assert template_cls.unsupported_capabilities == frozenset(capabilities)
        template = object.__new__(template_cls)
        template.unsupported_capabilities = template_cls.unsupported_capabilities
        with pytest.raises(NotImplementedError, match=next(iter(capabilities)).value):
            template.require_capabilities(*capabilities)


def test_covered_family_overrides_are_removed():
    from swift.dev.template.families.glm import GLM4vPackingTemplateMixin
    from swift.dev.template.families.internvl import InternvlTemplate
    from swift.dev.template.families.minicpm import MiniCPMVTemplate
    from swift.dev.template.families.qwen import Qwen2VLTemplate

    assert "packing_row" not in vars(Qwen2VLTemplate)
    assert "packing_row" not in vars(GLM4vPackingTemplateMixin)
    assert "forward_context" not in vars(Qwen2VLTemplate)
    assert "forward_context" not in vars(InternvlTemplate)
    assert "prepare_lmdeploy_turbomind_inputs" not in vars(MiniCPMVTemplate)


def test_template_contract_is_a_lightweight_leaf_package():
    import ast
    from pathlib import Path

    import swift.dev.template_contract as contract

    root = Path(contract.__file__).parent
    violations = []
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            for module in modules:
                if module == "swift" or module.startswith(("swift.", "twinkle")):
                    violations.append(f"{path.name}:{node.lineno}: {module}")
    assert not violations, "\n".join(violations)

    code = (
        "import sys; import swift.dev.template_contract; "
        "assert 'swift.dev.template' not in sys.modules; "
        "assert 'librosa' not in sys.modules; assert 'av' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
