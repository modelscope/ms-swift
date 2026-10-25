"""S18d/f — template registry + agent-template map structural contracts (L1).

``register_template`` singleton semantics (dup raises, exist_ok re-declares), the agent-template
map's full reachability (32 entries), and the agent protocol's structural pieces (Function
JSON-serializes its arguments; ToolDesc carries the model/human names separately).

Maps to ut_plan S18-18/33/35.
"""
import pytest


class TestRegisterTemplate:

    def test_duplicate_registration_rejected(self):
        """S18-18: registering an existing template_type twice raises -- a silent overwrite could
        shadow a built-in with a half-configured meta."""
        from swift.dev.template.register import TEMPLATE_MAPPING, _ensure_registered, register_template

        _ensure_registered()
        key = next(iter(TEMPLATE_MAPPING))
        with pytest.raises(ValueError, match='TEMPLATE_MAPPING'):
            register_template(TEMPLATE_MAPPING[key])

    def test_exist_ok_redeclares_in_place(self):
        """exist_ok=True re-registers (the refresh path); the map size is stable."""
        from swift.dev.template.register import TEMPLATE_MAPPING, _ensure_registered, register_template

        _ensure_registered()
        key = next(iter(TEMPLATE_MAPPING))
        size = len(TEMPLATE_MAPPING)
        register_template(TEMPLATE_MAPPING[key], exist_ok=True)
        assert len(TEMPLATE_MAPPING) == size


class TestAgentTemplateMap:
    """S18-33: every agent template family is reachable by name from one map."""

    EXPECTED = {
        'chatglm4', 'deepseek_v3_1', 'deepseek_v4', 'gemma4', 'glm4', 'glm4_5', 'glm4_7', 'glm5_1',
        'hermes', 'hunyuan_hermes', 'hy_v3', 'hy_v3_preview', 'kimi_k25', 'kimi_k3', 'llama3',
        'llama4', 'minicpm5', 'minimax_m2', 'minimax_m3', 'mistral', 'qwen3_5', 'qwen3_coder',
        'qwen_en', 'qwen_en_parallel', 'qwen_zh', 'qwen_zh_parallel', 'react_en', 'react_grpo',
        'react_zh', 'seed_oss', 'toolbench', 'youtu',
    }

    def test_map_is_complete(self):
        from swift.dev.template.agent import agent_template_map

        assert set(agent_template_map) == self.EXPECTED

    def test_every_entry_is_a_class(self):
        from swift.dev.template.agent import agent_template_map

        for name, entry in agent_template_map.items():
            assert isinstance(entry, type), f'{name} must map to a class, got {type(entry)}'


class TestAgentProtocolStructures:
    """S18-35: Function / ToolDesc -- the tool-call protocol's data carriers."""

    def test_function_serializes_arguments_to_json(self):
        """Function.arguments accepts a dict and stores it as its JSON string (the wire form tool
        calls carry); a caller passing a str passes through verbatim."""
        import json

        from swift.dev.template.agent.protocol import Function

        fn = Function(name='add', arguments={'a': 1})
        assert fn.name == 'add'
        assert json.loads(fn.arguments) == {'a': 1}  # stored as JSON text
        fn_str = Function(name='add', arguments='{"a": 1}')
        assert fn_str.arguments == '{"a": 1}'

    def test_tool_desc_carries_model_and_human_names(self):
        """ToolDesc keeps separate names: the model-facing (constrained) name and the human-facing
        label, plus the JSON-schema parameters."""
        from swift.dev.template.agent.base import ToolDesc

        desc = ToolDesc('add', 'Add two numbers', 'Add two numbers.',
                        {'type': 'object'}, 'en')
        assert desc.name_for_model == 'add'
        assert desc.name_for_human == 'Add two numbers'
        assert desc.parameters == {'type': 'object'}
        assert desc.args_format == 'en'
