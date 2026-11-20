# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import glob
import json
import os
import torch
import unittest
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, PretrainedConfig
from types import SimpleNamespace

from swift.model.model_meta import ModelInfo
from swift.model.register import MODEL_MAPPING
from swift.template import TEMPLATE_MAPPING, get_template
from swift.template.base import Template

CACHE = os.path.join(os.environ.get('MODELSCOPE_CACHE', os.path.expanduser('~/.cache/modelscope/hub')), 'models')
TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'weather',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {
                    'type': 'string'
                }
            },
            'required': ['city']
        }
    }
}


def _model_id(model):
    return model.ms_model_id or model.hf_model_id or model.model_path


def _local_snapshot(model):
    mid = _model_id(model)
    if not mid or os.path.isabs(mid):
        return mid if mid and os.path.isdir(mid) else None
    base = os.path.join(CACHE, mid.replace('/', '--'))
    snaps = sorted(glob.glob(os.path.join(base, 'snapshots', '*')))
    if snaps:
        return snaps[-1]
    return base if os.path.exists(os.path.join(base, 'config.json')) else None


def _load(model):
    """Load only config/tokenizer/processor from a local snapshot; never download models."""
    source = _local_snapshot(model)
    if source is None:
        raise unittest.SkipTest(f'{_model_id(model)}: local snapshot not available')
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    except Exception as e:
        raise unittest.SkipTest(f'{source}: tokenizer unavailable ({type(e).__name__}: {e})')
    try:
        config = AutoConfig.from_pretrained(source, trust_remote_code=True)
    except Exception:
        config_path = os.path.join(source, 'config.json')
        if not os.path.exists(config_path):
            raise unittest.SkipTest(f'{source}: config.json unavailable')
        try:
            config = PretrainedConfig.from_dict(json.load(open(config_path, encoding='utf-8')))
        except Exception:
            config = PretrainedConfig()
    try:
        processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
    except Exception:
        processor = tokenizer
    if not getattr(tokenizer, 'chat_template', None):
        chat_template = getattr(processor, 'chat_template', None)
        if chat_template:
            tokenizer.chat_template = chat_template
    if not getattr(tokenizer, 'chat_template', None):
        jinja_path = os.path.join(source, 'chat_template.jinja')
        if os.path.exists(jinja_path):
            tokenizer.chat_template = open(jinja_path, encoding='utf-8').read()
    if not getattr(tokenizer, 'chat_template', None):
        raise unittest.SkipTest(f'{source}: no official chat_template (tokenizer or processor)')
    return source, config, tokenizer, processor


def _build(template_type, model_type, source, config, tokenizer, processor, **kwargs):
    info = ModelInfo(
        model_type=model_type,
        model_dir=source,
        torch_dtype=torch.bfloat16,
        max_model_len=32768,
        quant_method=None,
        quant_bits=0,
        config=config,
        task_type='causal_lm')
    processor = SimpleNamespace(
        tokenizer=tokenizer,
        image_processor=getattr(processor, 'image_processor', None),
        model_info=info,
        model_meta=MODEL_MAPPING[model_type])
    return get_template(processor, template_type=template_type, **kwargs)


def _mode_kwargs(mode):
    if mode == 'explicit_false':
        return {'preserve_thinking': False}
    if mode == 'explicit_true':
        return {'preserve_thinking': True}
    return {}


def _render(template_type, model_type, source, config, tokenizer, processor, swift_messages, mode, tools=None):
    kwargs = _mode_kwargs(mode)
    if mode == 'last_round':
        kwargs['loss_scale'] = 'last_round'
    template = _build(template_type, model_type, source, config, tokenizer, processor, **kwargs)
    template.set_mode('transformers' if mode == 'infer_default' else 'train')
    encoded = template.encode({'messages': copy.deepcopy(swift_messages), 'tools': [TOOL] if tools is None else tools})
    return tokenizer.decode(encoded['input_ids'], skip_special_tokens=False)


class TestThinkingPolicyUnits(unittest.TestCase):
    """Policy branches, no model loading."""

    def _fake(self, meta, is_training=True, preserve_thinking=None):

        class _FakeTemplate(Template):

            @property
            def is_training(self):
                return is_training

            @property
            def loss_scale(self):
                return SimpleNamespace(base_strategy='default')

            def _get_enable_thinking(self, inputs=None):
                return False

            def __init__(self):
                self.template_meta = meta
                self.preserve_thinking = preserve_thinking

        return _FakeTemplate()

    def _messages(self):
        return [
            {
                'role': 'user',
                'content': 'q1'
            },
            {
                'role': 'assistant',
                'content': 'a1'
            },
            {
                'role': 'tool',
                'content': 't1'
            },
            {
                'role': 'assistant',
                'content': 'a2'
            },
            {
                'role': 'tool',
                'content': 't2',
                '_tool_user_followup': True
            },
            {
                'role': 'assistant',
                'content': 'a3'
            },
        ]

    def _apply(self, policy, is_training=True):
        meta = copy.deepcopy(TEMPLATE_MAPPING['qwen3'])
        meta.non_thinking_prefix = '<think></think>'
        meta.non_thinking_prefix_policy = policy
        meta.preserve_thinking = True
        messages = self._messages()
        template = self._fake(meta, is_training)
        template._add_non_thinking_prefix(SimpleNamespace(messages=messages))
        return messages

    def test_non_thinking_prefix_all(self):
        messages = self._apply('all')
        self.assertTrue(
            all(message['content'].startswith('<think></think>') for message in messages
                if message['role'] == 'assistant'))

    def test_non_thinking_prefix_last_response(self):
        messages = self._apply('last_response')
        prefixed = [
            i for i, m in enumerate(messages) if m['role'] == 'assistant' and m['content'].startswith('<think>')
        ]
        self.assertEqual(prefixed, [5])

    def test_non_thinking_prefix_after_user(self):
        messages = self._apply('after_user')
        prefixed = [
            i for i, m in enumerate(messages) if m['role'] == 'assistant' and m['content'].startswith('<think>')
        ]
        self.assertEqual(prefixed, [1, 5])

    def test_auto_thinking_leaves_prefix_empty(self):
        meta = copy.deepcopy(TEMPLATE_MAPPING['minicpm5_2b'])

        class _AutoTemplate(Template):

            @property
            def is_training(self):
                return False

            @property
            def loss_scale(self):
                return SimpleNamespace(base_strategy='default')

            def __init__(self):
                self.template_meta = meta
                self.enable_thinking = None
                self.preserve_thinking = None
                self.response_prefix = None
                self.use_chat_template = True

        template = _AutoTemplate()
        self.assertIsNone(template._get_enable_thinking())
        self.assertEqual(template._get_response_prefix(), '')
        self.assertEqual(
            template._get_response_prefix(SimpleNamespace(chat_template_kwargs={'enable_thinking': True})), '<think>\n')
        self.assertEqual(
            template._get_response_prefix(SimpleNamespace(chat_template_kwargs={'enable_thinking': False})),
            '<think>\n\n</think>\n\n')

    def test_preserve_policy_always_default_and_never_all(self):
        always = copy.deepcopy(TEMPLATE_MAPPING['glm5_3'])
        default_policy = copy.deepcopy(TEMPLATE_MAPPING['qwq'])
        self.assertIs(Template._get_preserve_thinking(self._fake(always), None), True)
        self.assertIs(Template._get_preserve_thinking(self._fake(always, preserve_thinking=False), None), False)
        self.assertIs(Template._get_preserve_thinking(self._fake(default_policy, preserve_thinking=False), None), False)
        self.assertIs(Template._get_preserve_thinking(self._fake(default_policy), None), True)
        messages = self._messages()
        self.assertEqual(Template._get_current_round_start(self._fake(default_policy), messages), 4)
        self.assertEqual(Template._get_current_round_start(self._fake(default_policy, is_training=False), messages), 4)
        never_all = copy.deepcopy(TEMPLATE_MAPPING['glm4_z1_rumination'])
        self.assertEqual(
            Template._get_current_round_start(self._fake(never_all, is_training=False), messages), len(messages))
        ends_with_assistant = [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}]
        self.assertEqual(
            Template._get_current_round_start(self._fake(default_policy, is_training=False), ends_with_assistant), 0)
        self.assertEqual(
            Template._get_current_round_start(self._fake(never_all, is_training=False), ends_with_assistant),
            len(ends_with_assistant))

    def test_current_round_start_uses_user_and_followup(self):
        messages = self._messages()
        template = self._fake(copy.deepcopy(TEMPLATE_MAPPING['qwen3']))
        self.assertEqual(Template._get_current_round_start(template, messages), 4)
        messages[4] = {'role': 'tool', 'content': 't2'}
        self.assertEqual(Template._get_current_round_start(template, messages), 0)


class TestThinkingRegressions(unittest.TestCase):
    """Targeted regressions; model loading is local-only."""

    def test_qwq_last_round_keeps_current_thinking(self):
        meta = MODEL_MAPPING['qwen2']
        group = next((g for g in meta.model_groups if (g.template or meta.template) == 'qwq'), None)
        if group is None:
            self.skipTest('qwq group not found')
        try:
            source, config, tokenizer, processor = _load(group.models[0])
        except unittest.SkipTest as e:
            self.skipTest(str(e))
        messages = [
            {
                'role': 'user',
                'content': 'q'
            },
            {
                'role': 'assistant',
                'content': '<think>\nCUR\n</think>\n\nA1'
            },
        ]
        text = _render('qwq', 'qwen2', source, config, tokenizer, processor, messages, 'last_round')
        self.assertIn('CUR', text)
        self.assertIn('<think', text)

    def test_deepseek_plain_history_and_ending_assistant(self):
        meta = MODEL_MAPPING['deepseek_v3']
        group = next((g for g in meta.model_groups if (g.template or meta.template) == 'deepseek_v3_1'), None)
        if group is None:
            self.skipTest('deepseek_v3_1 group not found')
        try:
            source, config, tokenizer, processor = _load(group.models[0])
        except unittest.SkipTest as e:
            self.skipTest(str(e))
        cases = (
            ('explicit_false', [
                {
                    'role': 'user',
                    'content': 'q1'
                },
                {
                    'role': 'assistant',
                    'content': '<think>\nPLAIN_OLD\n</think>\n\nA1'
                },
                {
                    'role': 'user',
                    'content': 'q2'
                },
                {
                    'role': 'assistant',
                    'content': 'A2'
                },
            ], 'PLAIN_OLD'),
            ('infer_default', [
                {
                    'role': 'user',
                    'content': 'q1'
                },
                {
                    'role': 'assistant',
                    'content': '<think>\nEND_OLD\n</think>\n\nA1'
                },
            ], 'END_OLD'),
        )
        for mode, messages, marker in cases:
            official = tokenizer.apply_chat_template(
                copy.deepcopy(messages),
                tools=[TOOL],
                tokenize=False,
                add_generation_prompt=False,
                preserve_thinking=False)
            swift = _render('deepseek_v3_1', 'deepseek_v3', source, config, tokenizer, processor, messages, mode)
            self.assertEqual(marker in swift, marker in official, f'{mode}\nswift={swift}\nofficial={official}')

    def test_deepseek_v4_custom_encoder(self):
        import importlib.util

        tested = 0
        for model_type, meta in MODEL_MAPPING.items():
            for group in meta.model_groups:
                template_type = group.template or meta.template
                if template_type not in ('deepseek_v4', 'deepseek_v4_flash'):
                    continue
                source = _local_snapshot(group.models[0])
                if not source:
                    continue
                encoder_path = os.path.join(source, 'encoding', 'encoding_dsv4.py')
                if not os.path.exists(encoder_path):
                    continue
                spec = importlib.util.spec_from_file_location(f'encoding_dsv4_{tested}', encoder_path)
                encoder = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(encoder)
                tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
                try:
                    config = AutoConfig.from_pretrained(source, trust_remote_code=True)
                except Exception:
                    try:
                        config = PretrainedConfig.from_dict(
                            json.load(open(os.path.join(source, 'config.json'), encoding='utf-8')))
                    except Exception:
                        config = PretrainedConfig()
                for tools in (False, True):
                    official_messages = []
                    if tools:
                        official_messages.append({'role': 'system', 'content': 'sys', 'tools': [TOOL]})
                    official_messages += [
                        {
                            'role': 'user',
                            'content': 'old q'
                        },
                        {
                            'role': 'assistant',
                            'content': 'A1',
                            'reasoning_content': 'OLD_REASON'
                        },
                        {
                            'role': 'user',
                            'content': 'new q'
                        },
                        {
                            'role': 'assistant',
                            'content': 'A2',
                            'reasoning_content': 'CUR_REASON'
                        },
                    ]
                    swift_messages = [
                        {
                            'role': 'user',
                            'content': 'old q'
                        },
                        {
                            'role': 'assistant',
                            'content': '<think>\nOLD_REASON\n</think>\n\nA1'
                        },
                        {
                            'role': 'user',
                            'content': 'new q'
                        },
                        {
                            'role': 'assistant',
                            'content': '<think>\nCUR_REASON\n</think>\n\nA2'
                        },
                    ]
                    official = encoder.encode_messages(official_messages, thinking_mode='thinking', drop_thinking=True)
                    swift = _render(
                        template_type,
                        model_type,
                        source,
                        config,
                        tokenizer,
                        tokenizer,
                        swift_messages,
                        'infer_default',
                        tools=([TOOL] if tools else []))
                    for marker in ('OLD_REASON', 'CUR_REASON'):
                        self.assertEqual(marker in swift, marker in official,
                                         f'{template_type} tools={tools}\nswift={swift}\nofficial={official}')
                    tested += 1
        if not tested:
            self.skipTest('no local DeepSeek-V4 encoding implementation')


class TestThinkingDefaultParity(unittest.TestCase):
    """Compare resolved enable_thinking defaults with the local official template."""

    def test_default_enable_thinking_matches_official(self):
        seen = set()
        tested = 0
        for model_type, meta in MODEL_MAPPING.items():
            for group in meta.model_groups:
                template_type = group.template or meta.template
                if template_type in seen or template_type not in TEMPLATE_MAPPING:
                    continue
                if not TEMPLATE_MAPPING[template_type].is_thinking:
                    continue
                seen.add(template_type)
                try:
                    source, config, tokenizer, processor = _load(group.models[0])
                except unittest.SkipTest:
                    continue
                try:
                    template = _build(template_type, model_type, source, config, tokenizer, processor)
                    swift = template._get_enable_thinking()
                except Exception:
                    continue
                key = getattr(template, 'jinja_enable_thinking_key', 'enable_thinking')
                messages = [{'role': 'user', 'content': 'hi'}]
                try:
                    default = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    as_true = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, **{key: True})
                    as_false = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, **{key: False})
                except Exception:
                    continue
                if default == as_true and default != as_false:
                    expected = True
                elif default == as_false and default != as_true:
                    expected = False
                else:
                    continue
                self.assertEqual(swift, expected, f'{template_type}: swift={swift} official={expected}')
                tested += 1
        if not tested:
            self.skipTest('no local thinking template with distinguishable default')


if __name__ == '__main__':
    unittest.main()
