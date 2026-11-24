# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import glob
import hashlib
import itertools
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
from swift.template.template_meta import TemplateMeta

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


_RECORDED_SOURCES = set()


def _record_official_source(source, chat_template=None):
    if source in _RECORDED_SOURCES:
        return
    hashes = {}
    for name in ('tokenizer_config.json', 'chat_template.jinja', 'chat_template.json', 'encoding/encoding_dsv4.py',
                 'encoding/encoding_k3.py'):
        path = os.path.join(source, name)
        if os.path.isfile(path):
            with open(path, 'rb') as file:
                hashes[name] = hashlib.sha256(file.read()).hexdigest()
    if chat_template is not None:
        hashes['active_template'] = hashlib.sha256(
            json.dumps(chat_template, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    print('OFFICIAL_REFERENCE ' + json.dumps({'source': source, 'sha256': hashes}, sort_keys=True))
    _RECORDED_SOURCES.add(source)


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
    _record_official_source(source, tokenizer.chat_template)
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


def _render(template_type, model_type, source, config, tokenizer, processor, swift_messages, mode, tools=None):
    kwargs = {}
    if mode in ('explicit_false', 'explicit_true'):
        kwargs['preserve_thinking'] = mode == 'explicit_true'
    if mode == 'last_round':
        kwargs['loss_scale'] = 'last_round'
    template = _build(template_type, model_type, source, config, tokenizer, processor, **kwargs)
    template.set_mode('transformers' if mode == 'infer_default' else 'train')
    encoded = template.encode({'messages': copy.deepcopy(swift_messages), 'tools': [TOOL] if tools is None else tools})
    return tokenizer.decode(encoded['input_ids'], skip_special_tokens=False)


class TestThinkingPolicyUnits(unittest.TestCase):
    """Exercise registered template hooks without loading model weights."""

    def _template(self, name, is_training=True, **kwargs):
        meta = TEMPLATE_MAPPING[name]
        template = meta.template_cls(None, meta, **kwargs)
        template.set_mode('train' if is_training else 'transformers')
        return template

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

    def test_answer_prefix_model_rules(self):
        cases = {
            'qwen3_5': ([1, 3, 5], [5]),
            'qwen3': ([5], [5]),
            'deepseek_v3_1': ([1, 5], [5]),
            'gemma4': ([1, 5], [5]),
            'minicpm5_2b': ([1, 3, 5], [1, 3, 5]),
            'minicpmv4_6': ([5], [5]),
            'minicpmv4_7': ([1, 3, 5], [5]),
            'minicpm5': ([], []),
            'ernie_vl': ([], []),
            'ovis2_5': ([], []),
        }
        for name, expected in cases.items():
            for is_training, preserve in itertools.product((True, False), (None, True, False)):
                with self.subTest(name=name, training=is_training, preserve=preserve):
                    template = self._template(name, is_training, preserve_thinking=preserve)
                    template.template_meta.non_thinking_prefix = '<think></think>'
                    messages = self._messages()
                    template._add_non_thinking_prefix(SimpleNamespace(messages=messages))
                    prefixed = [
                        i for i, message in enumerate(messages)
                        if message['role'] == 'assistant' and message['content'].startswith('<think>')
                    ]
                    self.assertEqual(prefixed, expected[0 if is_training else 1])

    def test_unspecified_thinking_leaves_prefix_empty(self):
        for name in ('minicpm5', 'minicpm5_2b', 'minicpmv4_5'):
            template = self._template(name, False)
            self.assertIsNone(template._get_enable_thinking(), name)
            self.assertEqual(template._get_response_prefix(), '', name)
            for value, expected in ((True, template.template_meta.thinking_prefix),
                                    (False, template.template_meta.non_thinking_prefix)):
                inputs = SimpleNamespace(chat_template_kwargs={'enable_thinking': value})
                self.assertEqual(template._get_response_prefix(inputs), expected, name)

    def test_thinking_only_templates_ignore_explicit_false(self):
        always = ('deepseek_r1', 'glm4_z1_rumination', 'glm5_3', 'glm5_next', 'minimax_m2', 'minimax_m2_1',
                  'minimax_m2_5', 'minimax_m2_7', 'qwen3_thinking', 'qwq', 'step3_vl', 'ernie_thinking',
                  'ernie_vl_thinking', 'olmo3', 'ring2', 'ring2_5')
        for name in always:
            meta = copy.deepcopy(TEMPLATE_MAPPING[name])
            self.assertIs(meta.force_thinking, True, name)
            template = meta.template_cls(None, meta, enable_thinking=False)
            self.assertIs(template.enable_thinking, True, name)
            self.assertIs(template._get_enable_thinking(), True, name)
            self.assertIs(
                template._get_enable_thinking(SimpleNamespace(chat_template_kwargs={'enable_thinking': False})), True,
                name)
            self.assertEqual(
                template._get_response_prefix(SimpleNamespace(chat_template_kwargs={'enable_thinking': False})),
                meta.thinking_prefix, name)

    def test_non_chat_thinking_compatibility(self):
        combinations = itertools.product(('glm5_3', 'glm5_next', 'nemotron_h_omni', 'kimi_k3', 'xing4_0'),
                                         (None, True, False), (True, False))
        for name, enable, training in combinations:
            with self.subTest(template=name, enable=enable, training=training):
                meta = TEMPLATE_MAPPING[name]
                template = meta.template_cls(None, meta, use_chat_template=False, enable_thinking=enable)
                template.set_mode('train' if training else 'transformers')
                if name in ('glm5_3', 'glm5_next'):
                    expected_instance = False if enable is None else enable
                    expected_resolved = True
                else:
                    expected_instance = True if enable is None else enable
                    expected_resolved = expected_instance
                self.assertIs(template.enable_thinking, expected_instance)
                self.assertIs(template._get_enable_thinking(), expected_resolved)
                self.assertEqual(template._get_response_prefix(), '')
                self.assertIs(template._get_preserve_thinking(), True if name == 'kimi_k3' else
                              (not expected_resolved or training))

    def test_preserve_defaults_and_local_history_boundary(self):
        self.assertIs(self._template('glm5_3')._get_preserve_thinking(), True)
        self.assertIs(self._template('glm5_3', preserve_thinking=False)._get_preserve_thinking(), False)
        self.assertIs(self._template('qwq', preserve_thinking=False)._get_preserve_thinking(), False)
        self.assertIs(self._template('qwq')._get_preserve_thinking(), True)
        messages = self._messages()
        self.assertEqual(self._template('qwq', False)._get_current_round_start(messages), 4)
        self.assertEqual(self._template('glm4_z1_rumination', False)._get_current_round_start(messages), len(messages))
        self.assertEqual(self._template('glm4_z1_rumination')._get_current_round_start(messages), 4)

    def test_current_round_start_uses_user_and_followup(self):
        messages = self._messages()
        template = self._template('qwen3')
        self.assertEqual(template._get_current_round_start(messages), 4)
        messages[4] = {'role': 'tool', 'content': 't2'}
        self.assertEqual(template._get_current_round_start(messages), 0)

    def test_parameter_precedence(self):
        omitted = object()
        states = []
        for canonical, native in itertools.product((omitted, None, True, False), repeat=2):
            row = {}
            if canonical is not omitted:
                row['enable_thinking'] = canonical
            if native is not omitted:
                row['thinking'] = native
            states.append(row)
        combinations = itertools.product(states, states, (None, True, False), (None, True, False))
        for request, defaults, instance, model_default in combinations:
            meta = TemplateMeta(
                'unit',
                prefix=[],
                prompt=['{{QUERY}}'],
                chat_sep=None,
                is_thinking=True,
                default_enable_thinking=model_default)
            template = Template(None, meta, enable_thinking=instance)
            template.jinja_enable_thinking_key = 'thinking'
            template.chat_template_kwargs = defaults.copy()
            inputs = SimpleNamespace(
                chat_template_kwargs=request.copy(), messages=[{
                    'role': 'user',
                    'content': 'q'
                }], tools=[], system=None)
            candidates = [
                request.get('enable_thinking'),
                request.get('thinking'),
                defaults.get('enable_thinking'),
                defaults.get('thinking'), instance, model_default
            ]
            expected = next((value for value in candidates if value is not None), None)
            self.assertIs(template._get_enable_thinking(inputs), expected)
            seen = {}

            def apply_chat_template(messages, **kwargs):
                seen.update(kwargs)
                return ''

            template.processor = SimpleNamespace(tokenizer=SimpleNamespace(apply_chat_template=apply_chat_template))
            template._jinja_encode(inputs)
            self.assertIs(seen.get('thinking'), expected)

    def test_template_preserve_priority(self):
        combinations = itertools.product(('hy_v3', 'kimi_k3', 'qwen3'), (None, True, False), (None, True, False),
                                         (None, True, False), ('default', 'all', 'last_round', 'infer'), (False, True))
        for name, request, defaults, instance, mode, tools in combinations:
            template = self._template(
                name, mode != 'infer', preserve_thinking=instance, loss_scale='default' if mode == 'infer' else mode)
            template.chat_template_kwargs = {'preserve_thinking': defaults}
            inputs = SimpleNamespace(chat_template_kwargs={'preserve_thinking': request}, tools=[TOOL] if tools else [])
            factory_default = name == 'kimi_k3' or (name == 'hy_v3' and tools) or mode in ('default', 'all')
            values = [request, defaults, instance, factory_default]
            expected = next(value for value in values if value is not None)
            self.assertIs(template._get_preserve_thinking(inputs), expected, (name, mode, tools))

    def test_history_whitespace_preserved(self):
        for name in ('deepseek_v4', 'deepseek_v41', 'deepseek_v4_flash', 'nemotron_h', 'nemo_speechlm',
                     'nemotron_h_omni'):
            template = self._template(name)
            prefix = template.template_meta.history_thinking_prefix
            for content, tail in (('<think>r</think>\n  answer \t', '\n  answer \t'), (' plain \n', 'plain'),
                                  ('<think>r</think>a</think>\n', '\n')):
                self.assertEqual(template._remove_thinking_content(content), prefix + tail, name)

    def test_explicit_registered_defaults(self):
        true_names = set('deepseek_r1 deepseek_v4 deepseek_v41 deepseek_v4_flash ernie_thinking ernie_vl '
                         'ernie_vl_thinking glm4_5 glm4_5v glm4_7 glm4_z1_rumination glm5_1 glm5_2 glm5_3 glm5_next '
                         'hunyuan interns1 kimi_k25 kimi_k3 minicpmo4_5 minimax_m2 minimax_m2_1 minimax_m2_5 '
                         'minimax_m2_7 nemo_speechlm nemotron_h nemotron_h_omni olmo3 ovis2_5 qwen3 qwen3_5 '
                         'qwen3_8 qwen3_thinking qwq ring2 ring2_5 spark2_5 step3_vl xing4_0 youtu_llm '
                         'lfm3 lfm3_vl'.split())
        unspecified = {'minicpm5', 'minicpm5_2b', 'minicpmv4_5'}
        self.assertTrue((true_names | unspecified).issubset(TEMPLATE_MAPPING))
        preserve_true = {'olmo3', 'glm5_3', 'glm5_next', 'qwen3_8', 'minicpm5_2b', 'ling3', 'spark2_5', 'minimax_m3_vl'}
        preserve_false = {'xing4_0', 'nemotron_h', 'nemo_speechlm', 'nemotron_h_omni'}
        for name, meta in TEMPLATE_MAPPING.items():
            expected = None if name in unspecified else name in true_names
            self.assertIs(meta.default_enable_thinking, expected, name)
            if meta.force_thinking:
                self.assertIs(meta.default_enable_thinking, True, name)
            expected_preserve = True if name in preserve_true else False if name in preserve_false else None
            self.assertIs(meta.preserve_thinking, expected_preserve, name)
            if name == 'sail_vl2':
                # Covered by the processor-backed constructor test.
                continue
            for mode in ('default', 'all', 'last_round', 'infer'):
                template = self._template(name, mode != 'infer', loss_scale='default' if mode == 'infer' else mode)
                if expected_preserve is not None:
                    resolved = expected_preserve
                elif name == 'kimi_k3':
                    resolved = True
                elif not meta.is_thinking and not expected:
                    resolved = True
                else:
                    resolved = mode in ('default', 'all') and name != 'diffusion_gemma'
                self.assertIs(template._get_preserve_thinking(), resolved, (name, mode))


class TestThinkingRegressions(unittest.TestCase):
    """Targeted regressions; model loading is local-only."""

    @staticmethod
    def _byte_processor():
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            eos_token='<|im_end|>',
            pad_token='<|endoftext|>',
            additional_special_tokens=['<|im_start|>'])
        return SimpleNamespace(
            tokenizer=tokenizer,
            num_image_token=1,
            model_info=SimpleNamespace(config={}, task_type='causal_lm', max_model_len=8192),
            model_meta=SimpleNamespace(is_multimodal=False))

    def _assert_fragments(self, tokenizer, encoded, fragments, training=True):
        expected_ids = tokenizer.encode(''.join(text for text, _ in fragments), add_special_tokens=False)
        self.assertEqual(encoded['input_ids'], expected_ids)
        if training:
            expected_labels = []
            for text, supervised in fragments:
                ids = tokenizer.encode(text, add_special_tokens=False)
                expected_labels.extend(ids if supervised else [-100] * len(ids))
            self.assertEqual(encoded['labels'], expected_labels)

    def test_registered_constructor_requires_processor(self):
        processor = self._byte_processor()
        processor.model_meta.is_multimodal = True
        for chat, enable in itertools.product((True, False), (None, True, False)):
            template = get_template(processor, template_type='sail_vl2', use_chat_template=chat, enable_thinking=enable)
            expected = False if enable is None else enable
            self.assertIs(template.enable_thinking, expected)
            self.assertIs(template._get_enable_thinking(), expected)
            self.assertEqual(template._get_response_prefix(), '')
            self.assertIs(template._get_preserve_thinking(), not expected)

    def test_merged_answer_fragments_and_thinking_edges(self):
        processor = self._byte_processor()
        empty = '<think>\n\n</think>\n\n'
        cases = (
            ('', empty, ''),
            ('<think>', '<think>', '<think>'),
            ('<think>unfinished', '<think>unfinished', '<think>unfinished'),
            ('<think></think>', '<think></think>', ''),
            (' plain \n', empty + ' plain \n', 'plain'),
            ('<think>r</think>a</think> tail ', '<think>r</think>a</think> tail ', 'tail'),
        )
        for preserve, (content, retained, stripped) in itertools.product((True, False), cases):
            with self.subTest(preserve=preserve, content=content):
                template = get_template(processor, template_type='minicpm5_2b', preserve_thinking=preserve)
                template.set_mode('train')
                messages = [{
                    'role': 'user',
                    'content': 'q1'
                }, {
                    'role': 'assistant',
                    'content': content,
                    'loss': False
                }, {
                    'role': 'assistant',
                    'content': 'X',
                    'loss': True
                }, {
                    'role': 'user',
                    'content': 'q2'
                }, {
                    'role': 'assistant',
                    'content': content,
                    'loss': False
                }, {
                    'role': 'assistant',
                    'content': 'Y',
                    'loss': True
                }]
                original = copy.deepcopy(messages)
                encoded = template.encode({'messages': messages})
                fragments = [('<|im_start|>user\nq1<|im_end|>\n<|im_start|>assistant\n', False),
                             (retained if preserve else stripped, False), ('X<|im_end|>\n', True),
                             ('<|im_start|>user\nq2<|im_end|>\n<|im_start|>assistant\n', False), (retained, False),
                             ('Y<|im_end|>\n', True)]
                self._assert_fragments(processor.tokenizer, encoded, fragments)
                self.assertEqual(messages, original)

    def test_gemma_answer_boundaries_preserve_encoding(self):
        model_type, source, config, tokenizer, processor = self._load_template('gemma4')
        if not callable(processor) or not hasattr(processor, 'tokenizer'):
            self.skipTest('Gemma4 processor unavailable')
        prepared = _build('gemma4', model_type, source, config, tokenizer, processor)
        processor.model_info = prepared.model_info
        processor.model_meta = prepared.model_meta
        empty = '<|channel>thought\n<channel|>'
        for mode in ('default', 'last_round', 'infer'):
            with self.subTest(mode=mode):
                template = get_template(
                    processor, template_type='gemma4', loss_scale='default' if mode == 'infer' else mode)
                template.set_mode('transformers' if mode == 'infer' else 'train')
                messages = [{
                    'role': 'user',
                    'content': 'q1'
                }, {
                    'role': 'assistant',
                    'content': 'a1'
                }, {
                    'role': 'user',
                    'content': 'q2'
                }, {
                    'role': 'assistant',
                    'content': 'a2'
                }]
                encoded = template.encode({'messages': messages, 'add_eos': True})
                old = empty + 'a1' if mode == 'default' else 'a1'
                fragments = [('<bos><|turn>user\nq1<turn|>\n<|turn>model\n', False),
                             (old + '<turn|>\n', mode == 'default'), ('<|turn>user\nq2<turn|>\n<|turn>model\n', False),
                             (empty + 'a2<turn|>\n', True)]
                self._assert_fragments(tokenizer, encoded, fragments, training=mode != 'infer')

        for mode in ('default', 'last_round', 'infer'):
            with self.subTest(mode=mode, tool_chain=True):
                template = get_template(
                    processor, template_type='gemma4', loss_scale='default' if mode == 'infer' else mode)
                template.set_mode('transformers' if mode == 'infer' else 'train')
                messages = [{
                    'role': 'user',
                    'content': 'q1'
                }, {
                    'role': 'assistant',
                    'content': 'a1'
                }, {
                    'role': 'tool_call',
                    'content': {
                        'name': 'get_weather',
                        'arguments': {
                            'city': 'Paris'
                        }
                    }
                }, {
                    'role': 'tool',
                    'name': 'get_weather',
                    'content': 'result'
                }, {
                    'role': 'assistant',
                    'content': 'a2'
                }]
                encoded = template.encode({'messages': messages, 'add_eos': True})
                call = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
                result = '<|tool_response>response:get_weather{value:<|"|>result<|"|>}<tool_response|>'
                fragments = [('<bos><|turn>user\nq1<turn|>\n<|turn>model\n', False),
                             (empty + 'a1' + call, mode == 'default'), (result, False), ('a2<turn|>\n', True)]
                self._assert_fragments(tokenizer, encoded, fragments, training=mode != 'infer')

    def test_qwq_last_round_keeps_current_thinking(self):
        model_type, source, config, tokenizer, processor = self._load_template('qwq')
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
        text = _render('qwq', model_type, source, config, tokenizer, processor, messages, 'last_round')
        self.assertIn('CUR', text)
        self.assertIn('<think', text)

    def test_deepseek_plain_history_and_ending_assistant(self):
        model_type, source, config, tokenizer, processor = self._load_template('deepseek_v3_1')
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
            swift = _render('deepseek_v3_1', model_type, source, config, tokenizer, processor, messages, mode)
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
                _record_official_source(source)
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
                            'content': '\n\nA1',
                            'reasoning_content': '\nOLD_REASON\n'
                        },
                        {
                            'role': 'user',
                            'content': 'new q'
                        },
                        {
                            'role': 'assistant',
                            'content': '\n\nA2',
                            'reasoning_content': '\nCUR_REASON\n'
                        },
                    ]
                    swift_messages = []
                    for message in official_messages:
                        if message['role'] == 'system':
                            continue
                        content = message['content']
                        if message['role'] == 'assistant':
                            content = '<think>' + message['reasoning_content'] + '</think>' + content
                        swift_messages.append({'role': message['role'], 'content': content})
                    if tools:
                        swift_messages.insert(0, {'role': 'system', 'content': 'sys'})
                    for training in (False, True):
                        with self.subTest(template=template_type, tools=tools, training=training):
                            # Continuing an existing answer corresponds to wo_eos=True.
                            official_messages[-1]['wo_eos'] = not training
                            official = encoder.encode_messages(
                                official_messages, thinking_mode='thinking', drop_thinking=True)
                            expected_ids = tokenizer.encode(official, add_special_tokens=False)
                            template = _build(
                                template_type,
                                model_type,
                                source,
                                config,
                                tokenizer,
                                tokenizer,
                                loss_scale='last_round')
                            template.set_mode('train' if training else 'transformers')
                            encoded = template.encode({
                                'messages': copy.deepcopy(swift_messages),
                                'tools': [TOOL] if tools else []
                            })
                            self.assertEqual(encoded['input_ids'], expected_ids)
                            if training:
                                assistant = tokenizer.convert_tokens_to_ids('<｜Assistant｜>')
                                start = max(i for i, token in enumerate(expected_ids) if token == assistant) + 1
                                self.assertEqual(encoded['labels'], [-100] * start + expected_ids[start:])
                            tested += 1
        if not tested:
            self.skipTest('no local DeepSeek-V4 encoding implementation')

    @staticmethod
    def _group_of(template_type):
        for model_type, meta in MODEL_MAPPING.items():
            for group in meta.model_groups:
                if (group.template or meta.template) == template_type:
                    return model_type, group
        return None, None

    def _load_template(self, template_type):
        model_type, group = self._group_of(template_type)
        if group is None:
            self.skipTest(f'{template_type} group not found')
        return model_type, *_load(group.models[0])

    def test_kimi_tool_boundary_matches_official(self):
        """Kimi splits history at the last assistant without tool calls; verify the edge shapes."""
        model_type, source, config, tokenizer, processor = self._load_template('kimi_k25')

        def u(text):
            return {'role': 'user', 'content': text}

        arguments = {'city': 'Paris'}

        def swift_asst(marker, tool=False):
            return {'role': 'assistant', 'content': f'<think>{marker}</think>ans'}

        def official_asst(marker, tool=False):
            message = {'role': 'assistant', 'content': 'ans', 'reasoning_content': marker}
            if tool:
                message['tool_calls'] = [{
                    'id': 'functions.get_weather:0',
                    'type': 'function',
                    'function': {
                        'name': TOOL['function']['name'],
                        'arguments': json.dumps(arguments)
                    }
                }]
            return message

        tool_call = {'role': 'tool_call', 'content': {'name': TOOL['function']['name'], 'arguments': arguments}}
        swift_tool = {'role': 'tool', 'content': 'result', 'tool_call_id': 'functions.get_weather:0'}
        official_tool = {'role': 'tool', 'content': 'result', 'tool_call_id': 'functions.get_weather:0'}
        cases = {
            'trailing user': ([u('q1'), swift_asst('M1'), u('q2')], [u('q1'), official_asst('M1'),
                                                                     u('q2')]),
            'ends with answer': ([u('q1'), swift_asst('M1'), u('q2'),
                                  swift_asst('M2')], [u('q1'),
                                                      official_asst('M1'),
                                                      u('q2'),
                                                      official_asst('M2')]),
            'pending tool call': ([u('q1'), swift_asst('M1'), tool_call,
                                   swift_tool], [u('q1'), official_asst('M1', True), official_tool]),
            'answer after tool':
            ([u('q1'), swift_asst('M1'), tool_call, swift_tool,
              swift_asst('M2')], [u('q1'), official_asst('M1', True), official_tool,
                                  official_asst('M2')]),
            'follow-up after tool': ([u('q1'), swift_asst('M1'), tool_call, swift_tool,
                                      u('q2')], [u('q1'), official_asst('M1', True), official_tool,
                                                 u('q2')]),
            'dangling tool call': ([u('q1'), swift_asst('M1'), tool_call], [u('q1'), official_asst('M1', True)]),
        }
        template = _build(
            'kimi_k25',
            model_type,
            source,
            config,
            tokenizer,
            processor,
            preserve_thinking=False,
            loss_scale='last_round')
        template.set_mode('train')
        raw = '<think>M1</think>ans' + template.agent_template._format_tool_calls([tool_call])
        cases['raw pending call'] = ([u('q1'), {
            'role': 'assistant',
            'content': raw
        }], [u('q1'), official_asst('M1', True)])
        quoted = 'Mention <|tool_calls_section_begin|> as text.'
        cases['quoted marker'] = ([u('q1'), {
            'role': 'assistant',
            'content': '<think>M1</think>' + quoted
        }, u('q2')], [u('q1'), {
            'role': 'assistant',
            'content': quoted,
            'reasoning_content': 'M1'
        }, u('q2')])
        second_call = {'role': 'tool_call', 'content': {'name': 'get_weather', 'arguments': {'city': 'Berlin'}}}
        second_tool = {'role': 'tool', 'content': 'second result', 'tool_call_id': 'functions.get_weather:1'}
        official_multi = official_asst('M1', True)
        official_multi['tool_calls'].append({
            'id': 'functions.get_weather:1',
            'type': 'function',
            'function': {
                'name': 'get_weather',
                'arguments': '{"city": "Berlin"}'
            }
        })
        cases['multiple calls and results'] = ([
            u('q1'), swift_asst('M1'), tool_call, second_call, swift_tool, second_tool
        ], [u('q1'), official_multi, official_tool, second_tool])
        cases['independent call supervision'] = ([
            u('q1'), {
                **swift_asst('M1'), 'loss': False
            }, {
                **tool_call, 'loss': True
            }
        ], [u('q1'), official_asst('M1', True)])
        for name, (swift_messages, official_messages) in cases.items():
            with self.subTest(shape=name):
                encoded = template.encode({'messages': copy.deepcopy(swift_messages), 'tools': [TOOL]})
                official = tokenizer.apply_chat_template(
                    official_messages,
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=official_messages[-1]['role'] != 'assistant',
                    thinking=True,
                    tools=[TOOL])
                self.assertEqual(encoded['input_ids'], official)
                if name == 'independent call supervision':
                    start = official.index(tokenizer.convert_tokens_to_ids('<|tool_calls_section_begin|>'))
                    self.assertEqual(encoded['labels'], [-100] * start + official[start:])

    def test_hy_v3_template_preserve_false_with_tools(self):
        model_type, source, config, tokenizer, processor = self._load_template('hy_v3')
        template = _build('hy_v3', model_type, source, config, tokenizer, processor)
        template.chat_template_kwargs = {'preserve_thinking': False}
        inputs = SimpleNamespace(chat_template_kwargs={}, tools=[TOOL])
        self.assertFalse(template._get_preserve_thinking(inputs))

    def test_minicpm_answer_block_matches_official(self):
        """Compare answer formatting tokens and independently derived supervision labels."""
        cases = ('minicpm5', 'minicpm5_2b', 'minicpmv4_5', 'minicpmv4_6')
        checked = 0
        skips = {}
        for template_type in cases:
            try:
                model_type, source, config, tokenizer, processor = self._load_template(template_type)
            except unittest.SkipTest as e:
                skips[template_type] = str(e)
                continue
            header = tokenizer.encode('<|im_start|>assistant\n', add_special_tokens=False)
            end = tokenizer.convert_tokens_to_ids('<|im_end|>')
            suffix_len = len(tokenizer.encode('<|im_end|>\n', add_special_tokens=False))
            for enable, mode, turns in itertools.product((None, True, False), ('default', 'all', 'last_round', 'infer'),
                                                         (1, 2)):
                with self.subTest(template=template_type, enable=enable, mode=mode, turns=turns):
                    template = _build(
                        template_type,
                        model_type,
                        source,
                        config,
                        tokenizer,
                        processor,
                        loss_scale='default' if mode == 'infer' else mode)
                    template.set_mode('transformers' if mode == 'infer' else 'train')
                    messages = []
                    for turn in range(turns):
                        messages.extend([{
                            'role': 'user',
                            'content': f'q{turn}'
                        }, {
                            'role': 'assistant',
                            'content': f'a{turn}'
                        }])
                    if mode == 'infer':
                        messages.pop()
                    kwargs = {} if enable is None else {'enable_thinking': enable}
                    encoded = template.encode({'messages': copy.deepcopy(messages), 'chat_template_kwargs': kwargs})
                    official = tokenizer.apply_chat_template(
                        messages, tokenize=True, return_dict=False, add_generation_prompt=mode == 'infer', **kwargs)
                    self.assertEqual(encoded['input_ids'], official)
                    if mode != 'infer':
                        expected = [-100] * len(official)
                        starts = [
                            i + len(header) for i in range(len(official) - len(header) + 1)
                            if official[i:i + len(header)] == header
                        ]
                        self.assertEqual(len(starts), turns)
                        for turn, start in enumerate(starts):
                            if mode == 'last_round' and turn != turns - 1:
                                continue
                            stop = len(official) if turn == turns - 1 else official.index(end, start) + suffix_len
                            expected[start:stop] = official[start:stop]
                        if mode == 'all':
                            expected = official.copy()
                        expected[0] = -100
                        self.assertEqual(encoded['labels'], expected)
                    checked += 1
        if not checked:
            self.skipTest(f'no MiniCPM tokenizer available: {skips}')
        self.assertFalse(skips, f'MiniCPM representatives were not checked: {skips}')


class TestThinkingDefaultParity(unittest.TestCase):
    """Compare resolved enable_thinking defaults with the local official template."""

    # One representative per official default-semantics group. Whenever local snapshots are
    # available these must be verified, otherwise a broken template hides behind a silent skip.
    REQUIRED = ('qwen3', 'qwen3_5', 'glm4_5', 'glm4_7', 'hunyuan', 'kimi_k25', 'deepseek_v3_1', 'gemma4', 'minicpmv4_6',
                'ernie_vl', 'spark2_5')

    def test_default_enable_thinking_matches_official(self):
        seen = set()
        verified = {}
        skips = {}
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
                except unittest.SkipTest as e:
                    skips[template_type] = f'load: {str(e)[:80]}'
                    continue
                key = 'enable_thinking'
                try:
                    template = _build(template_type, model_type, source, config, tokenizer, processor)
                    swift = template._get_enable_thinking()
                    key = getattr(template, 'jinja_enable_thinking_key', 'enable_thinking')
                    messages = [{'role': 'user', 'content': 'hi'}]
                    default = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    as_true = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, **{key: True})
                    as_false = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, **{key: False})
                except Exception as e:
                    skips[template_type] = f'{type(e).__name__}: {str(e)[:80]}'
                    continue
                if default == as_true and default != as_false:
                    expected = True
                elif default == as_false and default != as_true:
                    expected = False
                else:
                    skips[template_type] = f'official switch not distinguishable (key={key})'
                    continue
                self.assertEqual(swift, expected, f'{template_type}: swift={swift} official={expected}')
                verified[template_type] = swift
        if not verified:
            self.skipTest(f'no local thinking template could be verified; skips={skips}')
        missing = {name: skips.get(name, 'not visited') for name in self.REQUIRED if name not in verified}
        self.assertFalse(missing, f'representative templates were not verified: {missing}')
        print(f'\nverified {len(verified)} defaults, skipped {len(skips)}: {skips}')


if __name__ == '__main__':
    unittest.main()
