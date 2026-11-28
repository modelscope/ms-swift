# Copyright (c) ModelScope Contributors. All rights reserved.
"""Opt-in tests: run manually on the validation host, never during local development."""
import json
import os
import pytest
import shutil
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

if os.environ.get('SWIFT_TEST_D3') != '1':
    pytest.skip('Set SWIFT_TEST_D3=1 on the validation host to run D3 tests.', allow_module_level=True)

from swift.dataset import load_dataset
from swift.model.decision_head import D3ReadoutHead, load_d3_config, validate_d3_config


@pytest.mark.parametrize('installed,allowed', [
    ('5.13.0', False),
    ('5.13.1', True),
    ('5.14.0', True),
    ('5.17.0', True),
    ('5.18.0', True),
    ('6.0.0', True),
])
def test_transformers_version_guard(monkeypatch, installed, allowed):
    from packaging.requirements import Requirement
    from transformers import Qwen3_5Config  # noqa: F401
    from transformers.utils import versions
    from types import SimpleNamespace

    from swift.model import MODEL_MAPPING
    from swift.model.models.decision import D3ModelLoader

    registered_requirement = Requirement(MODEL_MAPPING['d3'].requires[0])
    assert registered_requirement.name == 'transformers'
    assert (installed in registered_requirement.specifier) == allowed

    original_version = versions.importlib.metadata.version

    def package_version(name):
        return installed if name == 'transformers' else original_version(name)

    monkeypatch.setattr(versions.importlib.metadata, 'version', package_version)
    loader = object.__new__(D3ModelLoader)
    # Stop at task validation: exercise the actual dependency gate without opening a model directory.
    loader.model_info = SimpleNamespace(task_type='causal_lm')
    error = ValueError if allowed else ImportError
    match = 'task_type=decision' if allowed else 'transformers>=5.13.1'
    with pytest.raises(error, match=match):
        loader.get_config('unused')


@pytest.fixture(scope='module')
def release():
    path = Path(os.environ['D3_MODEL_DIR'])
    assert path.is_dir(), 'D3_MODEL_DIR must be an existing local release; tests never download weights.'
    return path


@pytest.fixture(scope='module')
def device():
    value = os.environ.get('D3_TEST_DEVICE', 'npu:0')
    if value.startswith('npu'):
        import torch_npu  # noqa: F401
    return value


@contextmanager
def reference_attention_masks(encoder, device_type):
    """Materialize the reference's broadcast key mask for Ascend SDPA only."""
    if device_type != 'npu':
        yield
        return

    def expand_mask(module, args, kwargs):
        # PromptEncoder calls each decoder layer directly, bypassing the text-model hook.
        # Repeat the same key mask for every query; keep its polarity and values intact.
        mask = kwargs['attention_mask']
        kwargs['attention_mask'] = mask.expand(-1, -1, args[0].shape[1], -1).contiguous()
        return args, kwargs

    handles = []
    try:
        for layer, full_attention in encoder.stack:
            if full_attention:
                handles.append(layer.register_forward_pre_hook(expand_mask, with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


@pytest.fixture(scope='module')
def tiny_dir(release, tmp_path_factory):
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3_5Config, Qwen3_5Model
    out = tmp_path_factory.mktemp('tiny_d3')
    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja',
                 'vocab.json', 'merges.txt', 'added_tokens.json', 'decision_config.json'):
        if (release / name).is_file():
            shutil.copyfile(release / name, out / name)
    base = json.loads((release / 'config.json').read_text())
    base.pop('auto_map', None)
    text = dict(
        base['text_config'],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        mtp_num_hidden_layers=0,
        layer_types=['linear_attention', 'full_attention', 'linear_attention', 'full_attention'])
    text['rope_parameters'] = dict(text['rope_parameters'], mrope_section=[3, 3, 2])
    vision = dict(base['vision_config'], depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
    torch.manual_seed(17)
    model = Qwen3_5Model(Qwen3_5Config(**dict(base, text_config=text, vision_config=vision)))
    model.save_pretrained(str(out))
    save_file({'weight': torch.randn(255, 64, dtype=torch.float32) * .05}, str(out / 'readout.safetensors'))
    return str(out)


def load_tiny(path, device, dtype='float32'):
    import torch

    from swift.model import get_model_processor
    from swift.template import get_template
    model, processor = get_model_processor(
        path, model_type='d3', torch_dtype=getattr(torch, dtype), device_map=device, attn_impl='sdpa')
    template = get_template(processor, template_type='d3', padding_side='left', max_length=1024)
    return model, template


def tiny_rows():
    return [
        {
            'messages': [{
                'role': 'user',
                'content': 'A short example.'
            }],
            'questions': ['Choose the best matching option.'],
            'kinds': ['choice'],
            'options': [['b', 'a: ']],
            'target_probs': [[.2, .8]]
        },
        {
            'messages': [{
                'role': 'user',
                'content': 'A short example.'
            }],
            'questions': ['Is this a substantially longer second question?'],
            'kinds': ['noul'],
            'options': [['No / false', 'Yes / true']],
            'target_probs': [[1., 0.]]
        },
        {
            'messages': [{
                'role': 'user',
                'content': 'A short example.'
            }],
            'questions': ['Choose the best matching option.'],
            'kinds': ['score'],
            'options': [['low', 'medium', 'high']],
            'target_probs': [[0., 0., 1.]]
        },
    ]


def move_batch(batch, device):
    import torch
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
def test_generation_wrapper_preserves_release_weights(tiny_dir, device, dtype, monkeypatch):
    import torch
    from safetensors import safe_open
    from transformers import Qwen3_5ForConditionalGeneration

    config_path = Path(tiny_dir) / 'config.json'
    release_config = config_path.read_bytes()
    model, template = load_tiny(tiny_dir, device, dtype)
    assert isinstance(model, Qwen3_5ForConditionalGeneration)
    assert model.lm_head.weight is model.get_input_embeddings().weight
    assert model.lm_head.weight.dtype == getattr(torch, dtype)
    # Adding the generation wrapper must load every original backbone tensor, including embeddings.
    backbone = model.model.state_dict()
    with safe_open(str(Path(tiny_dir) / 'model.safetensors'), framework='pt', device='cpu') as handle:
        assert set(backbone) == set(handle.keys())
        for key in handle.keys():
            torch.testing.assert_close(backbone[key], handle.get_tensor(key).to(backbone[key]), rtol=0, atol=0)
    assert config_path.read_bytes() == release_config

    def reject_vocab_projection(*args, **kwargs):
        raise AssertionError('D3 decision forward must bypass lm_head.')

    monkeypatch.setattr(model.lm_head, 'forward', reject_vocab_projection)
    model.eval()
    batch = move_batch(template.data_collator([template.encode(row) for row in tiny_rows()]), device)
    with torch.no_grad():
        output = model(**batch)
    assert output.logits.dtype == torch.float32
    assert torch.isfinite(output.logits[output.option_mask]).all()


@pytest.mark.parametrize('streaming', [False, True])
def test_standard_dataset_loading(tmp_path, streaming):
    rows = tiny_rows()
    rows[0]['messages'][0]['content'] = json.dumps({'消息': '你好', 'image': 'business field'}, ensure_ascii=False)
    rows[1]['messages'][0]['content'] = ''
    data = tmp_path / 'train.jsonl'
    data.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    loaded, validation = load_dataset(str(data), streaming=streaming, strict=True, load_from_cache_file=False)
    assert validation is None
    # Generic loading must preserve Unicode/JSON text, option order and ragged target distributions.
    keys = tuple(rows[0])
    assert [{key: row[key] for key in keys} for row in loaded] == rows
    train, validation = load_dataset(
        str(data), split_dataset_ratio=.34, seed=17, strict=True, load_from_cache_file=False)
    assert len(train) + len(validation) == len(rows)

    def identity(row):
        return json.dumps({key: row[key] for key in keys}, sort_keys=True, ensure_ascii=False)

    assert {identity(row) for row in train}.isdisjoint(identity(row) for row in validation)
    assert {identity(row) for row in train} | {identity(row) for row in validation} == {identity(row) for row in rows}


@pytest.mark.parametrize('streaming', [False, True])
def test_d3_registered_dataset(tmp_path, monkeypatch, streaming):
    import runpy
    from unittest.mock import patch

    from swift.dataset import DATASET_MAPPING, dataset_syntax

    example = Path(__file__).resolve().parents[1] / 'examples/train/decision'
    data = tmp_path / 'custom_d3.jsonl'
    data.write_bytes((example / 'data/d3flash.jsonl').read_bytes())
    monkeypatch.setenv('D3_TRAIN_FILE', str(data))
    monkeypatch.delenv('D3_VAL_FILE', raising=False)
    monkeypatch.setattr(dataset_syntax, '_dataset_meta_mapping', None)

    def load(name):
        return list(load_dataset(name, strict=True, streaming=streaming, load_from_cache_file=False)[0])

    with patch.dict(DATASET_MAPPING):
        runpy.run_path(str(example / 'dataset.py'))
        assert Path(DATASET_MAPPING['d3_decision'].dataset_path) == data
        choice, noul, score, received, soft_score = load('d3_decision')
        arrived, status = load('d3_decision_val')
        other_dir = tmp_path / 'other'
        other_dir.mkdir()
        for filename in ('train.jsonl', 'val.jsonl'):
            other = other_dir / filename
            other.touch()
            meta = dataset_syntax.DatasetSyntax.parse(str(other)).get_dataset_meta(use_hf=False)
            assert meta.dataset_name not in ('d3_decision', 'd3_decision_val')
    assert choice['options'] == [['paid: 已付款', 'delivered: 已送达']]
    assert choice['target_probs'] == [[1., 0.]]
    assert noul['options'] == [['No / false', 'Yes / true']] and noul['target_probs'] == [[1., 0.]]
    assert score['kinds'] == ['score'] and score['target_probs'] == [[0., 1., 0.]]
    assert received['messages'] == [{'role': 'user', 'content': '{"status": "delivered", "note": "客户已经签收"}'}]
    assert received['target_probs'] == [[0., 1.]]
    assert soft_score['questions'] == ['Choose the best matching option.']
    assert soft_score['options'] == [['未付款', '运输中', '已签收']] and soft_score['target_probs'] == [[0., 0.1, 0.9]]
    assert arrived['options'] == [['No / false', 'Yes / true']] and arrived['target_probs'] == [[0., 1.]]
    assert status['options'] == [['pending', 'delivered: Arrived']] and status['target_probs'] == [[0., 1.]]


def test_d3_preprocessor_native_and_standard_rows():
    from swift.dataset.preprocessor import D3Preprocessor
    preprocessor = D3Preprocessor()
    choice, noul = preprocessor.preprocess({
        'state': {
            '备注': '好'
        },
        'questions': {
            'status': {
                'type': 'choice',
                'criteria': {
                    '1': None,
                    '0': ''
                }
            },
            'missing': None,  # Missing object field padded by Arrow.
            'accepted': {
                'type': 'noul',
                'criteria': {
                    'true': '是'
                }
            },
        },
        'answers': {
            'status': '0',
            'accepted': False
        },
    })
    assert choice['messages'] == [{'role': 'user', 'content': '{"备注": "好"}'}]
    assert choice['options'] == [['1', '0: ']] and choice['target_probs'] == [[0., 1.]]
    assert noul['options'] == [['No / false', '是']] and noul['target_probs'] == [[1., 0.]]
    standard = tiny_rows()[0]
    assert preprocessor.preprocess(standard) == [standard]


def test_head_autocast_and_calibration(release, device, tmp_path):
    import torch
    head = D3ReadoutHead(8, load_d3_config(str(release))).to(device=device)
    head.head_temperature.fill_(1.5)
    head.temperatures.copy_(head.temperatures.new_tensor([.8, 1.2, 1.4]))
    hidden = torch.randn(2, 6, 8, device=device, requires_grad=True)
    meta = dict(left_padding=True, seq_lens=[4, 6], record_index=[0, 1], counts=[1, 1], n_options=[2, 3], kinds=[0, 2])
    expected = hidden[:, -1].float() @ head.readout.weight.T
    with torch.autocast(device_type=hidden.device.type, dtype=torch.bfloat16):
        output = head(hidden, None, meta)
    assert output.logits.dtype == torch.float32
    torch.testing.assert_close(output.logits[0, :2], expected[0, :2] / 1.5 / .8, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(output.logits[1], expected[1, :3] / 1.5 / 1.4, rtol=1e-5, atol=1e-6)
    output.logits[output.option_mask].sum().backward()
    assert hidden.grad[:, -1].abs().sum() > 0
    assert hidden.grad[:, :-1].count_nonzero() == 0
    assert head.readout.weight.grad.abs().sum() > 0
    head.save_pretrained(str(tmp_path))
    restored = D3ReadoutHead(8, load_d3_config(str(release))).to(device=device)
    assert restored.load_pretrained(str(tmp_path))
    assert head.extra_state_for_save() == restored.extra_state_for_save()
    torch.testing.assert_close(head.readout.weight, restored.readout.weight, rtol=0, atol=0)


@pytest.mark.parametrize('key,value', [
    ('format_version', 2),
    ('format_id', 'unknown'),
    ('attention_mode', 'causal'),
    ('pooling', 'mean'),
    ('temperature', 0),
    ('temperature', float('nan')),
    ('codes', ['A'] * 255),
    ('token_ids', [0] * 255),
    ('readout_dtype', 'bfloat16'),
])
def test_reject_release_config(release, key, value):
    config = load_d3_config(str(release))
    config[key] = value
    with pytest.raises(ValueError):
        validate_d3_config(config)


def test_template_batch_workers_and_padding(tiny_dir, device, monkeypatch):
    import torch
    from torch.utils.data import DataLoader

    from swift.model.models.decision import noncausal_full_attention_mask
    model, template = load_tiny(tiny_dir, device)
    model.eval()
    rows = tiny_rows()
    encoded = [template.encode(row) for row in rows]
    changed = deepcopy(rows[0])
    changed['target_probs'] = [[1., 0.]]
    assert template.encode(changed)['input_ids'] == encoded[0]['input_ids']
    assert template.dummy_model is None
    for field in ('images', 'videos', 'audios'):
        media_row = dict(rows[0], **{field: ['/definitely/missing/media']})
        with pytest.raises(ValueError, match='text'):
            template.encode(media_row)
    batch = template.data_collator(deepcopy(encoded))
    assert batch['decision_meta']['left_padding']
    assert batch['decision_meta']['counts'] == [1, 1, 1]
    mask = noncausal_full_attention_mask(batch['attention_mask'], batch['input_ids'].shape[1], 'npu')
    assert mask.shape == (3, 1, batch['input_ids'].shape[1], batch['input_ids'].shape[1])
    assert torch.equal(mask[:, 0, 0], batch['attention_mask'].bool())
    with torch.no_grad():
        outputs = model(**move_batch(batch, device))
        for i, item in enumerate(encoded):
            single = model(**move_batch(template.data_collator([deepcopy(item)]), device))
            n = single.logits.shape[1]
            torch.testing.assert_close(outputs.logits[i, :n], single.logits[0], rtol=1e-5, atol=1e-6)
    # The model supplies release positions for both inference and the generic training collator.
    template.set_mode('train')
    training_batch = move_batch(template.data_collator([template.encode(row) for row in rows]), device)
    with torch.no_grad():
        training_outputs = model(**training_batch)
    torch.testing.assert_close(outputs.logits, training_outputs.logits, rtol=1e-5, atol=1e-6)
    encoded = [template.encode(row) for row in rows]
    loader = DataLoader(encoded, batch_size=2, num_workers=2, collate_fn=template.data_collator)
    worker_batch = next(iter(loader))
    assert worker_batch['labels'].shape == (2, 2)
    assert template.dummy_model is None
    unlabeled = deepcopy(rows[0])
    unlabeled.pop('target_probs')
    with pytest.raises(ValueError, match='label'):
        template.encode(unlabeled)
    with monkeypatch.context() as patch:
        patch.setattr(template, 'padding_side', 'right')
        with pytest.raises(ValueError, match='left'):
            template.encode(rows[0])
    soft = deepcopy(rows[0])
    soft['target_probs'] = [[2., 3.]]
    assert template.encode(soft)['target_probs'] == [[.4, .6]]
    for target in ([0, 0], [-1, 2], [float('nan'), 1], [float('inf'), 1], [1], True, 2, 'missing'):
        invalid = dict(rows[0], target_probs=[target])
        with pytest.raises(ValueError, match='target'):
            template.encode(invalid)
    for kind, options in [('choice', []), ('choice', ['a'] * 256), ('noul', ['no']), ('score', ['same', 'same']),
                          ('score', ['', 'ok']), ('score', ['x'] * 11)]:
        invalid = dict(rows[0], kinds=[kind], options=[options], target_probs=[[1.] * len(options)])
        with pytest.raises(ValueError, match='option|score|noul'):
            template.encode(invalid)
    template.max_length = 2
    from swift.template.base import MaxLengthError
    with pytest.raises(MaxLengthError):
        template.encode(rows[0])


@pytest.mark.parametrize('feature', ['packing', 'padding_free'])
def test_template_capability_rejects_packing(tiny_dir, tmp_path, feature):
    from swift import SftArguments
    values = dict(
        model=tiny_dir,
        model_type='d3',
        task_type='decision',
        dataset=['unused'],
        split_dataset_ratio=0,
        padding_side='left',
        torch_dtype='float32',
        output_dir=str(tmp_path / 'output'),
        report_to=['none'])
    values[feature] = True
    with pytest.raises(ValueError, match='does not support padding free or packing'):
        SftArguments(**values)


def test_default_lora_training_updates(tiny_dir, device, tmp_path):
    import torch
    from peft import PeftModelForCausalLM

    from swift import SftArguments
    from swift.loss.decision import ScoringLoss
    from swift.pipelines.train.tuner import TunerMixin
    from swift.trainers.decision_trainer import ScoringTrainer
    model, template = load_tiny(tiny_dir, device)
    assert model.model_meta.is_multimodal
    assert model.model_meta.model_arch.arch_name == 'qwen3_5'
    template.set_mode('train')
    args = SftArguments(
        model=tiny_dir,
        model_type='d3',
        task_type='decision',
        dataset=['unused'],
        split_dataset_ratio=0,
        padding_side='left',
        torch_dtype='float32',
        output_dir=str(tmp_path / 'output'),
        report_to=['none'],
        lora_dropout=0.,
        loss_type='scoring')
    model = TunerMixin.prepare_model(args, model, template=template)
    assert isinstance(model, PeftModelForCausalLM)
    assert model.peft_config['default'].task_type == 'CAUSAL_LM'
    base_model = model.get_base_model()
    assert base_model.lm_head.weight is base_model.get_input_embeddings().weight
    ScoringTrainer._activate_scoring_head(type('Holder', (), {'model': model})())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert any('scoring_head.' in name for name in trainable)
    assert any('lora_' in name for name in trainable)
    before = {name: p.detach().clone() for name, p in trainable.items()}
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-3)
    assert len({id(p) for g in optimizer.param_groups for p in g['params']}) == len(trainable)
    for _ in range(3):
        batch = move_batch(template.data_collator([template.encode(row) for row in tiny_rows()]), device)
        labels = batch.pop('labels')
        outputs = model(**batch)
        loss = ScoringLoss(None, object())(outputs, labels)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    changed = {name for name, p in trainable.items() if not torch.equal(before[name], p)}
    assert any('lora_' in name for name in changed)
    assert any('scoring_head.' in name for name in changed)
    assert base_model.lm_head.weight is base_model.get_input_embeddings().weight


@pytest.mark.parametrize('device_type', ['cpu', 'cuda', 'npu'])
@pytest.mark.parametrize('fail', [False, True])
def test_reference_attention_masks(device_type, fail):
    import torch
    from contextlib import nullcontext
    from types import SimpleNamespace

    class MaskProbe(torch.nn.Module):

        def forward(self, hidden_states, attention_mask):
            return attention_mask

    full, linear = MaskProbe(), MaskProbe()
    encoder = SimpleNamespace(stack=[(linear, False), (full, True)])
    hidden = torch.zeros(2, 4, 8)
    present = torch.tensor([[False, False, True, True], [True, True, True, True]])
    key_mask = present[:, None, None, :]
    expected = torch.tensor([[[[False, False, True, True]] * 4], [[[True, True, True, True]] * 4]])
    with pytest.raises(RuntimeError, match='probe failed') if fail else nullcontext():
        with reference_attention_masks(encoder, device_type):
            actual = full(hidden, attention_mask=key_mask)
            if device_type == 'npu':
                assert torch.equal(actual, expected)
                assert actual.is_contiguous()
            else:
                assert actual is key_mask
            assert linear(hidden, attention_mask=present) is present
            if fail:
                raise RuntimeError('probe failed')
    # Neither normal completion nor a reference failure may leave hooks on the shared model.
    assert full(hidden, attention_mask=key_mask) is key_mask
    assert linear(hidden, attention_mask=present) is present


def test_tiny_matches_official_prompt_encoder(tiny_dir, release, device):
    import importlib
    import torch
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
    model, template = load_tiny(tiny_dir, device)
    model.eval()
    # Explicit test opt-in authorizes loading the release reference code; production never imports it.
    sys.path.insert(0, str(release))
    try:
        reference = importlib.import_module('d3_runtime')
    finally:
        sys.path.pop(0)
    batch = move_batch(template.data_collator([template.encode(row) for row in tiny_rows()]), device)
    with torch.no_grad():
        output = model(**batch)
        embeds = model.model.language_model.embed_tokens(batch['input_ids'])
        encoder = reference.PromptEncoder(model.model.language_model)
        with reference_attention_masks(encoder, embeds.device.type):
            hidden = encoder(embeds, batch['attention_mask'])
        reference_logits = hidden[:, -1].float() @ model.scoring_head.readout.weight.T
        for i, n in enumerate(batch['decision_meta']['n_options']):
            torch.testing.assert_close(output.logits[i, :n], reference_logits[i, :n], rtol=1e-5, atol=1e-6)
        positions = torch.arange(embeds.shape[1], device=embeds.device)[None].expand(embeds.shape[0], -1)
        mask_kwargs = dict(
            config=model.model.language_model.config,
            inputs_embeds=embeds,
            attention_mask=batch['attention_mask'],
            past_key_values=None,
            position_ids=positions)
        causal_batch = dict(
            batch,
            attention_mask={
                'full_attention': create_causal_mask(**mask_kwargs),
                'linear_attention': create_recurrent_attention_mask(**mask_kwargs),
            })
        causal = model(**causal_batch)
    same = torch.allclose(output.logits[output.option_mask], causal.logits[causal.option_mask], rtol=1e-5, atol=1e-6)
    assert not same, 'Fixture must distinguish causal from D3 bidirectional full attention.'


def test_shared_decision_head_missing_remains_optional(tmp_path):
    import torch

    from swift.model.decision_head import H2OReadoutHead, maybe_load_trained_scoring_head
    model = torch.nn.Module()
    model.scoring_head = H2OReadoutHead(hidden_size=8, num_codes=3)
    assert not maybe_load_trained_scoring_head(model, str(tmp_path))


def test_shared_dataset_arguments(tiny_dir, tmp_path):
    from swift import SftArguments
    args = SftArguments(
        model=tiny_dir,
        model_type='d3',
        task_type='decision',
        dataset=['unused'],
        split_dataset_ratio=.2,
        padding_side='left',
        torch_dtype='float32',
        strict=False,
        output_dir=str(tmp_path / 'output'),
        report_to=['none'])
    assert args.split_dataset_ratio == .2
    assert not args.strict


@pytest.mark.parametrize('with_validation', [False, True])
def test_sft_training_start(tiny_dir, device, tmp_path, monkeypatch, with_validation):
    import hashlib
    import torch
    from safetensors.torch import load_file

    from swift import SftArguments, sft_main
    root = Path(__file__).resolve().parents[1]
    example = root / 'examples/train/decision'
    monkeypatch.delenv('D3_TRAIN_FILE', raising=False)
    monkeypatch.delenv('D3_VAL_FILE', raising=False)
    monkeypatch.setenv('SWIFT_SINGLE_DEVICE_MODE', '1')
    monkeypatch.setenv('SWIFT_CALIBRATION_MIN_ITEMS', '2')
    # Hash the tiny release, including all backbone tensors, before and after all operations.
    identity = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(tiny_dir).iterdir() if p.is_file()}
    common = dict(
        model=tiny_dir,
        model_type='d3',
        task_type='decision',
        external_plugins=[str(example / 'dataset.py')],
        dataset=['d3_decision'],
        val_dataset=['d3_decision_val'] if with_validation else [],
        split_dataset_ratio=0,
        padding_side='left',
        loss_type='scoring',
        torch_dtype='float32',
        attn_impl='sdpa',
        learning_rate=1e-3,
        lora_dropout=0.,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        dataloader_num_workers=2,
        dataset_num_proc=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        save_steps=2,
        eval_steps=2,
        save_strategy='steps',
        eval_strategy='steps' if with_validation else 'no',
        optim='adamw_torch',
        report_to=['none'],
        save_only_model=False,
        strict=True,
        load_from_cache_file=False,
        seed=17)
    result = sft_main(SftArguments(**common, max_steps=4, output_dir=str(tmp_path / 'train')))
    checkpoint = Path(result['last_model_checkpoint'])
    assert load_file(str(checkpoint / 'scoring_head.safetensors'))
    metadata = json.loads((checkpoint / 'scoring_head.json').read_text())
    assert metadata['head_type'] == 'd3'
    assert set(metadata['temperatures']) == {'noul', 'choice', 'score'}
    if not with_validation:
        assert metadata['head_temperature'] == 1.
        assert set(metadata['temperatures'].values()) == {1.}
    adapter = load_file(str(checkpoint / 'adapter_model.safetensors'))
    assert any('lora_B' in k and torch.count_nonzero(v) for k, v in adapter.items())
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    assert state['global_step'] == 4
    assert identity == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(tiny_dir).iterdir() if p.is_file()
    }
