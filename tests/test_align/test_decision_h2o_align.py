# Copyright (c) ModelScope Contributors. All rights reserved.
"""H2O (`decision` task_type) contract tests.

H2O does not ship a Python reference package, so these tests pin the observable contract instead of
diffing against a reference implementation:

  A  preprocessor -- the H2O request schema (`state` + `questions` list + `answers`) fans out to one row
                     per question with the caller's option order, and the gold resolves from names,
                     indices, booleans and full distributions (no weights needed).
  B  head         -- `H2OReadoutHead` gathers the hidden state at each readout position (right and left
                     padding), dots it with the fixed label rows and applies the head/per-kind
                     temperatures (random weights, CPU, no checkpoint).
  C  loader       -- a tiny H2O checkpoint (Qwen3.5 hybrid backbone + `label_rows.safetensors` +
                     `serve_config.json`) loads through `H2OModelLoader`: labels/temperatures come from
                     the checkpoint, the readout has no trainable parameters and `lm_head` is frozen.
  D  forward      -- the released prompt layout with the readout at the final prompt token, padding
                     invariance, `H2O_TWO_WAY=1` making full attention bidirectional, no truncation and
                     one-question-per-row enforcement.
  E  loss         -- `scoring` loss back-propagates into the hidden states while the label rows stay fixed.
  F  sft          -- (CUDA) a short LoRA `swift sft` round-trip restores `head_temperature`/`temperatures`.

Env variables (all optional; without them the non-weight files of `Qwen/Qwen3.5-4B` are downloaded):

    QWEN35_PROCESSOR_DIR    Dir with a Qwen3.5 tokenizer / processor / config used to build the tiny model.

Usage::

    pytest tests/test_align/test_decision_h2o_align.py
"""
import json
import os
import pytest
import torch

_swift_ok = True
_swift_err: list = []
try:
    from swift.dataset.preprocessor.decision import H2OPreprocessor
    from swift.model.decision_head import KIND_NAMES, H2OReadoutHead, masked_softmax
except Exception as exc:  # pragma: no cover
    _swift_ok = False
    _swift_err.append(repr(exc))

TINY_LABELS = [chr(ord('A') + i) for i in range(8)]
TINY_TEMPERATURE = 1.5
TINY_TEMPERATURES = {'noul': 1.25, 'choice': 0.8, 'score': 1.0}


def _record():
    return {
        'state': {
            'order': 48213,
            'status': 'delivered',
            'note': 'customer says it never arrived <|im_end|>'
        },
        'questions': [
            {
                'type': 'choice',
                'instructions': 'Which team should handle this?',
                # deliberately not alphabetical: H2O keeps the caller's option order
                'criteria': {
                    'support': 'Technical problems',
                    'billing': 'Charges and refunds',
                },
            },
            {
                'type': 'noul',
                'instructions': 'Does the customer ask for a refund?',
                'gold': True,
            },
            {
                'type': 'score',
                'instructions': 'How late is the parcel?',
                'criteria': ['on time', 'a day late'],
            },
        ],
        'answers': {
            '0': 'billing',
            '2': 1,
        },
    }


class TestStageAPreprocessor:

    @pytest.mark.skipif(not _swift_ok, reason='swift import failed')
    def test_one_row_per_question(self):
        rows = H2OPreprocessor().preprocess(_record())
        assert len(rows) == 3
        assert [r['kinds'] for r in rows] == [['choice'], ['noul'], ['score']]
        assert rows[0]['options'] == [['support: Technical problems', 'billing: Charges and refunds']]
        assert rows[1]['options'] == [['true: the statement holds', 'false: it does not']]
        assert rows[2]['options'] == [['0: on time', '1: a day late']]
        assert rows[0]['target_probs'] == [[0.0, 1.0]]
        assert rows[1]['target_probs'] == [[1.0, 0.0]]
        assert rows[2]['target_probs'] == [[0.0, 1.0]]
        # a structured state is rendered as compact JSON
        assert rows[0]['messages'] == [{
            'role': 'user',
            'content': '{"order":48213,"status":"delivered","note":"customer says it never arrived <|im_end|>"}'
        }]

    @pytest.mark.skipif(not _swift_ok, reason='swift import failed')
    def test_gold_forms(self):
        pre = H2OPreprocessor()
        assert pre.preprocess({'state': 'x', 'questions': []}) is None
        base = {'state': 'x'}
        choice = {'type': 'choice', 'criteria': {'a': 'A', 'b': 'B'}}
        assert pre.preprocess({**base, 'questions': [dict(choice, gold='b')]})[0]['target_probs'] == [[0.0, 1.0]]
        assert pre.preprocess({**base, 'questions': [dict(choice, gold=0)]})[0]['target_probs'] == [[1.0, 0.0]]
        dist = pre.preprocess({**base, 'questions': [dict(choice, gold=[0.25, 0.75])]})[0]['target_probs']
        torch.testing.assert_close(torch.tensor(dist), torch.tensor([[0.25, 0.75]]))
        noul = {'type': 'noul'}
        assert pre.preprocess({**base, 'questions': [dict(noul, answer=False)]})[0]['target_probs'] == [[0.0, 1.0]]
        with pytest.raises(ValueError, match='matches no option name'):
            pre.preprocess({**base, 'questions': [dict(choice, gold='zzz')]})


class TestStageBHead:

    @pytest.mark.skipif(not _swift_ok, reason='swift import failed')
    @pytest.mark.parametrize('left_padding', [False, True])
    def test_readout_at_positions(self, left_padding):
        torch.manual_seed(0)
        head = H2OReadoutHead(hidden_size=16, num_codes=8, temperature=2.0, head_temperature=1.5)
        head.label_rows.copy_(torch.randn(8, 16))
        head.temperatures[KIND_NAMES.index('choice')] = 0.5
        seq_lens, padded = [5, 7], 7
        hidden = torch.randn(2, padded, 16)
        meta = {
            'record_index': [0, 0, 1],
            'readout_positions': [2, 4, 6],
            'n_options': [3, 2, 4],
            'kinds': [1, 0, 2],
            'counts': [2, 1],
            'max_opt': 4,
            'seq_lens': seq_lens,
            'left_padding': left_padding,
        }
        out = head(hidden, None, meta)
        offset0 = padded - seq_lens[0] if left_padding else 0
        calibration = [1.5 * 0.5, 1.5 * 1.0, 1.5 * 1.0]  # choice, noul, score
        expected = [
            torch.nn.functional.linear(hidden[0, offset0 + 2], head.label_rows)[:3] / 2.0 / calibration[0],
            torch.nn.functional.linear(hidden[0, offset0 + 4], head.label_rows)[:2] / 2.0 / calibration[1],
            torch.nn.functional.linear(hidden[1, 6], head.label_rows)[:4] / 2.0 / calibration[2],
        ]
        for i, e in enumerate(expected):
            torch.testing.assert_close(out.logits[i, :len(e)], e.float())
        assert out.option_mask.tolist() == [[True, True, True, False], [True, True, False, False],
                                            [True, True, True, True]]
        assert out.counts.tolist() == [2, 1]

    @pytest.mark.skipif(not _swift_ok, reason='swift import failed')
    def test_head_has_no_parameters(self):
        head = H2OReadoutHead(hidden_size=16, num_codes=8)
        assert sum(p.numel() for p in head.parameters()) == 0
        assert head.label_rows.requires_grad is False
        assert float(head.head_temperature) == 1.0

    @pytest.mark.skipif(not _swift_ok, reason='swift import failed')
    def test_calibration_meta_round_trip(self, tmp_path):
        head = H2OReadoutHead(hidden_size=16, num_codes=8, temperature=TINY_TEMPERATURE)
        head.label_rows.copy_(torch.randn(8, 16))
        hidden = torch.randn(1, 6, 16)
        meta = {
            'record_index': [0, 0],
            'readout_positions': [2, 5],
            'n_options': [3, 2],
            'kinds': [1, 0],
            'counts': [1],
            'max_opt': 3,
            'seq_lens': [6],
        }
        before = head(hidden, None, meta).logits
        head.head_temperature = torch.tensor(2.0)
        head.temperatures[0] = 0.5  # noul
        head.save_pretrained(str(tmp_path))
        fresh = H2OReadoutHead(hidden_size=16, num_codes=8, temperature=TINY_TEMPERATURE)
        fresh.label_rows.copy_(head.label_rows)
        assert fresh.load_pretrained(str(tmp_path))
        after = fresh(hidden, None, meta).logits
        torch.testing.assert_close(after[0], before[0] / 2.0)  # choice: head_temperature only
        torch.testing.assert_close(after[1, :2], before[1, :2] / 1.0)  # noul: 2.0 x 0.5


# --------------------------------------------------------------------------------------------------------------
# Stages C/D/E: tiny H2O checkpoint (no release weights needed)
# --------------------------------------------------------------------------------------------------------------


def _qwen35_processor_dir():
    path = os.environ.get('QWEN35_PROCESSOR_DIR')
    if path and os.path.isdir(path):
        return path
    patterns = ['*.json', '*.jinja', '*.txt']
    try:
        if os.environ.get('USE_HF', '0') == '1':
            from huggingface_hub import snapshot_download
            return snapshot_download('Qwen/Qwen3.5-4B', allow_patterns=patterns)
        from modelscope import snapshot_download
        return snapshot_download('Qwen/Qwen3.5-4B', allow_patterns=patterns)
    except Exception:
        return None


def _tiny_device():
    if torch.cuda.is_available():
        return 'cuda'
    try:  # flash-linear-attention kernels need a GPU; without them the reference torch path runs on CPU
        import fla  # noqa: F401
        return None
    except ImportError:
        return 'cpu'


def _two_way_available() -> bool:
    try:
        from transformers.masking_utils import create_recurrent_attention_mask  # noqa: F401
        return True
    except ImportError:
        return False


def _build_tiny_h2o(src: str, out: str) -> None:
    """H2O file layout (standard Qwen3.5 config + model.safetensors, label_rows.safetensors,
    serve_config.json) around a 4-layer Qwen3.5 hybrid backbone (3 gated-deltanet + 1 full-attention)."""
    import shutil
    from safetensors.torch import save_file
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(src):
        if name.endswith(('.json', '.jinja', '.txt')) and name not in ('config.json', 'model.safetensors.index.json'):
            shutil.copy(os.path.join(src, name), out)
    with open(os.path.join(src, 'config.json')) as f:
        base = json.load(f)
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
        layer_types=['linear_attention'] * 3 + ['full_attention'])
    text['rope_parameters'] = dict(text['rope_parameters'], mrope_section=[3, 3, 2])
    vision = dict(base['vision_config'], depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**dict(base, text_config=text, vision_config=vision)))
    model.save_pretrained(out)
    save_file(
        {
            'W': torch.randn(len(TINY_LABELS), 64) * 0.5,
            'token_ids': torch.arange(len(TINY_LABELS), dtype=torch.long),
        }, os.path.join(out, 'label_rows.safetensors'))
    with open(os.path.join(out, 'serve_config.json'), 'w') as f:
        json.dump({'labels': TINY_LABELS, 'temperature': TINY_TEMPERATURES}, f)


@pytest.fixture(scope='module')
def tiny_dir(tmp_path_factory):
    if not _swift_ok:
        pytest.skip(f'swift import failed: {_swift_err}')
    if _tiny_device() is None:
        pytest.skip('flash-linear-attention is installed but no GPU is available')
    src = _qwen35_processor_dir()
    if src is None:
        pytest.skip('Qwen3.5 processor files unavailable (set QWEN35_PROCESSOR_DIR)')
    out = str(tmp_path_factory.mktemp('tiny_h2o'))
    _build_tiny_h2o(src, out)
    return out


@pytest.fixture(scope='module')
def tiny_model(tiny_dir):
    from swift.model import get_model_processor
    model, processor = get_model_processor(
        tiny_dir, model_type='h2o', torch_dtype=torch.float32, attn_impl='sdpa', device_map=_tiny_device())
    return model.eval(), processor


def _template(processor):
    from swift.template import get_template
    template = get_template(processor, template_type='h2o')
    template.set_mode('transformers')
    return template


def _rows(record):
    return H2OPreprocessor().preprocess(record)


def _tiny_forward(model, template, records):
    batch = template.data_collator([template.encode(row) for record in records for row in _rows(record)])
    device = next(model.parameters()).device
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with torch.no_grad():
        return model(**batch)


class TestStageCLoader:

    def test_readout_and_labels_come_from_checkpoint(self, tiny_dir, tiny_model):
        from swift.model.decision_head import get_scoring_head
        model, processor = tiny_model
        head = get_scoring_head(model)
        assert head is not None
        assert processor.h2o_labels == TINY_LABELS
        assert head.num_codes == len(TINY_LABELS)
        assert sum(p.numel() for p in head.parameters()) == 0
        for kind, value in TINY_TEMPERATURES.items():
            assert float(head.temperatures[KIND_NAMES.index(kind)]) == pytest.approx(value)

    def test_lm_head_is_not_trained(self, tiny_model):
        # The decision forward never applies lm_head: it is either tied to embed_tokens (trained
        # through the embeddings) or an untied copy the loader freezes.
        model, _ = tiny_model
        lm_head, embeddings = model.get_output_embeddings(), model.get_input_embeddings()
        if lm_head is None:
            pytest.skip('model has no lm_head')
        tied = lm_head.weight.data_ptr() == embeddings.weight.data_ptr()
        assert tied or lm_head.weight.requires_grad is False


class TestStageDForward:

    def test_prompt_layout_and_readout_position(self, tiny_model):
        _, processor = tiny_model
        template = _template(processor)
        row = _rows(_record())[0]
        encoded = template.encode(row)
        ids = encoded['input_ids']
        assert encoded['decision_meta']['readout_positions'] == [len(ids) - 1]
        assert encoded['decision_meta']['n_options'] == [2]
        text = processor.tokenizer.decode(ids)
        assert 'record: {"order":48213' in text
        assert 'options:\nA) support: Technical problems\nB) billing: Charges and refunds' in text
        assert text.rstrip().endswith('Answer:')

    def test_one_question_per_row(self, tiny_model):
        _, processor = tiny_model
        template = _template(processor)
        row = dict(_rows(_record())[0])
        row['questions'] = row['questions'] * 2
        row['kinds'] = row['kinds'] * 2
        row['options'] = row['options'] * 2
        with pytest.raises(ValueError, match='exactly one question per row'):
            template.encode(row)

    def test_never_truncated(self, tiny_model):
        from swift.template.base import MaxLengthError
        _, processor = tiny_model
        template = _template(processor)
        template.max_length = 16
        for strategy in ('raise', 'left', 'right'):
            template.truncation_strategy = strategy
            with pytest.raises(MaxLengthError, match='never truncated'):
                template.encode(_rows(_record())[0])

    def test_padding_invariance(self, tiny_model):
        model, processor = tiny_model
        template = _template(processor)
        records = [_record(), {**_record(), 'state': 'A different ticket about a refund.'}]
        batched = _tiny_forward(model, template, records)
        singles = torch.cat([_tiny_forward(model, template, [r]).logits for r in records])
        torch.testing.assert_close(batched.logits, singles, atol=1e-4, rtol=0)

    @pytest.mark.skipif(not _two_way_available(), reason='transformers<5.13 has no two-way support')
    def test_two_way_changes_logits(self, tiny_dir, tiny_model, monkeypatch):
        from swift.model import get_model_processor
        causal = _tiny_forward(tiny_model[0], _template(tiny_model[1]), [_record()]).logits
        monkeypatch.setenv('H2O_TWO_WAY', '1')
        model, processor = get_model_processor(
            tiny_dir, model_type='h2o', torch_dtype=torch.float32, attn_impl='sdpa', device_map=_tiny_device())
        bidirectional = _tiny_forward(model.eval(), _template(processor), [_record()]).logits
        assert (causal - bidirectional).abs().max() > 1e-4

    def test_packing_rejected_before_loading(self, tiny_dir):
        from swift import SftArguments
        for flag in ('packing', 'padding_free'):
            with pytest.raises(ValueError, match='does not support padding free or packing'):
                SftArguments(model=tiny_dir, model_type='h2o', task_type='decision', dataset=['dummy'], **{flag: True})


class TestStageELoss:

    @pytest.mark.skipif(not _swift_ok, reason='swift import failed')
    def test_scoring_loss_trains_hidden_states_only(self):
        from swift.loss.decision import ScoringLoss
        torch.manual_seed(0)
        head = H2OReadoutHead(hidden_size=16, num_codes=8, temperature=TINY_TEMPERATURE)
        head.label_rows.copy_(torch.randn(8, 16))
        hidden = torch.randn(1, 6, 16, requires_grad=True)
        meta = {
            'record_index': [0, 0],
            'readout_positions': [2, 5],
            'n_options': [3, 2],
            'kinds': [1, 0],
            'counts': [1],
            'max_opt': 3,
            'seq_lens': [6],
            'left_padding': False,
        }
        out = head(hidden, None, meta)
        loss = ScoringLoss(None, _DummyTrainer())(out, torch.tensor([[0.1, 0.8, 0.1], [0.0, 1.0, 0.0]]))
        loss.backward()
        assert torch.isfinite(loss)
        assert hidden.grad is not None and hidden.grad.abs().sum() > 0
        # the label rows are a buffer: fixed, no gradient, not trained
        assert head.label_rows.grad is None
        probs = masked_softmax(out.logits, out.option_mask)
        assert torch.allclose(probs.sum(-1)[out.option_mask.any(-1)], torch.ones(2))


class _DummyTrainer:
    pass


class TestStageFSftRoundTrip:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for the training round-trip')
    def test_sft_lora_calibration_and_reload(self, tiny_dir, tmp_path, monkeypatch):
        from swift import InferArguments, SftArguments, sft_main
        from swift.dataset import DatasetMeta, register_dataset
        from swift.dataset.preprocessor.decision import H2OPreprocessor as Pre
        from swift.model.decision_head import get_scoring_head
        from swift.pipelines import prepare_model_template
        rows = []
        for i in range(12):
            row = _record()
            row['state'] = f'Ticket {i}: ' + row['state']['note']
            row['questions'] = [{
                'type': 'choice',
                'instructions': 'Which team should handle this?',
                'criteria': {
                    'billing': 'Charges',
                    'support': 'Problems'
                },
                'gold': 'billing' if i % 2 else 'support',
            }]
            rows.append(row)
        data_file = tmp_path / 'tiny.jsonl'
        data_file.write_text('\n'.join(json.dumps(r) for r in rows))
        register_dataset(
            DatasetMeta(dataset_name='h2o_tiny_test', dataset_path=str(data_file), preprocess_func=Pre()),
            exist_ok=True)
        monkeypatch.setenv('SWIFT_CALIBRATION_MIN_ITEMS', '2')
        result = sft_main(
            SftArguments(
                model=tiny_dir,
                model_type='h2o',
                task_type='decision',
                loss_type='scoring',
                dataset=['h2o_tiny_test'],
                split_dataset_ratio=0.5,
                tuner_type='lora',
                target_modules=['all-linear'],
                freeze_vit=True,
                torch_dtype='float32',
                attn_impl='sdpa',
                num_train_epochs=2,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                learning_rate=1e-3,
                output_dir=str(tmp_path / 'output'),
                save_strategy='epoch',
                eval_strategy='epoch',
                report_to=['none']))
        checkpoint = result['last_model_checkpoint']
        assert os.path.exists(os.path.join(checkpoint, 'scoring_head.safetensors'))
        with open(os.path.join(checkpoint, 'scoring_head.json')) as f:
            head_meta = json.load(f)
        assert head_meta['temperature'] == 1.0
        assert 'head_temperature' in head_meta and 'temperatures' in head_meta
        model, _ = prepare_model_template(
            InferArguments(adapters=[checkpoint], model_type='h2o', load_data_args=True))
        head = get_scoring_head(model)
        assert float(head.head_temperature) == pytest.approx(head_meta['head_temperature'])
        assert [float(t) for t in head.temperatures] == pytest.approx(list(head_meta['temperatures'].values()))
