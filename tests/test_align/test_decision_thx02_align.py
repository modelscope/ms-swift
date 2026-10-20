# Copyright (c) ModelScope Contributors. All rights reserved.
"""THX-02 (`decision` task_type) alignment against the reference implementation shipped in the model repo.

The THX-02 repository carries its own inference package (`thx02/`), which is the reference: it renders the
joint inline-readout prompt, runs the noncausal backbone and applies the readout at every question marker.

Stages:

  A  encoding   -- Thx02Template._encode input_ids / pixel_values / marker positions are identical to the
                   reference encoder (processor only; no model weights).
  B  head       -- Thx02ReadoutHead gathers the hidden state at each marker (right and left padding) and
                   returns readout(h)[:n_options] / temperature (random weights, CPU, no checkpoint).
  C  full fwd   -- the swift-loaded model vs the reference model on the same records (text, JSON state with
                   special tokens, image, several questions per record): probabilities must match.
  D  loss+grad  -- distill_rps on the head output back-propagates into the readout (random weights, CPU).
  E  tiny model -- a randomly initialised THX-02 with the real file layout (small Qwen3.5 hybrid backbone, a few
                   MB) built on the fly, so CI covers the swift path without the checkpoint: loading, template
                   rendering, the noncausal mask, padding invariance, the tied lm_head, no truncation, the
                   --max_pixels bound, early --packing rejection, calibration meta round-trip, and a short
                   `swift sft` (LoRA + head + temperature calibration) followed by reloading the checkpoint.

Env variables (all optional; missing checkpoint -> stages A and C skip):

    THX02_MODEL_DIR         Path to a THX-02 snapshot dir (else snapshot_download of the hub id).
    QWEN35_PROCESSOR_DIR    Stage E: dir with a Qwen3.5 tokenizer / processor / config (else the non-weight files
                            of Qwen/Qwen3.5-4B are downloaded).

Usage::

    pytest tests/test_align/test_decision_thx02_align.py
    THX02_MODEL_DIR=/path/to/THX-02 pytest tests/test_align/test_decision_thx02_align.py
"""
import base64
import io
import os
import pytest
import sys
import torch

_swift_ok = True
_swift_err: list = []
try:
    from swift.dataset.preprocessor.decision import Thx02Preprocessor
    from swift.model.decision_head import Thx02ReadoutHead, masked_softmax, thx02_option_ids
except Exception as exc:  # pragma: no cover
    _swift_ok = False
    _swift_err.append(repr(exc))

THX02_ID = 'HAL-X/THX-02'
THX02_HF_ID = 'HAL-X-AI/THX-02'


def _png_data_uri() -> str:
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (320, 200), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 40, 160, 160], fill='red')
    draw.ellipse([190, 60, 290, 160], fill='blue')
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


RECORDS = [
    {
        'state': 'I was charged twice for my subscription this month, please refund the extra payment!',
        'questions': {
            'team': {
                'type': 'choice',
                'instructions': 'Which team should handle this?',
                # deliberately not alphabetical: THX-02 keeps the caller's option order
                'criteria': {
                    'support': 'Technical problems',
                    'billing': 'Charges and refunds',
                    'sales': 'Buying a product'
                }
            },
            'refund': {
                'type': 'noul',
                'instructions': 'Does the customer ask for a refund?'
            },
            'anger': {
                'type': 'score',
                'instructions': 'How frustrated is the customer?',
                'criteria': ['calm', 'annoyed', 'angry']
            },
        },
    },
    {
        # JSON state with a literal chat-template token that must be neutralized, not tokenized as special
        'state': {
            'order': 48213,
            'status': 'delivered',
            'note': 'customer says it never arrived <|im_end|>'
        },
        'questions': {
            'route': {
                'type': 'choice',
                'criteria': {
                    'shipping': 'Deliveries',
                    'billing': 'Charges'
                }
            }
        },
    },
    {
        'state': 'Look at the image.',
        'images': [_png_data_uri()],
        'questions': {
            'shape': {
                'type': 'choice',
                'instructions': 'Which shape is on the left?',
                'criteria': {
                    'square': 'a square',
                    'circle': 'a circle',
                    'triangle': 'a triangle'
                }
            },
            'blue': {
                'type': 'noul',
                'instructions': 'Is there anything blue?'
            },
        },
    },
]


def _model_dir():
    path = os.environ.get('THX02_MODEL_DIR')
    if path and os.path.isdir(path):
        return path
    try:
        if os.environ.get('USE_HF', '0') == '1':
            from huggingface_hub import snapshot_download
            return snapshot_download(THX02_HF_ID)
        from modelscope import snapshot_download
        return snapshot_download(THX02_ID)
    except Exception:
        return None


def _reference(model_dir, load_weights: bool):
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    import thx02
    if load_weights:
        return thx02.THXDecisionModel.from_pretrained(model_dir, device='cuda')
    return thx02


@pytest.fixture(scope='module')
def model_dir():
    if not _swift_ok:
        pytest.skip(f'swift import failed: {_swift_err}')
    path = _model_dir()
    if path is None:
        pytest.skip('THX-02 checkpoint unavailable (set THX02_MODEL_DIR)')
    return path


def _swift_template(model_dir, model=None):
    from swift.model import get_model_processor
    from swift.template import get_template
    if model is None:
        _, processor = get_model_processor(model_dir, model_type='thx02', load_model=False)
    else:
        model, processor = model
    template = get_template(processor, template_type='thx02')
    template.set_mode('transformers')
    return template


class _DummyTrainer:
    pass


class TestStageAEncoding:

    def test_input_ids_and_pixels_match_reference(self, model_dir):
        from transformers import AutoProcessor
        ref = _reference(model_dir, load_weights=False)
        import json
        with open(os.path.join(model_dir, 'decision_config.json')) as f:
            decision_config = json.load(f)
        ref_processor = AutoProcessor.from_pretrained(model_dir)
        ref_processor.image_processor.size = {
            'shortest_edge': 65536,
            'longest_edge': int(decision_config['max_pixels'])
        }
        encoder = ref.protocol.Encoder(ref_processor, decision_config['codes'])
        template = _swift_template(model_dir)
        pre = Thx02Preprocessor()
        for record in RECORDS:
            expected = encoder.encode(record)
            encoded = template.encode(pre.preprocess(record))
            assert encoded['input_ids'] == expected.input_ids.tolist()
            assert encoded['decision_meta']['readout_positions'] == expected.markers.tolist()
            assert encoded['decision_meta']['n_options'] == list(expected.counts)
            if record.get('images'):
                assert torch.equal(encoded['pixel_values'], expected.pixel_values)
                assert torch.equal(encoded['image_grid_thw'], expected.image_grid_thw)


class TestStageBHead:

    @pytest.mark.skipif(not _swift_ok, reason='swift import failed')
    @pytest.mark.parametrize('left_padding', [False, True])
    def test_readout_at_markers(self, left_padding):
        torch.manual_seed(0)
        head = Thx02ReadoutHead(hidden_size=16, num_codes=8, temperature=1.5)
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
        expected = [
            head.readout(hidden[0, offset0 + 2])[:3] / 1.5,
            head.readout(hidden[0, offset0 + 4])[:2] / 1.5,
            head.readout(hidden[1, 6])[:4] / 1.5,
        ]
        for i, e in enumerate(expected):
            torch.testing.assert_close(out.logits[i, :len(e)], e.float())
        assert out.option_mask.tolist() == [[True, True, True, False], [True, True, False, False],
                                            [True, True, True, True]]
        assert out.counts.tolist() == [2, 1]

    @pytest.mark.skipif(not _swift_ok, reason='swift import failed')
    def test_option_ids_keep_caller_order(self):
        assert thx02_option_ids('choice', {'b': 'x', 'a': 'y'}) == ['b', 'a']
        assert thx02_option_ids('noul', None) == ['false', 'true']
        assert thx02_option_ids('score', ['lo', 'mid', 'hi']) == ['0', '1', '2']


class TestStageCFullForward:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for alignment tests')
    def test_probabilities_match_reference(self, model_dir):
        from swift.model import get_model_processor
        model, processor = get_model_processor(
            model_dir, model_type='thx02', torch_dtype=torch.bfloat16, attn_impl='sdpa', device_map='cuda')
        template = _swift_template(model_dir, (model, processor))
        pre = Thx02Preprocessor()
        batch = template.data_collator([template.encode(pre.preprocess(r)) for r in RECORDS])
        batch = {k: (v.to('cuda') if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        with torch.inference_mode():
            out = model(**batch)
        probs = masked_softmax(out.logits, out.option_mask)
        swift_probs = [probs[i][out.option_mask[i]].tolist() for i in range(probs.shape[0])]
        del model
        torch.cuda.empty_cache()

        ref = _reference(model_dir, load_weights=True)
        options = sys.modules['thx02.protocol'].options
        cursor = 0
        for record, response in zip(RECORDS, ref.decide(RECORDS)):
            for key, question in record['questions'].items():
                answer = response['answers'][key]
                keys = options(question)[0]
                if answer['type'] == 'noul':
                    expected = [answer['noul'] if k == 'true' else 1 - answer['noul'] for k in keys]
                else:
                    expected = [answer['probabilities'][k] for k in keys]
                torch.testing.assert_close(torch.tensor(swift_probs[cursor]), torch.tensor(expected), atol=2e-2, rtol=0)
                cursor += 1
        assert cursor == len(swift_probs)


class TestStageDLossGrad:

    @pytest.mark.skipif(not _swift_ok, reason='swift import failed')
    def test_distill_rps_trains_readout(self):
        from swift.loss.decision import DistillRPSLoss
        torch.manual_seed(0)
        head = Thx02ReadoutHead(hidden_size=16, num_codes=8, temperature=1.554)
        hidden = torch.randn(1, 6, 16)
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
        labels = torch.tensor([[0.1, 0.8, 0.1], [0.0, 1.0, 0.0]])
        loss = DistillRPSLoss(None, _DummyTrainer())(out, labels)
        loss.backward()
        assert torch.isfinite(loss)
        assert head.readout.weight.grad is not None and head.readout.weight.grad.abs().sum() > 0
        assert head.readout.weight.grad[3:].abs().sum() == 0  # codes beyond every question's options get no gradient


# --------------------------------------------------------------------------------------------------------------
# Stage E: tiny randomly initialised THX-02 (no checkpoint needed)
# --------------------------------------------------------------------------------------------------------------

TINY_CODES = [chr(ord('A') + i) for i in range(8)]
TINY_TEMPERATURE = 1.5


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


def _build_tiny_thx02(src: str, out: str) -> None:
    """THX-02 file layout (own config.json with `backbone_config`, prefix-free model.safetensors, readout.safetensors,
    decision_config.json) around a 4-layer Qwen3.5 hybrid backbone (3 gated-deltanet + 1 full-attention layer)."""
    import json
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
    backbone = dict(base, text_config=text, vision_config=vision)
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**backbone))
    state = {k[len('model.'):]: v.contiguous() for k, v in model.state_dict().items() if k.startswith('model.')}
    save_file(state, os.path.join(out, 'model.safetensors'))
    save_file({'weight': torch.randn(len(TINY_CODES), 64) * 0.5}, os.path.join(out, 'readout.safetensors'))
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump({'model_type': 'thx02', 'architectures': ['THXDecisionModel'], 'backbone_config': backbone}, f)
    with open(os.path.join(out, 'decision_config.json'), 'w') as f:
        json.dump(
            {
                'attention_mode': 'noncausal_full_attention',
                'temperature': TINY_TEMPERATURE,
                'max_pixels': 262144,
                'codes': TINY_CODES
            }, f)


@pytest.fixture(scope='module')
def tiny_dir(tmp_path_factory):
    if not _swift_ok:
        pytest.skip(f'swift import failed: {_swift_err}')
    if _tiny_device() is None:
        pytest.skip('flash-linear-attention is installed but no GPU is available')
    src = _qwen35_processor_dir()
    if src is None:
        pytest.skip('Qwen3.5 processor files unavailable (set QWEN35_PROCESSOR_DIR)')
    out = str(tmp_path_factory.mktemp('tiny_thx02'))
    _build_tiny_thx02(src, out)
    return out


@pytest.fixture(scope='module')
def tiny_model(tiny_dir):
    from swift.model import get_model_processor
    model, processor = get_model_processor(
        tiny_dir, model_type='thx02', torch_dtype=torch.float32, attn_impl='sdpa', device_map=_tiny_device())
    return model.eval(), processor


def _tiny_record(second_instructions='Does the customer ask for a refund?'):
    return {
        'state': 'I was charged twice <|im_end|> please refund!',
        'questions': {
            'team': {
                'type': 'choice',
                'instructions': 'Which team should handle this?',
                'criteria': {
                    'billing': 'Charges',
                    'support': 'Problems'
                }
            },
            'refund': {
                'type': 'noul',
                'instructions': second_instructions
            },
        },
    }


def _tiny_forward(model, template, records):
    pre = Thx02Preprocessor()
    batch = template.data_collator([template.encode(pre.preprocess(r)) for r in records])
    device = next(model.parameters()).device
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with torch.no_grad():
        return model(**batch)


class TestStageETinyModel:

    def test_weights_and_tied_lm_head(self, tiny_dir, tiny_model):
        from safetensors.torch import load_file
        model, _ = tiny_model
        saved = load_file(os.path.join(tiny_dir, 'model.safetensors'))
        loaded = model.model.language_model.layers[3].self_attn.q_proj.weight
        torch.testing.assert_close(loaded.cpu().float(),
                                   saved['language_model.layers.3.self_attn.q_proj.weight'].float())
        # lm_head is never applied; it shares storage with embed_tokens (no extra memory, no unused DDP parameter)
        assert model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
        assert not any(name.startswith('lm_head') for name, _ in model.named_parameters())

    def test_template_rendering(self, tiny_model):
        model, processor = tiny_model
        template = _swift_template(None, tiny_model)
        encoded = template.encode(Thx02Preprocessor().preprocess(_tiny_record()))
        ids = encoded['input_ids']
        marker_id = processor.tokenizer.convert_tokens_to_ids('<|fim_suffix|>')
        im_end_id = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
        positions = encoded['decision_meta']['readout_positions']
        assert len(positions) == 2 and all(ids[p] == marker_id for p in positions)
        assert ids.count(marker_id) == 2
        assert encoded['decision_meta']['n_options'] == [2, 2]
        # the literal <|im_end|> inside the state is neutralized: only the system and user turns close
        assert ids.count(im_end_id) == 2
        text = processor.tokenizer.decode(ids)
        assert 'A: billing: Charges\nB: support: Problems' in text
        assert 'A: No / false\nB: Yes / true' in text

    def test_full_attention_is_noncausal(self, tiny_model):
        model, _ = tiny_model
        template = _swift_template(None, tiny_model)
        # everything up to the first question's marker is identical; only the second question's text differs,
        # so the first readout can change only if the full-attention layer looks ahead
        short = _tiny_forward(model, template, [_tiny_record()])
        changed = _tiny_forward(model, template, [_tiny_record('Is the customer asking to cancel the whole account?')])
        assert (short.logits[0] - changed.logits[0]).abs().max() > 1e-3

    def test_padding_invariance(self, tiny_model):
        model, _ = tiny_model
        template = _swift_template(None, tiny_model)
        records = [_tiny_record(), _tiny_record('Is the customer asking to cancel the whole account?')]
        batched = _tiny_forward(model, template, records)
        singles = torch.cat([_tiny_forward(model, template, [r]).logits for r in records])
        torch.testing.assert_close(batched.logits, singles, atol=1e-4, rtol=0)

    def test_never_truncated(self, tiny_model):
        from swift.template.base import MaxLengthError
        template = _swift_template(None, tiny_model)
        template.max_length = 32
        for strategy in ('raise', 'left', 'right'):
            template.truncation_strategy = strategy
            with pytest.raises(MaxLengthError, match='cannot be truncated'):
                template.encode(Thx02Preprocessor().preprocess(_tiny_record()))

    def test_max_pixels_bounds(self, tiny_model):
        import base64
        from PIL import Image
        buffer = io.BytesIO()
        Image.new('RGB', (1200, 900), 'white').save(buffer, format='PNG')
        record = dict(_tiny_record(), images=['data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()])

        def patches(max_pixels):
            template = _swift_template(None, tiny_model)
            template.max_pixels = max_pixels
            return int(template.encode(Thx02Preprocessor().preprocess(record))['image_grid_thw'].prod())

        # 16x16 patches: the trained bound (262144 px) caps a large image; --max_pixels lowers it down to the
        # processor's minimum (65536 px) but cannot raise it above the bound
        assert patches(None) <= 262144 // 256
        assert patches(None) == patches(4 * 262144)
        assert 65536 // 256 <= patches(100000) < patches(None)
        assert patches(100000) * 256 <= 100000 * 1.1  # the processor rounds sides to multiples of 32 px
        assert patches(16384) >= 65536 // 256

    def test_calibration_meta_round_trip(self, tmp_path):
        head = Thx02ReadoutHead(hidden_size=16, num_codes=8, temperature=TINY_TEMPERATURE)
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
        fresh = Thx02ReadoutHead(hidden_size=16, num_codes=8, temperature=TINY_TEMPERATURE)
        fresh.readout.load_state_dict(head.readout.state_dict())
        assert fresh.load_pretrained(str(tmp_path))
        after = fresh(hidden, None, meta).logits
        torch.testing.assert_close(after[0], before[0] / 2.0)  # choice: head_temperature only
        torch.testing.assert_close(after[1, :2], before[1, :2] / 1.0)  # noul: 2.0 x 0.5

    def test_packing_rejected_before_loading(self, tiny_dir):
        from swift import SftArguments
        for flag in ('packing', 'padding_free'):
            with pytest.raises(ValueError, match='THX-02 does not support'):
                SftArguments(
                    model=tiny_dir, model_type='thx02', task_type='decision', dataset=['dummy'], **{flag: True})

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for the training round-trip')
    def test_sft_lora_calibration_and_reload(self, tiny_dir, tmp_path, monkeypatch):
        import json

        from swift import InferArguments, SftArguments, sft_main
        from swift.dataset import DatasetMeta, register_dataset
        from swift.dataset.preprocessor.decision import Thx02Preprocessor as Pre
        from swift.model.decision_head import get_scoring_head
        from swift.pipelines import prepare_model_template
        rows = []
        for i in range(12):
            record = _tiny_record()
            record['state'] = f'Ticket {i}: ' + record['state']
            record['answers'] = {'team': 'billing' if i % 2 else 'support', 'refund': bool(i % 3)}
            rows.append(record)
        data_file = tmp_path / 'tiny.jsonl'
        data_file.write_text('\n'.join(json.dumps(r) for r in rows))
        register_dataset(
            DatasetMeta(dataset_name='thx02_tiny_test', dataset_path=str(data_file), preprocess_func=Pre()),
            exist_ok=True)
        monkeypatch.setenv('SWIFT_CALIBRATION_MIN_ITEMS', '2')
        result = sft_main(
            SftArguments(
                model=tiny_dir,
                model_type='thx02',
                task_type='decision',
                loss_type='distill_rps',
                dataset=['thx02_tiny_test'],
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
        assert head_meta['temperature'] == TINY_TEMPERATURE
        assert head_meta['head_temperature'] != 1.0  # the calibration callback fitted it
        model, template = prepare_model_template(InferArguments(adapters=[checkpoint], load_data_args=True))
        head = get_scoring_head(model)
        assert float(head.head_temperature) == pytest.approx(head_meta['head_temperature'])
        assert [float(t) for t in head.temperatures] == pytest.approx(list(head_meta['temperatures'].values()))
