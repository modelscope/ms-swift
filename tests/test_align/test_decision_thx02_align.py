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

Env variables (all optional; missing checkpoint -> stages A and C skip):

    THX02_MODEL_DIR         Path to a THX-02 snapshot dir (else snapshot_download of the hub id).

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
THX02_HF_ID = 'doofz/THX-02'


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
