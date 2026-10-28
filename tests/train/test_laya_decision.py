# Copyright (c) ModelScope Contributors. All rights reserved.
import torch

from swift.dataset.preprocessor.decision import LayaPreprocessor
from swift.loss.decision import ScoringLoss
from swift.model.decision_head import QUESTION_TYPES, LayaHead, masked_softmax
from swift.model.models.decision import LayaModelLoader


def _make_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {'[UNK]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3, '[PAD]': 4}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token='[UNK]',
        cls_token='[CLS]',
        sep_token='[SEP]',
        mask_token='[MASK]',
        pad_token='[PAD]')
    return fast


def _make_template(max_length=64, head_max_len=None):
    from swift.template.templates.decision import LayaTemplate
    tmpl = object.__new__(LayaTemplate)
    # LayaTemplate only needs a plain tokenizer; assign via `processor` (a plain
    # attribute) so the ProcessorMixin.tokenizer property getter works without
    # running the full Template.__init__.
    tmpl.processor = _make_tokenizer()
    tmpl.max_length = max_length
    if head_max_len is not None:
        tmpl._head_max_len = head_max_len
    return tmpl


def _inputs(state, kind, question, options):
    from swift.template.template_inputs import StdTemplateInputs
    return StdTemplateInputs.from_dict({
        'messages': [{
            'role': 'user',
            'content': state
        }],
        'kinds': [kind],
        'questions': [question],
        'options': [options],
    })


def test_laya_template_layout_and_markers():
    tmpl = _make_template()
    encoded = tmpl._encode(_inputs('the sky is blue', 'noul', 'Is the sky blue?', ['false: no', 'true: yes']))
    ids = encoded['input_ids']
    tok = tmpl.tokenizer
    assert ids[0] == tok.cls_token_id and ids[-1] == tok.sep_token_id
    assert ids.count(tok.sep_token_id) == 3  # after question / after options / end of state
    meta = encoded['decision_meta']
    assert meta['n_options'] == [2]
    assert meta['kinds'] == [QUESTION_TYPES['noul']]
    markers = meta['laya_markers'][0]
    assert len(markers) == 2
    for m in markers:
        assert ids[m] == tok.mask_token_id
    # marker j is the token right before option j's text; options are in order.
    assert markers[1] > markers[0]


def test_laya_template_truncates_state_not_markers():
    tmpl = _make_template(max_length=40)
    encoded = tmpl._encode(_inputs('word ' * 200, 'choice', 'pick one', ['a', 'b', 'c']))  # huge state
    ids = encoded['decision_meta'] and encoded['input_ids']
    assert len(ids) <= 40
    markers = encoded['decision_meta']['laya_markers'][0]
    assert len(markers) == 3  # all option markers survived
    for m in markers:
        assert m < len(ids)


def test_laya_template_option_budget_shrink_keeps_markers():
    # 40 options cannot fit head budget 192 with 48-token texts -> shrink evenly, keep all markers.
    tmpl = _make_template(max_length=256, head_max_len=192)
    encoded = tmpl._encode(_inputs('s', 'choice', 'pick', [f'option {i} ' + 'x ' * 40 for i in range(40)]))
    assert len(encoded['decision_meta']['laya_markers'][0]) == 40


def test_laya_head_forward_shapes_and_masks():
    torch.manual_seed(0)
    d, batch, padded_len = 32, 3, 24
    head = LayaHead(hidden_size=d, head_layers=1)
    head.eval()
    hidden = torch.randn(batch, padded_len, d)
    n_options = [2, 4, 3]
    meta = {
        'n_options': n_options,
        'kinds': [QUESTION_TYPES[k] for k in ('noul', 'choice', 'score')],
        'record_index': list(range(batch)),
        'counts': [1] * batch,
        'laya_markers': [[3, 5], [2, 6, 9, 12], [4, 8, 11]],
        'seq_lens': [24, 20, 24],
        'left_padding': False,
        'max_opt': 4,
    }
    out = head(hidden, None, meta)
    assert out.logits.shape == (3, 4)
    assert out.option_mask.dtype == torch.bool
    for i, n in enumerate(n_options):
        assert out.option_mask[i, :n].all() and not out.option_mask[i, n:].any()
        assert torch.isinf(out.logits[i, n:]).all() or (out.logits[i, n:] <= -1e4).all()
        probs = masked_softmax(out.logits[i], out.option_mask[i])
        torch.testing.assert_close(probs.sum(), torch.tensor(1.0))
    assert out.counts.tolist() == [1, 1, 1]


def test_laya_head_eval_temperature_calibration():
    # dropout=0 keeps the forward deterministic in train mode, so the comparison
    # isolates the eval-only temperature division from dropout noise.
    torch.manual_seed(0)
    head = LayaHead(hidden_size=32, head_layers=1, dropout=0.0, temperatures=(2.0, 2.0, 2.0))
    hidden = torch.randn(1, 8, 32)
    meta = {
        'n_options': [2],
        'kinds': [QUESTION_TYPES['noul']],
        'record_index': [0],
        'counts': [1],
        'laya_markers': [[2, 4]],
        'seq_lens': [8],
        'left_padding': False,
        'max_opt': 2,
    }
    head.train()
    train_logits = head(hidden, None, meta).logits
    head.eval()
    eval_logits = head(hidden, None, meta).logits
    torch.testing.assert_close(eval_logits, train_logits / 2.0)


def test_laya_head_temperature_accessors_and_bucket_clearing():
    # Buffer order is Laya qtype (choice, score, noul); accessors are swift-kind order (noul, choice, score).
    head = LayaHead(
        hidden_size=32,
        head_layers=1,
        temperatures=(2.0, 3.0, 4.0),
        temperature_by_options={
            'choice:2': 9.0,
            'choice:6-10': 8.0,
            'noul:2': 5.0
        })
    assert [head.get_temperature(k) for k in range(3)] == [4.0, 2.0, 3.0]
    # A callback refit of `choice` writes through to the Laya-order buffer slot and drops only
    # the choice buckets, so the refit value (not a stale factory bucket) wins at serving.
    head.set_temperature(1, 7.0)
    assert head.temperature.tolist() == [7.0, 3.0, 4.0]
    assert head.temperature_by_options == {'noul:2': 5.0}
    # Reads always reflect the latest write (no snapshot aliasing).
    assert head.get_temperature(1) == 7.0
    # Unrelated kinds' buckets survive a different kind's refit.
    head.set_temperature(0, 1.5)  # noul
    assert head.temperature.tolist() == [7.0, 3.0, 1.5]
    assert head.temperature_by_options == {}
    assert head.get_temperature(0) == 1.5


def _tiny_modernbert():
    from transformers import ModernBertConfig, ModernBertModel
    cfg = ModernBertConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=512,
        pad_token_id=4)
    return ModernBertModel(cfg)


def test_laya_loader_forward_end_to_end_with_loss():
    torch.manual_seed(0)
    model = _tiny_modernbert()
    head = LayaHead(hidden_size=64, head_layers=1)
    loader = object.__new__(LayaModelLoader)
    loader.attach_scoring_head(model, head)
    assert model.scoring_head is head

    tmpl = _make_template(max_length=32)
    encoded = tmpl._encode(_inputs('some state', 'choice', 'pick one', ['a', 'b', 'c']))
    input_ids = torch.tensor([encoded['input_ids']])
    attention_mask = torch.ones_like(input_ids)
    meta = {
        'n_options': encoded['decision_meta']['n_options'],
        'kinds': encoded['decision_meta']['kinds'],
        'record_index': [0],
        'counts': [1],
        'laya_markers': encoded['decision_meta']['laya_markers'],
        'seq_lens': [input_ids.shape[1]],
        'left_padding': False,
        'max_opt': 3,
    }
    model.train()
    out = model(input_ids=input_ids, attention_mask=attention_mask, decision_meta=meta)
    assert out.logits.shape == (1, 3)
    target = torch.tensor([[0.0, 1.0, 0.0]])
    loss = ScoringLoss(None, None)(out, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    # the encoder receives gradients too (full fine-tune path)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
               if 'scoring_head' not in str(id(p)) and p.requires_grad)


def test_laya_head_matches_checkpoint_state_dict_keys():
    """The head's own keys must cover everything the factory model.safetensors carries besides
    encoder.* / act_head.* (head.layers.* / type_emb / scorer / temperature)."""
    head = LayaHead(hidden_size=32, head_layers=2)
    keys = set(head.state_dict().keys())
    assert {'type_emb.weight', 'temperature'} <= keys
    assert any(k.startswith('head.layers.0.') for k in keys)
    assert any(k.startswith('scorer.1.') and 'weight' in k for k in keys)
    assert any(k.startswith('scorer.3.') for k in keys)


def test_laya_preprocessor_native_and_unified_rows():
    prep = LayaPreprocessor()
    native = prep.preprocess({
        'state':
        'billing issue',
        'qs': [{
            't': 'choice',
            'ins': 'Which department?',
            'crit': {
                'billing': 'invoices',
                'other': 'rest'
            },
            'y': 0
        }, {
            't': 'noul',
            'ins': 'Angry?',
            'y': 1
        }, {
            't': 'score',
            'ins': 'How urgent?',
            'crit': ['low', 'high'],
            'soft': [0.25, 0.75]
        }]
    })
    assert len(native) == 3  # one row per question
    choice_row = native[0]
    assert choice_row['options'][0] == ['billing: invoices', 'other: rest']
    assert choice_row['target_probs'] == [[1.0, 0.0]]
    noul_row = native[1]
    assert noul_row['options'][0] == ['false: no, the statement does not hold', 'true: yes, the statement holds']
    assert noul_row['target_probs'] == [[0.0, 1.0]]
    score_row = native[2]
    assert score_row['options'][0] == ['level 0: low', 'level 1: high']
    assert score_row['target_probs'] == [[0.25, 0.75]]

    unified = prep.preprocess({
        'state':
        'review',
        'questions': [{
            'kind': 'choice',
            'question': 'sentiment?',
            'options': ['pos', 'neg'],
            'gold': 1
        }]
    })
    assert unified[0]['options'] == [['pos', 'neg']]
    assert unified[0]['target_probs'] == [[0.0, 1.0]]


def test_laya_preprocessor_kind_inference_and_gold_alias():
    prep = LayaPreprocessor()
    rows = prep.preprocess({
        'state': 's',
        'questions': [{
            'question': 'spam?',
            'options': ['false', 'true'],
            'gold': True
        }]
    })
    assert rows[0]['kinds'] == ['noul']
    assert rows[0]['target_probs'] == [[0.0, 1.0]]
