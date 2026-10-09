# Copyright (c) ModelScope Contributors. All rights reserved.
"""Clef (`decision` task_type) alignment against the official `joint_schema_model.py` oracle.

Portable version of ``tests/test_align/test_decision_clef_align.py``.  The original test
logic (Stages A / BC / D and their assertions) is preserved verbatim; the only changes are:

  1. **Env / fixture injection**: the Clef checkpoint dir is resolved from
     ``CLEF_MODEL_DIR`` (falling back to ``snapshot_download``) instead of a
     hard-coded path.  ``joint_schema_model`` is imported lazily after the
     checkpoint dir is added to ``sys.path``.
  2. **Skip-on-missing**: when the checkpoint is unavailable, alignment tests
     ``pytest.skip()`` instead of hard-failing.
  3. **Structural / contract tier**: Stage D (loss + grad) is also runnable with
     a *random-initialised* ``ClefJointSchemaHead`` of identical structure, so
     the loss/grad contract is exercised on CPU without a checkpoint download.

Stages (unchanged from the original):

  A  encoding   -- ClefTemplate._encode input_ids + question/option spans are BYTE-identical to
                   official encode_record (tokenizer only; no model weights, cheap).
  B  head-only  -- feed the official backbone's last_hidden_state into BOTH the official
                   JointSchemaHead and our ClefJointSchemaHead (same weights, fp32): logits must
                   match to fp32 precision (isolates the head port).
  C  full fwd   -- our swift-loaded model end-to-end vs official ClefModel on the same encoding
                   (bf16, eager both sides): validates hidden_states[-1] == last_hidden_state and
                   the whole attach/patch path.
  D  loss+grad  -- Clef ships no training loss, so ClefLoss is validated three ways: value == an
                   independently re-derived label-smoothing-CE + Brier; its autograd grads == the
                   re-derived formula's grads (grad_norm + meaningful-param rel); autograd ==
                   central finite-difference.

Env variables (all optional; missing checkpoint → alignment tests skip):

    CLEF_MODEL_DIR          Path to the Cloudflare/clef snapshot dir.

Usage::

    pytest test_decision_clef_align.py                # structural D + alignment (skip if no ckpt)
    CLEF_MODEL_DIR=… pytest test_decision_clef_align.py   # all stages
    python test_decision_clef_align.py --stage all    # legacy CLI entrypoint
"""
import argparse
import copy
import os
import sys

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('USE_HF', '1')

import pytest
import torch

# ---------------------------------------------------------------------------
# Guarded imports — module import never hard-fails in a minimal CI environment.
# ---------------------------------------------------------------------------
_swift_ok = True
_swift_err: list = []
try:
    from swift.dataset.preprocessor.decision import ClefPreprocessor
    from swift.loss.decision import ClefLoss
    from swift.model.decision_head import ClefJointSchemaHead, masked_softmax
    from swift.template.template_inputs import StdTemplateInputs
except Exception as exc:  # pragma: no cover
    _swift_ok = False
    _swift_err.append(repr(exc))

CLEF_ID = 'Cloudflare/clef'

# ---------------------------------------------------------------------------
# Records — verbatim from the original.
# ---------------------------------------------------------------------------
RECORDS = [
    {
        'id': 'r_noul',
        'state': {'text': 'The sky is blue.', 'lang': 'en'},
        'questions': {'is_blue': {'type': 'noul', 'instructions': 'Is the sky blue?'}},
    },
    {
        'id': 'r_choice',
        'state': 'A review: the product arrived late and was damaged.',
        'questions': {
            'sentiment': {
                'type': 'choice',
                'instructions': 'Classify the sentiment.',
                'criteria': {'positive': 'Good', 'negative': 'Bad', 'neutral': 'Mixed'},
            },
        },
    },
    {
        'id': 'r_score',
        'state': {'doc': 'some text to rate'},
        'questions': {
            'quality': {
                'type': 'score',
                'instructions': 'Rate quality.',
                'criteria': ['terrible', 'poor', 'ok', 'good', 'great'],
            },
        },
    },
    {
        'id': 'r_joint',
        'state': {'user': 'alice', 'msg': 'hello there', 'flag': True},
        'questions': {
            'spam': {'type': 'noul', 'instructions': 'Is this spam?'},
            'topic': {
                'type': 'choice',
                'instructions': 'Pick a topic.',
                'criteria': {'b_tech': 'Technology', 'a_sport': 'Sports', 'c_food': 'Food'},
            },
            'polite': {'type': 'score', 'instructions': 'Politeness level.',
                       'criteria': ['rude', 'neutral', 'polite']},
        },
    },
]
GOLD_IDX = {'r_joint': [0, 1, 2]}  # noul->true, choice->sorted[1]=b_tech, score->levels[2]=polite


# ---------------------------------------------------------------------------
# Shared helpers — verbatim from the original.
# ---------------------------------------------------------------------------
class _DummyTrainer:
    pass


class _EmbShim(torch.nn.Module):
    """A real nn.Module wrapping an fp32 output-embedding weight (head.set_output_embeddings
    assigns it as a child module, so a plain object is rejected)."""

    def __init__(self, w):
        super().__init__()
        self.register_buffer('weight', w)


def build_meta(encoded, device, input_ids):
    """Flatten one encoded record into the batch=1 `decision_meta` our collator would produce
    (record-major, no padding)."""
    nq = len(encoded.questions)
    n_options = [len(q.option_ids) for q in encoded.questions]
    return {
        'n_options': n_options,
        'kinds': [q.question_type for q in encoded.questions],
        'question_spans': [tuple(q.question_span) for q in encoded.questions],
        'option_spans': [[tuple(s) for s in q.option_spans] for q in encoded.questions],
        'record_index': [0] * nq,
        'counts': [nq],
        'max_opt': max(n_options),
        'left_padding': False,
        'seq_lens': [len(encoded.input_ids)],
        'input_ids': input_ids,
    }


def maxdiff(a, b):
    a = torch.as_tensor(a, dtype=torch.float64).flatten()
    b = torch.as_tensor(b, dtype=torch.float64).flatten()
    return float((a - b).abs().max()) if a.numel() else 0.0


def rederived_clef_loss(logits, mask, target, eps=0.1, brier_w=1.0):
    """Independent re-derivation of ClefLoss: label-smoothing CE + Brier over real options only."""
    neg = torch.finfo(logits.dtype).min
    logp = torch.log_softmax(logits.masked_fill(~mask, neg), dim=-1)
    p = torch.exp(logp) * mask
    n_real = mask.sum(-1, keepdim=True).clamp(min=1).to(target.dtype)
    t_smooth = (1.0 - eps) * target + eps * (mask.to(target.dtype) / n_real)
    ce = -(t_smooth * logp * mask).sum(-1)
    ce_loss = ce.sum() / ce.shape[0]
    diff2 = (p - target).pow(2) * mask
    brier = diff2.sum(-1) / n_real.squeeze(-1).to(diff2.dtype)
    return ce_loss + brier_w * brier.sum() / brier.shape[0]


# ---------------------------------------------------------------------------
# Env / fixture plumbing
# ---------------------------------------------------------------------------
def _resolve_clef_dir():
    """Return the Clef snapshot dir, or *None* if unavailable."""
    d = os.environ.get('CLEF_MODEL_DIR')
    if d and os.path.isdir(d):
        return d
    try:
        from modelscope import snapshot_download
        return snapshot_download(CLEF_ID)
    except Exception:
        return None


_CLEF_DIR = None
_REF = None  # cached joint_schema_model oracle


def _get_clef_dir():
    global _CLEF_DIR
    if _CLEF_DIR is None:
        _CLEF_DIR = _resolve_clef_dir()
    return _CLEF_DIR


def _get_ref():
    """Import and cache the official ``joint_schema_model`` oracle."""
    global _REF
    if _REF is not None:
        return _REF
    clef_dir = _get_clef_dir()
    if clef_dir is None:
        pytest.skip('Clef checkpoint unavailable (set CLEF_MODEL_DIR to enable alignment tests)')
    if clef_dir not in sys.path:
        sys.path.insert(0, clef_dir)
    import joint_schema_model as ref
    _REF = ref
    return ref


def _need_swift():
    if not _swift_ok:
        pytest.skip(f'swift imports failed: {_swift_err}')


def _need_clef_dir():
    if _get_clef_dir() is None:
        pytest.skip('Clef checkpoint unavailable (set CLEF_MODEL_DIR to enable alignment tests)')


# ---------------------------------------------------------------------------
# Random-initialised head factory (structural D — no checkpoint needed)
# ---------------------------------------------------------------------------
# Small-dim ClefJointSchemaHead for CPU; same structural contract as the real
# head (all submodules + scalar params present), just random weights.
_RAND_HIDDEN = 128
_RAND_WIDTH = 64
_RAND_VOCAB = 1000


def _make_random_head():
    """Build a random ClefJointSchemaHead of identical structure (small dims for CPU).

    The scalar params (prior_logit_scale, joint_logit_scale, residual_gate) are *not* left
    at their zero default — the real checkpoint has non-zero values there, and the
    finite-difference check (eps=1e-3) is better conditioned with non-zero inits.
    """
    head = ClefJointSchemaHead(
        hidden_size=_RAND_HIDDEN, width=_RAND_WIDTH, routing_layers=1, layers=1,
        heads=4, feedforward=_RAND_WIDTH * 2, dropout=0.0)
    for p in head.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=0.02)
    return head


def _make_mock_meta(nq, n_options, seq_len, device='cpu'):
    """Build a mock decision_meta for structural head tests (single record, right-pad).

    Spans are 0-based over [0, seq_len): question_span covers the whole seq;
    each option spans 1 token at distinct positions.
    """
    input_ids = torch.randint(0, _RAND_VOCAB, (1, seq_len), device=device, dtype=torch.long,
                              generator=torch.Generator().manual_seed(99))
    return {
        'n_options': n_options,
        'kinds': [i % 3 for i in range(nq)],
        'question_spans': [(0, seq_len)] * nq,
        'option_spans': [[(i, i + 1) for i in range(n)] for n in n_options],
        'record_index': [0] * nq,
        'counts': [nq],
        'max_opt': max(n_options),
        'left_padding': False,
        'seq_lens': [seq_len],
        'input_ids': input_ids,
    }


# ===========================================================================
# Stage A: encoding  (requires checkpoint — skips if missing)
# ===========================================================================

class TestStageA:
    """Stage A: encoding byte-identity with the real tokenizer + oracle.

    Verbatim logic from the original ``stage_a()``.
    """

    def test_encoding_byte_identical(self):
        _need_swift()
        _need_clef_dir()
        ref = _get_ref()
        from swift.model import get_processor
        from swift.template import get_template
        clef_dir = _get_clef_dir()
        clef_dir = _get_clef_dir()
        processor = get_processor(clef_dir, model_type='clef', task_type='decision')
        template = get_template(processor, template_type='clef')
        template.set_mode('train')
        tok = processor.tokenizer
        for rec in RECORDS:
            norm = ClefPreprocessor().preprocess(dict(rec))
            enc = template._encode(StdTemplateInputs.from_dict(norm))
            meta = enc['decision_meta']
            o_ids = list(enc['input_ids'])
            o_q = [tuple(x) for x in meta['question_spans']]
            o_o = [[tuple(s) for s in q] for q in meta['option_spans']]
            o_k = list(meta['kinds'])
            e = ref.encode_record(tok, {'model': 'clef', **rec}, processor=processor)
            r_ids = list(e.input_ids)
            r_q = [tuple(q.question_span) for q in e.questions]
            r_o = [[tuple(s) for s in q.option_spans] for q in e.questions]
            r_k = [q.question_type for q in e.questions]
            ok = (o_ids == r_ids) and (o_q == r_q) and (o_o == r_o) and (o_k == r_k)
            if not ok and o_ids != r_ids:
                n = min(len(o_ids), len(r_ids))
                div = next((i for i in range(n) if o_ids[i] != r_ids[i]), n)
                lo = max(0, div - 8)
                assert False, (f'{rec["id"]}: first id divergence at {div}: '
                               f'ours={tok.decode(o_ids[lo:div + 8])!r} '
                               f'ref={tok.decode(r_ids[lo:div + 8])!r}')
            assert ok, (f'{rec["id"]}: ids={o_ids == r_ids} qspans={o_q == r_q} '
                        f'ospans={o_o == r_o} kinds={o_k == r_k} (len {len(o_ids)} vs {len(r_ids)})')


# ===========================================================================
# Stage B + C: forward logits  (requires checkpoint + CUDA — skips if missing)
# ===========================================================================

class TestStageBC:
    """Stage BC: head-only fp32 + full forward bf16 alignment.

    Verbatim logic from the original ``stage_bc()``.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for alignment tests')
    def test_head_and_forward_alignment(self, dev_ref='cuda:0', dev_ours='cuda:1'):
        _need_swift()
        _need_clef_dir()
        ref = _get_ref()
        from swift.model import get_model_processor, get_processor
        clef_dir = _get_clef_dir()
        print('loading reference ClefModel (bf16, eager) ...', flush=True)
        ref_model, processor = ref.load_release_model(
            clef_dir, device=dev_ref, dtype=torch.bfloat16, attn_implementation='eager')
        ref_model.eval()
        tok, pad_id = processor.tokenizer, processor.tokenizer.pad_token_id
        print('loading OUR swift clef model (bf16, eager) ...', flush=True)
        our_model, _ = get_model_processor(
            _get_clef_dir(), model_type='clef', task_type='decision', torch_dtype=torch.bfloat16,
            device_map=dev_ours, use_hf=True, attn_impl='eager')
        our_model.eval()

        ref_head32 = copy.deepcopy(ref_model.head).float().to(dev_ref)
        our_head32 = copy.deepcopy(our_model.scoring_head).float().to(dev_ref)

        base_model = ref_model.language_model
        text_model = base_model.model
        if hasattr(text_model, 'language_model'):
            text_model = text_model.language_model

        for rec in RECORDS:
            encoded = ref.encode_record(tok, {'model': 'clef', **rec}, processor=processor)
            nq = len(encoded.questions)
            n_options = [len(q.option_ids) for q in encoded.questions]
            ids_ref = torch.tensor([encoded.input_ids], dtype=torch.long, device=dev_ref)
            am_ref = torch.ones_like(ids_ref)
            batch_ref = ref.collate_records([encoded], pad_id, torch.device(dev_ref))
            with torch.inference_mode():
                last_hidden = text_model(
                    input_ids=ids_ref, attention_mask=am_ref, use_cache=False,
                    return_dict=True).last_hidden_state
                emb_w = base_model.get_output_embeddings().weight
                ref_logits = ref_model(batch_ref)[0]

            # Stage B: head-only fp32 on the SAME last_hidden.
            meta_ref = build_meta(encoded, dev_ref, ids_ref)
            h32, emb32 = last_hidden.float(), emb_w.float()
            our_head32.set_output_embeddings(_EmbShim(emb32))
            with torch.inference_mode():
                ref_head_logits = ref_head32(h32, ids_ref, am_ref, [encoded], emb32)[0]
                our32 = our_head32(h32, None, meta_ref)
            b_max = max((maxdiff(ref_head_logits[i].float().cpu(),
                                  our32.logits[i, :n_options[i]].float().cpu())
                         for i in range(nq)), default=0.0)
            assert b_max < 2e-4, (f'{rec["id"]}: B head-only fp32 max|logit diff|={b_max:.3e} '
                                  f'(FAIL <2e-4)')

            # Stage C: full forward, our model bf16 eager.
            ids_o, am_o = ids_ref.to(dev_ours), am_ref.to(dev_ours)
            meta_o = build_meta(encoded, dev_ours, ids_o)
            with torch.inference_mode():
                our_out = our_model(input_ids=ids_o, attention_mask=am_o, decision_meta=meta_o)
            c_l, c_p = [], []
            for i in range(nq):
                rl = ref_logits[i].detach().float().cpu()
                ol = our_out.logits[i, :n_options[i]].detach().float().cpu()
                c_l.append(maxdiff(rl, ol))
                op = masked_softmax(our_out.logits[i:i + 1].float(),
                                     our_out.option_mask[i:i + 1])[0, :n_options[i]].cpu()
                c_p.append(maxdiff(torch.softmax(rl, dim=-1), op))
            c_lmax, c_pmax = max(c_l, default=0.0), max(c_p, default=0.0)
            assert c_lmax < 0.25 and c_pmax < 2e-2, (
                f'{rec["id"]}: C full bf16 max|logit diff|={c_lmax:.3e} max|prob diff|={c_pmax:.3e} (FAIL)')


# ===========================================================================
# Stage D: loss + grad  (two paths — checkpoint alignment + structural random)
# ===========================================================================

class TestStageDAlignment:
    """Stage D with the real checkpoint: loss+grad alignment.

    Verbatim logic from the original ``stage_d()``.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for alignment tests')
    def test_loss_value_and_grad(self, dev='cuda:0'):
        _need_swift()
        _need_clef_dir()
        ref = _get_ref()
        from swift.model import get_model_processor, get_processor
        clef_dir = _get_clef_dir()
        rec = next(r for r in RECORDS if r['id'] == 'r_joint')
        gold = GOLD_IDX['r_joint']
        print('loading OUR swift clef model (bf16, eager) ...', flush=True)
        our_model, processor = get_model_processor(
            _get_clef_dir(), model_type='clef', task_type='decision', torch_dtype=torch.bfloat16,
            device_map=dev, use_hf=True, attn_impl='eager')
        our_model.eval()
        tok = processor.tokenizer
        encoded = ref.encode_record(tok, {'model': 'clef', **rec}, processor=processor)
        nq = len(encoded.questions)
        n_options = [len(q.option_ids) for q in encoded.questions]
        max_opt = max(n_options)
        ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=dev)
        am = torch.ones_like(ids)
        meta = build_meta(encoded, dev, ids)
        target = torch.zeros((nq, max_opt), dtype=torch.float32, device=dev)
        for i, g in enumerate(gold):
            target[i, g] = 1.0

        text_model = our_model.model
        if hasattr(text_model, 'language_model'):
            text_model = text_model.language_model
        with torch.no_grad():
            hidden = text_model(input_ids=ids, attention_mask=am, use_cache=False,
                                return_dict=True).last_hidden_state.detach().float().clone()
        emb32 = our_model.get_output_embeddings().weight.detach().float().clone()
        head32 = copy.deepcopy(our_model.scoring_head).float().to(dev)
        head32.set_output_embeddings(_EmbShim(emb32))
        for p in head32.parameters():
            p.requires_grad_(True)

        # (1) value
        clef_loss = ClefLoss(None, _DummyTrainer())
        out32 = head32(hidden, None, meta)
        our_val = float(clef_loss(outputs=out32, labels=target, num_items_in_batch=None))
        re_val = float(rederived_clef_loss(out32.logits, out32.option_mask, target))
        assert abs(our_val - re_val) < 1e-5, (
            f'D(1) loss value our={our_val:.8f} rederived={re_val:.8f} '
            f'|diff|={abs(our_val - re_val):.3e}')

        # (2) autograd our-vs-rederived
        named = [(n, p) for n, p in head32.named_parameters() if p.requires_grad]
        head32.zero_grad(set_to_none=True)
        clef_loss(outputs=head32(hidden, None, meta), labels=target,
                  num_items_in_batch=None).backward()
        g_our = {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in named}
        head32.zero_grad(set_to_none=True)
        o2 = head32(hidden, None, meta)
        rederived_clef_loss(o2.logits, o2.option_mask, target).backward()
        g_re = {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in named}

        def gnorm(gd):
            return float(torch.sqrt(sum((g.double() ** 2).sum() for g in gd.values() if g is not None)))

        n_our, n_re = gnorm(g_our), gnorm(g_re)
        norm_rel = abs(n_our - n_re) / max(n_our, n_re, 1e-12)
        worst, worst_name = 0.0, None
        for n, _ in named:
            a, b = g_our[n], g_re[n]
            if a is None or b is None or float(b.abs().max()) < 1e-4:
                continue
            rel = float((a - b).abs().max()) / float(b.abs().max())
            if rel > worst:
                worst, worst_name = rel, n
        assert norm_rel < 1e-4 and worst < 1e-4, (
            f'D(2) autograd our-vs-rederived grad_norm rel={norm_rel:.3e}; '
            f'worst meaningful-param rel={worst:.3e} @ {worst_name}')

        # (3) finite-difference
        eps = 1e-3
        for name, idx in [('prior_logit_scale', ()), ('joint_logit_scale', ()), ('residual_gate', ()),
                          ('residual_scorer.0.weight', (0, 0))]:
            p = dict(head32.named_parameters())[name]
            autog = float(g_re[name]) if idx == () else float(g_re[name][idx])

            def loss_val():
                with torch.no_grad():
                    o = head32(hidden, None, meta)
                    return float(rederived_clef_loss(o.logits, o.option_mask, target))

            def get_val():
                return float(p.data) if idx == () else float(p.data[idx])

            def set_val(v):
                with torch.no_grad():
                    if idx == ():
                        p.data.fill_(v)
                    else:
                        p.data[idx] = v

            orig = get_val()
            set_val(orig + eps)
            lp = loss_val()
            set_val(orig - eps)
            lm_ = loss_val()
            set_val(orig)
            fd = (lp - lm_) / (2 * eps)
            if max(abs(autog), abs(fd)) < 1e-5:
                continue  # NEGLIGIBLE
            rel = abs(fd - autog) / max(abs(autog), abs(fd), 1e-6)
            assert rel < 3e-2, (f'D(3) fd {name}{idx}: autograd={autog:.6e} fd={fd:.6e} '
                               f'rel={rel:.3e}')


class TestStageDStructural:
    """Stage D with a random-initialised head: loss+grad *contract* (no checkpoint).

    Same three-step contract as the original ``stage_d()`` — loss value ==
    re-derived formula, autograd == re-derived, autograd == finite-difference —
    but with a random ``ClefJointSchemaHead`` of identical structure, so the
    loss/grad contract is exercised on CPU without any model download.
    """

    def test_loss_value_and_grad(self):
        _need_swift()
        rec = next(r for r in RECORDS if r['id'] == 'r_joint')
        gold = GOLD_IDX['r_joint']
        dev = 'cpu'
        head = _make_random_head()
        for p in head.parameters():
            p.requires_grad_(True)
        # mock hidden states + emb weight + meta (same shapes the head expects)
        nq = 3
        n_options = [2, 3, 5]
        max_opt = max(n_options)
        seq_len = 12
        hidden = torch.randn(1, seq_len, _RAND_HIDDEN, generator=torch.Generator().manual_seed(3))
        emb_w = torch.randn(_RAND_VOCAB, _RAND_HIDDEN, generator=torch.Generator().manual_seed(7))
        head.set_output_embeddings(_EmbShim(emb_w))
        meta = _make_mock_meta(nq=nq, n_options=n_options, seq_len=seq_len)
        target = torch.zeros((nq, max_opt), dtype=torch.float32)
        for i, g in enumerate(gold):
            target[i, g] = 1.0

        # (1) value
        clef_loss = ClefLoss(None, _DummyTrainer())
        out = head(hidden, None, meta)
        our_val = float(clef_loss(outputs=out, labels=target, num_items_in_batch=None))
        re_val = float(rederived_clef_loss(out.logits, out.option_mask, target))
        assert abs(our_val - re_val) < 1e-5, (
            f'D(1) loss value our={our_val:.8f} rederived={re_val:.8f} '
            f'|diff|={abs(our_val - re_val):.3e}')

        # (2) autograd our-vs-rederived
        named = [(n, p) for n, p in head.named_parameters() if p.requires_grad]
        head.zero_grad(set_to_none=True)
        clef_loss(outputs=head(hidden, None, meta), labels=target,
                  num_items_in_batch=None).backward()
        g_our = {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in named}
        head.zero_grad(set_to_none=True)
        o2 = head(hidden, None, meta)
        rederived_clef_loss(o2.logits, o2.option_mask, target).backward()
        g_re = {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in named}

        def gnorm(gd):
            return float(torch.sqrt(sum((g.double() ** 2).sum() for g in gd.values() if g is not None)))

        n_our, n_re = gnorm(g_our), gnorm(g_re)
        norm_rel = abs(n_our - n_re) / max(n_our, n_re, 1e-12)
        worst, worst_name = 0.0, None
        for n, _ in named:
            a, b = g_our[n], g_re[n]
            if a is None or b is None or float(b.abs().max()) < 1e-4:
                continue
            rel = float((a - b).abs().max()) / float(b.abs().max())
            if rel > worst:
                worst, worst_name = rel, n
        assert norm_rel < 1e-4 and worst < 1e-4, (
            f'D(2) autograd our-vs-rederived grad_norm rel={norm_rel:.3e}; '
            f'worst meaningful-param rel={worst:.3e} @ {worst_name}')

        # (3) finite-difference
        eps = 1e-3
        for name, idx in [('prior_logit_scale', ()), ('joint_logit_scale', ()), ('residual_gate', ()),
                          ('residual_scorer.0.weight', (0, 0))]:
            p = dict(head.named_parameters())[name]
            autog = float(g_re[name]) if idx == () else float(g_re[name][idx])

            def loss_val():
                with torch.no_grad():
                    o = head(hidden, None, meta)
                    return float(rederived_clef_loss(o.logits, o.option_mask, target))

            def get_val():
                return float(p.data) if idx == () else float(p.data[idx])

            def set_val(v):
                with torch.no_grad():
                    if idx == ():
                        p.data.fill_(v)
                    else:
                        p.data[idx] = v

            orig = get_val()
            set_val(orig + eps)
            lp = loss_val()
            set_val(orig - eps)
            lm_ = loss_val()
            set_val(orig)
            fd = (lp - lm_) / (2 * eps)
            if max(abs(autog), abs(fd)) < 1e-5:
                continue  # NEGLIGIBLE
            rel = abs(fd - autog) / max(abs(autog), abs(fd), 1e-6)
            assert rel < 3e-2, (f'D(3) fd {name}{idx}: autograd={autog:.6e} fd={fd:.6e} '
                               f'rel={rel:.3e}')


# ===========================================================================
# Legacy CLI entrypoint (preserved for backward compatibility)
# ===========================================================================
def _cli_main():
    """Run the alignment suite the old way (all stages, needs checkpoint)."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all', choices=['a', 'bc', 'd', 'all'])
    args = ap.parse_args()
    import traceback

    if _get_clef_dir() is None:
        print('ERROR: Clef checkpoint not found. Set CLEF_MODEL_DIR.', file=sys.stderr)
        return 2
    if not _swift_ok:
        print(f'ERROR: swift imports failed: {_swift_err}', file=sys.stderr)
        return 2

    results = {}
    try:
        if args.stage in ('a', 'all'):
            results['A'] = _legacy_stage_a()
        if args.stage in ('bc', 'all'):
            results['BC'] = _legacy_stage_bc()
        if args.stage in ('d', 'all'):
            results['D'] = _legacy_stage_d()
    except Exception:
        print('\n=== HARNESS EXCEPTION ===', flush=True)
        traceback.print_exc()
        return 2
    print('STAGES:', {k: ('PASS' if v else 'FAIL') for k, v in results.items()}, flush=True)
    ok = all(results.values())
    print('ALL PASS' if ok else 'SOME FAILED', flush=True)
    return 0 if ok else 1


def _legacy_stage_a():
    """Legacy Stage A runner — verbatim original logic."""
    ref = _get_ref()
    from swift.model import get_processor
    from swift.template import get_template
    processor = get_processor(_get_clef_dir(), model_type='clef', task_type='decision')
    template = get_template(processor, template_type='clef')
    template.set_mode('train')
    tok = processor.tokenizer
    ok_all = True
    for rec in RECORDS:
        norm = ClefPreprocessor().preprocess(dict(rec))
        enc = template._encode(StdTemplateInputs.from_dict(norm))
        meta = enc['decision_meta']
        o_ids = list(enc['input_ids'])
        o_q = [tuple(x) for x in meta['question_spans']]
        o_o = [[tuple(s) for s in q] for q in meta['option_spans']]
        o_k = list(meta['kinds'])

        e = ref.encode_record(tok, {'model': 'clef', **rec}, processor=processor)
        r_ids = list(e.input_ids)
        r_q = [tuple(q.question_span) for q in e.questions]
        r_o = [[tuple(s) for s in q.option_spans] for q in e.questions]
        r_k = [q.question_type for q in e.questions]

        ok = (o_ids == r_ids) and (o_q == r_q) and (o_o == r_o) and (o_k == r_k)
        ok_all = ok_all and ok
        print(f'[{"PASS" if ok else "FAIL"}] A {rec["id"]}: ids={o_ids == r_ids} qspans={o_q == r_q} '
              f'ospans={o_o == r_o} kinds={o_k == r_k} (len {len(o_ids)} vs {len(r_ids)})')
        if not ok and o_ids != r_ids:
            n = min(len(o_ids), len(r_ids))
            div = next((i for i in range(n) if o_ids[i] != r_ids[i]), n)
            lo = max(0, div - 8)
            print(f'   first id divergence at {div}: ours={tok.decode(o_ids[lo:div + 8])!r} '
                  f'ref={tok.decode(r_ids[lo:div + 8])!r}')
    return ok_all


def _legacy_stage_bc(dev_ref='cuda:0', dev_ours='cuda:1'):
    """Legacy Stage BC runner — verbatim original logic."""
    if not torch.cuda.is_available():
        print('SKIP BC: no CUDA', flush=True)
        return True
    ref = _get_ref()
    from swift.model import get_model_processor, get_processor
    clef_dir = _get_clef_dir()
    print('loading reference ClefModel (bf16, eager) ...', flush=True)
    ref_model, processor = ref.load_release_model(
        clef_dir, device=dev_ref, dtype=torch.bfloat16, attn_implementation='eager')
    ref_model.eval()
    tok, pad_id = processor.tokenizer, processor.tokenizer.pad_token_id
    print('loading OUR swift clef model (bf16, eager) ...', flush=True)
    our_model, _ = get_model_processor(
        _get_clef_dir(), model_type='clef', task_type='decision', torch_dtype=torch.bfloat16,
        device_map=dev_ours, use_hf=True, attn_impl='eager')
    our_model.eval()

    ref_head32 = copy.deepcopy(ref_model.head).float().to(dev_ref)
    our_head32 = copy.deepcopy(our_model.scoring_head).float().to(dev_ref)

    base_model = ref_model.language_model
    text_model = base_model.model
    if hasattr(text_model, 'language_model'):
        text_model = text_model.language_model

    ok_all = True
    for rec in RECORDS:
        encoded = ref.encode_record(tok, {'model': 'clef', **rec}, processor=processor)
        nq = len(encoded.questions)
        n_options = [len(q.option_ids) for q in encoded.questions]
        ids_ref = torch.tensor([encoded.input_ids], dtype=torch.long, device=dev_ref)
        am_ref = torch.ones_like(ids_ref)
        batch_ref = ref.collate_records([encoded], pad_id, torch.device(dev_ref))
        with torch.inference_mode():
            last_hidden = text_model(
                input_ids=ids_ref, attention_mask=am_ref, use_cache=False,
                return_dict=True).last_hidden_state
            emb_w = base_model.get_output_embeddings().weight
            ref_logits = ref_model(batch_ref)[0]

        # Stage B: head-only fp32 on the SAME last_hidden.
        meta_ref = build_meta(encoded, dev_ref, ids_ref)
        h32, emb32 = last_hidden.float(), emb_w.float()
        our_head32.set_output_embeddings(_EmbShim(emb32))
        with torch.inference_mode():
            ref_head_logits = ref_head32(h32, ids_ref, am_ref, [encoded], emb32)[0]
            our32 = our_head32(h32, None, meta_ref)
        b_max = max((maxdiff(ref_head_logits[i].float().cpu(),
                              our32.logits[i, :n_options[i]].float().cpu())
                     for i in range(nq)), default=0.0)
        b_pass = b_max < 2e-4

        # Stage C: full forward, our model bf16 eager.
        ids_o, am_o = ids_ref.to(dev_ours), am_ref.to(dev_ours)
        meta_o = build_meta(encoded, dev_ours, ids_o)
        with torch.inference_mode():
            our_out = our_model(input_ids=ids_o, attention_mask=am_o, decision_meta=meta_o)
        c_l, c_p = [], []
        for i in range(nq):
            rl = ref_logits[i].detach().float().cpu()
            ol = our_out.logits[i, :n_options[i]].detach().float().cpu()
            c_l.append(maxdiff(rl, ol))
            op = masked_softmax(our_out.logits[i:i + 1].float(),
                                 our_out.option_mask[i:i + 1])[0, :n_options[i]].cpu()
            c_p.append(maxdiff(torch.softmax(rl, dim=-1), op))
        c_lmax, c_pmax = max(c_l, default=0.0), max(c_p, default=0.0)
        c_pass = c_lmax < 0.25 and c_pmax < 2e-2

        ok = b_pass and c_pass
        ok_all = ok_all and ok
        print(f'[{"PASS" if ok else "FAIL"}] BC {rec["id"]} nq={nq} nopt={n_options} L={len(encoded.input_ids)}')
        print(f'   B head-only fp32 max|logit diff|={b_max:.3e} ({"PASS" if b_pass else "FAIL"} <2e-4)')
        print(f'   C full bf16 max|logit diff|={c_lmax:.3e} max|prob diff|={c_pmax:.3e} '
              f'({"PASS" if c_pass else "FAIL"})')
    return ok_all


def _legacy_stage_d(dev='cuda:0'):
    """Legacy Stage D runner — verbatim original logic."""
    if not torch.cuda.is_available():
        print('SKIP D: no CUDA', flush=True)
        return True
    ref = _get_ref()
    from swift.model import get_model_processor, get_processor
    rec = next(r for r in RECORDS if r['id'] == 'r_joint')
    gold = GOLD_IDX['r_joint']
    print('loading OUR swift clef model (bf16, eager) ...', flush=True)
    our_model, processor = get_model_processor(
        _get_clef_dir(), model_type='clef', task_type='decision', torch_dtype=torch.bfloat16,
        device_map=dev, use_hf=True, attn_impl='eager')
    our_model.eval()
    tok = processor.tokenizer
    encoded = ref.encode_record(tok, {'model': 'clef', **rec}, processor=processor)
    nq = len(encoded.questions)
    n_options = [len(q.option_ids) for q in encoded.questions]
    max_opt = max(n_options)
    ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=dev)
    am = torch.ones_like(ids)
    meta = build_meta(encoded, dev, ids)
    target = torch.zeros((nq, max_opt), dtype=torch.float32, device=dev)
    for i, g in enumerate(gold):
        target[i, g] = 1.0

    text_model = our_model.model
    if hasattr(text_model, 'language_model'):
        text_model = text_model.language_model
    with torch.no_grad():
        hidden = text_model(input_ids=ids, attention_mask=am, use_cache=False,
                            return_dict=True).last_hidden_state.detach().float().clone()
    emb32 = our_model.get_output_embeddings().weight.detach().float().clone()
    head32 = copy.deepcopy(our_model.scoring_head).float().to(dev)
    head32.set_output_embeddings(_EmbShim(emb32))
    for p in head32.parameters():
        p.requires_grad_(True)

    # (1) value
    clef_loss = ClefLoss(None, _DummyTrainer())
    out32 = head32(hidden, None, meta)
    our_val = float(clef_loss(outputs=out32, labels=target, num_items_in_batch=None))
    re_val = float(rederived_clef_loss(out32.logits, out32.option_mask, target))
    v_pass = abs(our_val - re_val) < 1e-5
    print(f'[{"PASS" if v_pass else "FAIL"}] D(1) loss value our={our_val:.8f} rederived={re_val:.8f} '
          f'|diff|={abs(our_val - re_val):.3e}')

    # (2) autograd our-vs-rederived
    named = [(n, p) for n, p in head32.named_parameters() if p.requires_grad]
    head32.zero_grad(set_to_none=True)
    clef_loss(outputs=head32(hidden, None, meta), labels=target,
              num_items_in_batch=None).backward()
    g_our = {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in named}
    head32.zero_grad(set_to_none=True)
    o2 = head32(hidden, None, meta)
    rederived_clef_loss(o2.logits, o2.option_mask, target).backward()
    g_re = {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in named}

    def gnorm(gd):
        return float(torch.sqrt(sum((g.double() ** 2).sum() for g in gd.values() if g is not None)))

    n_our, n_re = gnorm(g_our), gnorm(g_re)
    norm_rel = abs(n_our - n_re) / max(n_our, n_re, 1e-12)
    worst, worst_name = 0.0, None
    for n, _ in named:
        a, b = g_our[n], g_re[n]
        if a is None or b is None or float(b.abs().max()) < 1e-4:
            continue
        rel = float((a - b).abs().max()) / float(b.abs().max())
        if rel > worst:
            worst, worst_name = rel, n
    g_pass = norm_rel < 1e-4 and worst < 1e-4
    print(f'[{"PASS" if g_pass else "FAIL"}] D(2) autograd our-vs-rederived grad_norm rel={norm_rel:.3e}; '
          f'worst meaningful-param rel={worst:.3e} @ {worst_name}')

    # (3) finite-difference
    eps = 1e-3
    fd_pass = True
    for name, idx in [('prior_logit_scale', ()), ('joint_logit_scale', ()), ('residual_gate', ()),
                      ('residual_scorer.0.weight', (0, 0))]:
        p = dict(head32.named_parameters())[name]
        autog = float(g_re[name]) if idx == () else float(g_re[name][idx])

        def loss_val():
            with torch.no_grad():
                o = head32(hidden, None, meta)
                return float(rederived_clef_loss(o.logits, o.option_mask, target))

        def get_val():
            return float(p.data) if idx == () else float(p.data[idx])

        def set_val(v):
            with torch.no_grad():
                if idx == ():
                    p.data.fill_(v)
                else:
                    p.data[idx] = v

        orig = get_val()
        set_val(orig + eps)
        lp = loss_val()
        set_val(orig - eps)
        lm = loss_val()
        set_val(orig)
        fd = (lp - lm) / (2 * eps)
        if max(abs(autog), abs(fd)) < 1e-5:
            print(f'   [NEGLIGIBLE] D(3) fd {name}{idx}: autograd={autog:.3e} fd={fd:.3e} (both ~0)')
            continue
        rel = abs(fd - autog) / max(abs(autog), abs(fd), 1e-6)
        ok = rel < 3e-2
        fd_pass = fd_pass and ok
        print(f'   [{"PASS" if ok else "FAIL"}] D(3) fd {name}{idx}: autograd={autog:.6e} fd={fd:.6e} rel={rel:.3e}')
    return v_pass and g_pass and fd_pass


if __name__ == '__main__':

