# Copyright (c) ModelScope Contributors. All rights reserved.
"""JEV (`decision` task_type) alignment against the official `serve_decide.py::s1_pass` oracle.

Runs the real integration path (official autotrust/JEV-27B-VL base + its `adapter_vllm` LoRA + our
JevPreprocessor / JevTemplate / JevModelLoader / JevVerbalizerHead / JevDistillLoss) and compares it
to the reference System-1 readout, in four stages:

  A  encoding   -- our JevTemplate._encode input_ids are BYTE-identical to the oracle s1_pass prompt
                   tokenized with add_special_tokens=False, the real collator flattens a fan-out batch
                   into the flat decision_meta, and noul/score are forced to the canonical verbalizer
                   options even when a record supplies custom text (tokenizer only; cheap).
  B  head-only  -- feed the reference backbone's last-token lm_logits (fp32) into BOTH the re-derived
                   oracle s1_pass math and our JevVerbalizerHead: probs must match to fp32 precision,
                   including a right- and a left-padded multi-question batch (isolates the head port +
                   `_readout_positions`).
  C  full fwd   -- our swift-loaded model (base + `adapter_vllm` via Swift.from_pretrained) end-to-end
                   vs the reference (plain transformers + peft) s1_pass readout on the same encoding
                   (bf16, eager both sides): validates the whole patch + adapter + head path.
  D  loss+grad  -- JevDistillLoss is validated three ways: value == an independently re-derived
                   KL(target||model) + 0.5*RPS(score); its autograd grads on the head bias == the
                   re-derived formula's grads; autograd == central finite-difference.

`serve_decide.py` needs a vLLM engine and cannot be imported here, so the oracle math is re-derived in
plain torch against the SAME base+LoRA weights (de-risked by `.align_scratch/probe_jev_ref.py`, which
reproduced the noul readout). Both model sides force attn_implementation='eager' (control group:
removes sdpa/flash kernel divergence so any mismatch is attributable to logic, not kernels).

Usage:
    python tests/test_align/test_decision_jev_align.py --stage a      # cheap, tokenizer only
    python tests/test_align/test_decision_jev_align.py --stage all    # loads the 27B checkpoint
"""
import argparse
import json
import os
import string
import sys
import traceback

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('USE_HF', '1')

import torch
from huggingface_hub import snapshot_download

JEV_ID = 'autotrust/JEV-27B-VL'
SNAP = snapshot_download(JEV_ID)
ADAPTER = os.path.join(SNAP, 'adapter_vllm')

from swift.dataset.preprocessor.decision import JevPreprocessor
from swift.loss.decision import JevDistillLoss
from swift.model import get_model_processor, get_processor
from swift.model.decision_head import QUESTION_TYPES, JevVerbalizerHead, masked_softmax
from swift.template import get_template
from swift.template.template_inputs import StdTemplateInputs
from swift.tuners import Swift

LETTERS = list(string.ascii_uppercase)
_HEAD = json.load(open(os.path.join(ADAPTER, 'decision_head.json')))
BIAS = _HEAD['bias']
VIDS = _HEAD['verbalizer_ids']
RANGES = {k: tuple(v) for k, v in _HEAD['slots']['ranges'].items()}
TEMPS = json.load(open(os.path.join(SNAP, 'calibration.json')))['per_kind']
KIND_OF = {'noul': 0, 'choice': 1, 'score': 2}

# Faithful records: noul/score OMIT options (canonical forced); choice supplies its own. `gold` is the
# teacher soft distribution (score) or an option index, matching the distill-corpus contract.
RECORDS = [
    {'id': 'noul', 'state': 'The sky is blue.',
     'questions': [{'kind': 'noul', 'question': 'Is the sky blue?', 'gold': 1}]},
    {'id': 'choice3', 'state': 'A review: late and damaged.',
     'questions': [{'kind': 'choice', 'question': 'Sentiment?',
                    'options': ['positive', 'negative', 'neutral'], 'gold': 1}]},
    {'id': 'choice4', 'state': 'Qwen is a hybrid model.',
     'questions': [{'kind': 'choice', 'question': 'Who built Qwen?',
                    'options': ['Alibaba', 'Google', 'Meta', 'OpenAI'], 'gold': 0}]},
    {'id': 'score', 'state': 'some text to rate',
     'questions': [{'kind': 'score', 'question': 'Rate quality.',
                    'gold': [0.05, 0.1, 0.15, 0.4, 0.2, 0.1]}]},
    {'id': 'joint', 'state': 'A joint record, one question per forward.',
     'questions': [{'kind': 'noul', 'question': 'Is this spam?', 'gold': 0},
                   {'kind': 'choice', 'question': 'Pick a topic.',
                    'options': ['Technology', 'Sports', 'Food'], 'gold': 1},
                   {'kind': 'score', 'question': 'Politeness level.', 'gold': 4}]},
]


# --------------------------- oracle s1_pass re-derivation ---------------------------
def ref_s1_text(kind, state, question, opts, prefix=''):
    """Verbatim serve_decide.py::s1_pass content assembly (no-image path), joined into one string.

    `decide()` forces opts=['false','true'] for noul and ['0'..'5'] for score; only choice renders the
    caller's options, prefixed with the A-P position letter (`f'{LETTERS[i]}) {o}'`).
    """
    lines = [f'{LETTERS[i]}) {o}' for i, o in enumerate(opts)] if kind == 'choice' else list(opts)
    return (f'{prefix}[kind] {kind}\n[state] ' + state + f'\n[question] {question}\n[options]\n' +
            '\n'.join(lines) + '\n[decision]:')


def ref_s1_probs(lm_last, kind, n):
    """serve_decide.py::s1_pass readout on ONE token's full-vocab logits `lm_last` [vocab] (fp32).

    log_softmax over the full vocab -> gather the kind's first `n` verbalizer ids -> +bias -> /temp ->
    softmax over just those n options (the oracle `_softmax`). Returns [n] fp32.
    """
    lo, _ = RANGES[kind]
    ids = torch.as_tensor(VIDS[lo:lo + n], device=lm_last.device, dtype=torch.long)
    b = torch.as_tensor(BIAS[lo:lo + n], device=lm_last.device, dtype=torch.float32)
    lp = torch.log_softmax(lm_last.float(), dim=-1)
    z = (lp.gather(0, ids) + b) / float(TEMPS[kind])
    return torch.softmax(z, dim=-1)


def flat_rows():
    """Fan every record out through the real JevPreprocessor -> [(rec_id, kind, row, opts)]."""
    out = []
    for rec in RECORDS:
        raw = {k: v for k, v in rec.items() if k != 'id'}
        rows = JevPreprocessor().preprocess(raw)
        assert len(rows) == len(rec['questions']), f'fan-out mismatch for {rec["id"]}'
        for row in rows:
            out.append((rec['id'], row['kinds'][0], row, row['options'][0]))
    return out


def meta1(kind_int, n, seq_len=1):
    return {
        'kinds': [kind_int], 'n_options': [n], 'record_index': [0], 'counts': [1],
        'max_opt': n, 'left_padding': False, 'seq_lens': [seq_len],
    }


def maxdiff(a, b):
    a = torch.as_tensor(a, dtype=torch.float64).flatten()
    b = torch.as_tensor(b, dtype=torch.float64).flatten()
    return float((a - b).abs().max()) if a.numel() else 0.0


# ------------------------------- lazy model singletons -------------------------------
_PROC = None
_REF = None
_OURS = None


def processor():
    global _PROC
    if _PROC is None:
        _PROC = get_processor(JEV_ID, model_type='jev', task_type='decision')
    return _PROC


def ref_model(dev='cuda:0'):
    """Reference side: plain transformers base + peft `adapter_vllm` (bf16, eager)."""
    global _REF
    if _REF is None:
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText
        print('loading REFERENCE base + adapter_vllm (plain transformers + peft, bf16, eager) ...', flush=True)
        base = AutoModelForImageTextToText.from_pretrained(
            SNAP, dtype=torch.bfloat16, attn_implementation='eager', trust_remote_code=True).to(dev).eval()
        _REF = (PeftModel.from_pretrained(base, ADAPTER).eval(), dev)
        n_lora = sum(1 for n, _ in _REF[0].named_parameters() if 'lora_' in n)
        print(f'   reference ready: {type(base).__name__} + {n_lora} LoRA params', flush=True)
    return _REF


def our_model(dev='cuda:1'):
    """Swift side: get_model_processor base + head, then Swift.from_pretrained(adapter_vllm) -- the
    exact续训 path (`tuner.py::prepare_model` adapter branch)."""
    global _OURS
    if _OURS is None:
        print('loading OUR swift model + head, then Swift.from_pretrained(adapter_vllm) ...', flush=True)
        model, proc = get_model_processor(
            SNAP, model_type='jev', task_type='decision', torch_dtype=torch.bfloat16, device_map=dev,
            use_hf=True, attn_impl='eager')
        assert hasattr(model, 'scoring_head'), 'FAIL: loader did not attach the head'
        model = Swift.from_pretrained(model, ADAPTER, is_trainable=True)
        base = getattr(model.base_model, 'model', None)
        assert getattr(base, '_patched', False), 'FAIL: forward patch lost after the PeftModel wrap'
        model.eval()
        _OURS = (model, proc, dev)
        print('   swift model ready (patch survives the wrap)', flush=True)
    return _OURS


def ref_last_logits(rows_texts, dev):
    """Per-prompt last-token full-vocab logits from the reference model -> list of [vocab] fp32."""
    model, mdev = ref_model(dev)
    tok = processor().tokenizer
    out = []
    for text in rows_texts:
        enc = tok(text, return_tensors='pt', add_special_tokens=False)
        with torch.inference_mode():
            o = model(input_ids=enc['input_ids'].to(mdev),
                      attention_mask=enc['attention_mask'].to(mdev), return_dict=True)
        out.append(o.logits[0, -1, :].float().to(dev))
    return out


# ------------------------------- Stage A: encoding -------------------------------
def stage_a():
    proc = processor()
    template = get_template(proc, template_type='jev')
    template.set_mode('train')
    tok = proc.tokenizer
    ok_all = True

    print('=== A6: A-P single-token ids == verbalizer_ids[8:24] (oracle setup() assert) ===', flush=True)
    letter_ids = [tok.encode(lab, add_special_tokens=False)[0] for lab in LETTERS[:16]]
    a6 = letter_ids == VIDS[8:24]
    ok_all = ok_all and a6
    print(f'[{"PASS" if a6 else "FAIL"}] A6 letter ids {letter_ids} == verbalizer choice slots', flush=True)

    print('=== A1-A5: faithful records byte-identical to the oracle s1_pass prompt ===', flush=True)
    rows = flat_rows()
    for rid, kind, row, opts in rows:
        enc = template._encode(StdTemplateInputs.from_dict(row))
        ours = list(enc['input_ids'])
        ref_text = ref_s1_text(kind, row['messages'][0]['content'], row['questions'][0], opts)
        ref = tok.encode(ref_text, add_special_tokens=False)
        meta = enc['decision_meta']
        ids_ok = ours == ref
        meta_ok = meta['kinds'] == [KIND_OF[kind]] and meta['n_options'] == [len(opts)]
        ok = ids_ok and meta_ok
        ok_all = ok_all and ok
        print(f'[{"PASS" if ok else "FAIL"}] A {rid}/{kind} nopt={len(opts)} ids={ids_ok} meta={meta_ok} '
              f'(len {len(ours)} vs {len(ref)})', flush=True)
        if not ids_ok:
            n = min(len(ours), len(ref))
            div = next((i for i in range(n) if ours[i] != ref[i]), n)
            lo = max(0, div - 6)
            print(f'   first divergence at {div}: ours={tok.decode(ours[lo:div + 6])!r} '
                  f'ref={tok.decode(ref[lo:div + 6])!r}', flush=True)

    print('=== A7/A8: real collator -> flat decision_meta + padded labels (train right / infer left) ===',
          flush=True)
    encoded = [template.encode(dict(row)) for _, _, row, _ in rows]
    collated = template.data_collator(encoded)
    dm = collated['decision_meta']
    total_q = len(rows)
    a7 = (len(dm['record_index']) == total_q and dm['max_opt'] == max(len(o) for *_, o in rows)
          and collated['input_ids'].shape[0] == total_q and 'labels' in collated
          and tuple(collated['labels'].shape) == (total_q, dm['max_opt'])
          and all(0 <= ri < total_q for ri in dm['record_index'])
          and all(sl <= collated['input_ids'].shape[1] for sl in dm['seq_lens'])
          and dm['left_padding'] is False)
    ok_all = ok_all and a7
    print(f'[{"PASS" if a7 else "FAIL"}] A7 train collator record_index={dm["record_index"]} '
          f'counts={dm["counts"]} n_options={dm["n_options"]} kinds={dm["kinds"]} max_opt={dm["max_opt"]} '
          f'left_padding={dm["left_padding"]} labels={tuple(collated["labels"].shape)}', flush=True)

    template.set_mode('transformers')
    coll_inf = template.data_collator([template.encode(dict(row)) for _, _, row, _ in rows])
    a8 = coll_inf['decision_meta']['left_padding'] is True
    ok_all = ok_all and a8
    print(f'[{"PASS" if a8 else "FAIL"}] A8 inference collator left_padding='
          f'{coll_inf["decision_meta"]["left_padding"]}', flush=True)
    template.set_mode('train')

    print('=== A9: DIVERGENCE guard -- custom noul/score options are FORCED to canonical ===', flush=True)
    a9 = True
    for kind, custom, gold in (('score', ['bad', 'ok', 'good'], 1), ('noul', ['no', 'yes'], 0)):
        bad = {'state': 'x', 'questions': [{'kind': kind, 'question': 'Q?', 'options': custom, 'gold': gold}]}
        brow = JevPreprocessor().preprocess(bad)[0]
        canonical = JevPreprocessor.default_options(kind)
        forced = brow['options'][0] == canonical
        benc = template._encode(StdTemplateInputs.from_dict(brow))
        ref = tok.encode(ref_s1_text(kind, 'x', 'Q?', canonical), add_special_tokens=False)
        byte_ok = list(benc['input_ids']) == ref
        a9 = a9 and forced and byte_ok
        print(f'[{"PASS" if forced and byte_ok else "FAIL"}] A9 {kind} custom={custom} -> forced='
              f'{brow["options"][0]} (canonical={canonical}) byte_ok={byte_ok}', flush=True)
    ok_all = ok_all and a9
    return ok_all


# --------------------------- Stage B: head-only fp32 ---------------------------
def stage_b(dev='cuda:0'):
    proc = processor()
    template = get_template(proc, template_type='jev')
    template.set_mode('train')
    tok = proc.tokenizer
    head = JevVerbalizerHead.from_pretrained(SNAP).float().to(dev)
    rows = flat_rows()
    texts = [ref_s1_text(kind, row['messages'][0]['content'], row['questions'][0], opts)
             for _, kind, row, opts in rows]
    last = ref_last_logits(texts, dev)  # one [vocab] per question-row

    ok_all = True
    print('=== B1-B3: per-kind batch=1, our head vs oracle s1_pass math (fp32) ===', flush=True)
    for i, (rid, kind, row, opts) in enumerate(rows):
        n = len(opts)
        lm = last[i].view(1, 1, -1)  # [1,1,vocab]; head reads col 0 (seq_len 1)
        with torch.inference_mode():
            out = head(None, lm, meta1(KIND_OF[kind], n))
        our = masked_softmax(out.logits.float(), out.option_mask)[0, :n]
        ref = ref_s1_probs(last[i], kind, n)
        d = maxdiff(our.cpu(), ref.cpu())
        ok = d < 1e-5
        ok_all = ok_all and ok
        print(f'[{"PASS" if ok else "FAIL"}] B {rid}/{kind} n={n} max|prob diff|={d:.3e} '
              f'our={[round(float(x), 5) for x in our]}', flush=True)

    print('=== B4: right- AND left-padded 3-question batch -> _readout_positions ===', flush=True)
    idx = [0, 1, 3]  # noul(2), choice3(3), score(6) -- distinct widths
    kinds_i = [KIND_OF[rows[j][1]] for j in idx]
    nopt_i = [len(rows[j][3]) for j in idx]
    seq_lens = [3, 2, 4]
    T = max(seq_lens)
    vocab = last[0].shape[0]
    for left in (False, True):
        lm = torch.zeros((len(idx), T, vocab), device=dev, dtype=torch.float32)
        for r, j in enumerate(idx):
            col = T - 1 if left else seq_lens[r] - 1
            lm[r, col, :] = last[j]
        meta = {
            'kinds': kinds_i, 'n_options': nopt_i, 'record_index': list(range(len(idx))),
            'counts': [1] * len(idx), 'max_opt': max(nopt_i), 'left_padding': left, 'seq_lens': seq_lens,
        }
        with torch.inference_mode():
            out = head(None, lm, meta)
        worst = 0.0
        for r, j in enumerate(idx):
            n = nopt_i[r]
            our = masked_softmax(out.logits.float(), out.option_mask)[r, :n]
            worst = max(worst, maxdiff(our.cpu(), ref_s1_probs(last[j], rows[j][1], n).cpu()))
        ok = worst < 1e-5
        ok_all = ok_all and ok
        print(f'[{"PASS" if ok else "FAIL"}] B4 {"left" if left else "right"}-pad seq_lens={seq_lens} '
              f'worst max|prob diff|={worst:.3e}', flush=True)
    return ok_all


# --------------------------- Stage C: full forward bf16 ---------------------------
def stage_c(dev_ref='cuda:0', dev_ours='cuda:1'):
    proc = processor()
    template = get_template(proc, template_type='jev')
    rows = flat_rows()
    texts = [ref_s1_text(kind, row['messages'][0]['content'], row['questions'][0], opts)
             for _, kind, row, opts in rows]
    last = ref_last_logits(texts, dev_ref)  # reference readout per question
    model, sproc, mdev = our_model(dev_ours)
    stok = sproc.tokenizer if hasattr(sproc, 'tokenizer') else sproc

    ok_all = True
    print('=== C1-C3: batch=1 per record, swift full forward vs reference s1_pass (bf16, eager) ===',
          flush=True)
    template.set_mode('train')
    row_i = 0
    for rid, kind, row, opts in rows:
        n = len(opts)
        enc = template.encode(dict(row))
        coll = template.data_collator([enc])
        feed = {k: (v.to(mdev) if torch.is_tensor(v) else v)
                for k, v in coll.items() if k in ('input_ids', 'attention_mask', 'decision_meta')}
        with torch.inference_mode():
            out = model(**feed)
        our = masked_softmax(out.logits.float(), out.option_mask)[0, :n].cpu()
        ref = ref_s1_probs(last[row_i], kind, n).cpu()
        d = maxdiff(our, ref)
        ok = d < 2e-2
        ok_all = ok_all and ok
        print(f'[{"PASS" if ok else "FAIL"}] C {rid}/{kind} n={n} max|prob diff|={d:.3e} '
              f'our={[round(float(x), 4) for x in our]}', flush=True)
        row_i += 1

    print('=== C4: multi-row TRAIN batch (right-pad) vs per-row reference (padding invariance) ===',
          flush=True)
    template.set_mode('train')
    encoded = [template.encode(dict(row)) for _, _, row, _ in rows]
    coll = template.data_collator(encoded)
    dm = coll['decision_meta']
    feed = {k: (v.to(mdev) if torch.is_tensor(v) else v)
            for k, v in coll.items() if k in ('input_ids', 'attention_mask', 'decision_meta')}
    with torch.inference_mode():
        out = model(**feed)
    worst = 0.0
    for q, (rid, kind, row, opts) in enumerate(rows):
        n = len(opts)
        our = masked_softmax(out.logits.float(), out.option_mask)[q, :n].cpu()
        worst = max(worst, maxdiff(our, ref_s1_probs(last[q], kind, n).cpu()))
    ok = worst < 2e-2 and dm['left_padding'] is False
    ok_all = ok_all and ok
    print(f'[{"PASS" if ok else "FAIL"}] C4 right-pad batch N={len(rows)} T={coll["input_ids"].shape[1]} '
          f'worst max|prob diff|={worst:.3e}', flush=True)
    return ok_all


# ------------------------------ Stage D: loss + grad ------------------------------
def rederived_jev_loss(logits, option_mask, kinds, target, rps_w=0.5):
    """Independent re-derivation of JevDistillLoss: masked soft-CE (== KL up to a constant) over each
    question's own options + 0.5 * mean_over_questions(RPS * 1{score})."""
    probs = masked_softmax(logits, option_mask)
    logp = torch.log(probs.clamp_min(1e-9))
    per_q = -(target * logp * option_mask).sum(-1)
    total_q = per_q.shape[0]
    ce = per_q.sum() / total_q if total_q else per_q.sum()
    score_mask = kinds == QUESTION_TYPES['score']
    term = torch.zeros((), device=logits.device, dtype=torch.float64)
    if bool(score_mask.any()):
        cum_m = probs.cumsum(-1)[:, :-1]
        cum_t = target.cumsum(-1)[:, :-1]
        gap2 = (cum_m - cum_t).pow(2)
        denom = (option_mask.sum(-1) - 1).clamp(min=1).to(gap2.dtype)
        per_row = gap2.sum(-1) / denom
        term = rps_w * (per_row[score_mask].sum() / total_q if total_q else per_row[score_mask].sum())
    return ce + term


class _DummyTrainer:
    pass


def stage_d(dev='cuda:0'):
    proc = processor()
    template = get_template(proc, template_type='jev')
    template.set_mode('train')
    rec = next(r for r in RECORDS if r['id'] == 'joint')
    jrows = JevPreprocessor().preprocess({k: v for k, v in rec.items() if k != 'id'})
    texts = [ref_s1_text(r['kinds'][0], r['messages'][0]['content'], r['questions'][0], r['options'][0])
             for r in jrows]
    last = ref_last_logits(texts, dev)  # 3 rows: noul / choice3 / score6

    nq = len(jrows)
    nopt = [len(r['options'][0]) for r in jrows]
    max_opt = max(nopt)
    vocab = last[0].shape[0]
    lm = torch.zeros((nq, 1, vocab), device=dev, dtype=torch.float32)
    for i in range(nq):
        lm[i, 0, :] = last[i]
    lm = lm.detach()
    meta = {
        'kinds': [KIND_OF[r['kinds'][0]] for r in jrows], 'n_options': nopt,
        'record_index': list(range(nq)), 'counts': [1] * nq, 'max_opt': max_opt,
        'left_padding': False, 'seq_lens': [1] * nq,
    }
    # real training target: the collator's padded labels for this record's fan-out
    coll = template.data_collator([template.encode(dict(r)) for r in jrows])
    target = coll['labels'].to(dev).float()

    head = JevVerbalizerHead.from_pretrained(SNAP).float().to(dev)
    head.bias.requires_grad_(True)
    loss_fn = JevDistillLoss(None, _DummyTrainer())

    # (1) value
    with torch.inference_mode():
        out = head(None, lm, meta)
    our_val = float(loss_fn(outputs=out, labels=target, num_items_in_batch=None))
    re_val = float(rederived_jev_loss(out.logits.double(), out.option_mask, out.kinds, target.double()))
    v_pass = abs(our_val - re_val) < 1e-5
    print(f'[{"PASS" if v_pass else "FAIL"}] D(1) loss value our={our_val:.8f} rederived={re_val:.8f} '
          f'|diff|={abs(our_val - re_val):.3e}', flush=True)

    # (2) autograd our-vs-rederived on the head bias
    head.zero_grad(set_to_none=True)
    o1 = head(None, lm, meta)
    loss_fn(outputs=o1, labels=target, num_items_in_batch=None).backward()
    g_our = head.bias.grad.detach().clone()
    head.zero_grad(set_to_none=True)
    o2 = head(None, lm, meta)
    rederived_jev_loss(o2.logits, o2.option_mask, o2.kinds, target).backward()
    g_re = head.bias.grad.detach().clone()
    norm_rel = (float((g_our.double() - g_re.double()).norm()) /
                max(float(g_re.double().norm()), 1e-12))
    act = [i for i in range(24) if float(g_re[i].abs()) > 1e-6]
    worst = max((float((g_our[i] - g_re[i]).abs()) / float(g_re[i].abs()) for i in act), default=0.0)
    g_pass = norm_rel < 1e-4 and worst < 1e-4
    print(f'[{"PASS" if g_pass else "FAIL"}] D(2) autograd our-vs-rederived grad_norm rel={norm_rel:.3e}; '
          f'worst activated-slot rel={worst:.3e} over {len(act)} slots', flush=True)

    # (3) central finite-difference vs autograd on a few activated bias slots
    slots = [0, 1, 2, 8]  # noul false/true, score '0', choice 'A'
    eps = 1e-3
    fd_pass = True

    def loss_val():
        with torch.no_grad():
            o = head(None, lm, meta)
            return float(rederived_jev_loss(o.logits, o.option_mask, o.kinds, target))

    for s in slots:
        autog = float(g_re[s])
        orig = float(head.bias.data[s])
        head.bias.data[s] = orig + eps
        lp = loss_val()
        head.bias.data[s] = orig - eps
        lm_ = loss_val()
        head.bias.data[s] = orig
        fd = (lp - lm_) / (2 * eps)
        if max(abs(autog), abs(fd)) < 1e-6:
            print(f'   [NEGLIGIBLE] D(3) fd bias[{s}]: autograd={autog:.3e} fd={fd:.3e} (both ~0)', flush=True)
            continue
        rel = abs(fd - autog) / max(abs(autog), abs(fd), 1e-6)
        ok = rel < 3e-2
        fd_pass = fd_pass and ok
        print(f'   [{"PASS" if ok else "FAIL"}] D(3) fd bias[{s}]: autograd={autog:.6e} fd={fd:.6e} '
              f'rel={rel:.3e}', flush=True)
    return v_pass and g_pass and fd_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all', choices=['a', 'b', 'c', 'd', 'all'])
    args = ap.parse_args()
    results = {}
    try:
        if args.stage in ('a', 'all'):
            results['A'] = stage_a()
        if args.stage in ('b', 'all'):
            results['B'] = stage_b()
        if args.stage in ('c', 'all'):
            results['C'] = stage_c()
        if args.stage in ('d', 'all'):
            results['D'] = stage_d()
    except Exception:
        print('\n=== HARNESS EXCEPTION ===', flush=True)
        traceback.print_exc()
        return 2
    print('STAGES:', {k: ('PASS' if v else 'FAIL') for k, v in results.items()}, flush=True)
    ok = all(results.values())
    print('ALL PASS' if ok else 'SOME FAILED', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
