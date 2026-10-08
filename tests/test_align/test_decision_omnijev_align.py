# Copyright (c) ModelScope Contributors. All rights reserved.
"""OmniJev (`decision` task_type) alignment against the official `mso.infer.MSO1` oracle.

Runs the REAL integration path on both sides and compares them in four stages:

  REF   the official OmniJev repo (`mso.infer.MSO1`) on the ModelScope Qwen3.5-4B base + the HF-cached
        `tinnel123/OmniJev` adapter, running its real hybrid prefix-BRANCH (`ask_branch`).
  OURS  swift `get_model_processor(model_type='omnijev', base)` + `Swift.from_pretrained(adapter)`,
        running `OmniJevTemplate` / `OmniJevPreprocessor` / `OmniJevModelLoader._omnijev_branch_forward`
        / `OmniJevHead` / `OmniJevLoss`.

  A  encoding  -- our `OmniJevTemplate._encode` input_ids are a BYTE-exact supersequence of the oracle
                  `_encode_many` ids (the oracle prompt has no assistant close; ours appends its
                  `<|im_end|>\\n` suffix strictly AFTER the last option marker, which the branch never
                  reads), our `option_markers` equal the oracle spans, and the emitted column layout
                  (noul 2 / choice K+1 / score L) is right. Also re-checks the image-resize alignment
                  (decision 6) across several image sizes.
  B  head-only -- the oracle branch's real `(u, zq, feats)` per question are fed to BOTH the oracle
                  head/ord + an UNROUNDED mirror of `_finish` and our `OmniJevHead.score_features`
                  (fp32, eval): probs must match to fp32 precision, which isolates the FiLM/ordinal port
                  and the serving-time calibration folding (per-kind temperature + noul log-odds bias).
                  The unrounded mirror is cross-checked against the REAL rounded `system_one` to 1e-4.
  C  full fwd  -- our swift-loaded model end-to-end (its own two-stage cached branch) vs the oracle's
                  own branch, bf16 + eager both sides: validates the whole patch + adapter + branch +
                  head path, batch=1 per record and a multi-row train batch (padding invariance).
  D  loss+grad -- `OmniJevLoss` is validated three ways: value == an independently re-derived
                  masked soft-CE + 0.5*RPS(score); its autograd grads on a head param == the re-derived
                  formula's grads; autograd == central finite-difference.

Both sides force MSO_FLA=0 (torch gated-delta path, no Triton kernel divergence) and MSO_ATTN=eager, so
any mismatch is attributable to logic, not kernels. The oracle's rounding (`round(p, 4)` in `_finish`)
loses precision, so the reference probabilities used for B/C are the UNROUNDED mirror, itself validated
against the real rounded output.

Usage:
    python tests/test_align/test_decision_omnijev_align.py --stage a      # encoding (loads MSO1 + swift)
    python tests/test_align/test_decision_omnijev_align.py --stage all    # all four stages
Env overrides: OMNIJEV_CKPT / OMNIJEV_BASE / OMNIJEV_REPO / OMNIJEV_IMG_DIR.
"""
import argparse
import json
import math
import os
import sys
import traceback

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
# control group: identical kernels on both sides (torch gated-delta path, eager attention)
os.environ['MSO_FLA'] = '0'
os.environ['MSO_ATTN'] = 'eager'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')

CKPT = os.environ.get(
    'OMNIJEV_CKPT',
    '/root/.cache/huggingface/hub/models--tinnel123--OmniJev/snapshots/'
    '91248107b7f5230bd5715db2e62e83c9a72fcd81')
BASE = os.environ.get('OMNIJEV_BASE',
                      '/mnt/workspace/.cache/modelscope/hub/models/Qwen--Qwen3.5-4B/snapshots/master')
REPO = os.environ.get('OMNIJEV_REPO', '/tmp/omnijev_repo_c7')
IMG_DIR = os.environ.get('OMNIJEV_IMG_DIR', '/tmp')

# split base/adapter layout: point swift's head-artifact locator at the adapter dir
for _k, _f in (('OMNIJEV_NEW_TOK_EMB', 'new_tok_emb.pt'), ('OMNIJEV_HEAD', 'head.pt'),
               ('OMNIJEV_ORD', 'ord.pt'), ('OMNIJEV_HEAD_META', 'head_meta.json')):
    os.environ.setdefault(_k, os.path.join(CKPT, _f))

sys.path.insert(0, REPO)
import torch  # noqa: E402
from mso.head import choice_outputs  # noqa: E402
from mso.infer import MSO1  # noqa: E402

from swift.dataset.preprocessor.decision import OmniJevPreprocessor  # noqa: E402
from swift.loss.decision import OmniJevLoss  # noqa: E402
from swift.model import get_model_processor  # noqa: E402
from swift.model.decision_head import get_scoring_head, masked_softmax  # noqa: E402
from swift.template import get_template  # noqa: E402
from swift.tuners import Swift  # noqa: E402

DEV_REF = 'cuda:0'
DEV_OURS = 'cuda:1'
KIND_INT = {'noul': 0, 'choice': 1, 'score': 2}

IMG = os.path.join(IMG_DIR, 'omni_probe.png')
# one single-still question per kind; `gold` is read only by the swift preprocessor (the oracle ignores
# it), giving Stage D a real target distribution. choice uses a criteria-map, score an explicit level list.
QUESTIONS = {
    'noul': {
        'type': 'noul', 'instructions': 'There is a cat in the image.', 'gold': 'yes'
    },
    'choice': {
        'type': 'choice',
        'instructions': 'Which animal is in the image?',
        'criteria': {
            'cat': None,
            'dog': None,
            'bird': None
        },
        'gold': 'cat'
    },
    'score': {
        'type': 'score', 'instructions': 'Rate the image quality.',
        'levels': ['bad', 'ok', 'good', 'great'], 'gold': 'good'
    },
}
QIDS = list(QUESTIONS)
# emitted column width per kind (== OmniJevHead's layout): noul 2, choice K+1, score L
NCOLS = {'noul': 2, 'choice': len(QUESTIONS['choice']['criteria']) + 1, 'score': len(QUESTIONS['score']['levels'])}


def ensure_image(path, w=224, h=224):
    """Write a deterministic RGB test image (a seeded gradient + noise) if it does not exist."""
    from PIL import Image
    import numpy as np
    if os.path.exists(path):
        return path
    rng = np.random.RandomState(0)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([(xx * 255 // max(1, w - 1)), (yy * 255 // max(1, h - 1)), ((xx + yy) * 255 // max(1, w + h - 2))],
                    axis=-1).astype('uint8')
    noise = rng.randint(0, 24, size=(h, w, 3), dtype='uint8')
    Image.fromarray(np.clip(base.astype('int32') + noise, 0, 255).astype('uint8')).save(path)
    return path


# --------------------------- lazy singletons ---------------------------
_REF = None


def ref_mso():
    """The official MSO1 on cuda:0 (loaded once)."""
    global _REF
    if _REF is None:
        ensure_image(IMG)
        print('=== REF: instantiating official MSO1 ===', flush=True)
        m = MSO1(CKPT, BASE)
        # pin the whole oracle to cuda:0 so it never collides with the swift side on cuda:1
        m.dev = torch.device(DEV_REF)
        m.model = m.model.to(DEV_REF)
        print(f'   MSO1 ready: branch={m.branch} ordinal={m.ordinal} lm_feats={m.lm_feats} '
              f'temps={m.temps} bias_noul={m.biases["noul"]:.6f} open_ids={m.open_ids} close_ids={m.close_ids}',
              flush=True)
        _REF = m
    return _REF


_OUR = None


def our_model():
    """swift OmniJev on cuda:1 (loaded once) -> (model, proc, template, head)."""
    global _OUR
    if _OUR is None:
        ensure_image(IMG)
        print('=== OURS: get_model_processor(omnijev, base) + Swift.from_pretrained(adapter) ===', flush=True)
        model, proc = get_model_processor(
            BASE, model_type='omnijev', task_type='decision', torch_dtype=torch.bfloat16, device_map=DEV_OURS,
            attn_impl='eager')
        head = get_scoring_head(model)
        assert head is not None, 'FAIL: OmniJev loader did not attach scoring_head'
        model = Swift.from_pretrained(model, CKPT, is_trainable=True)
        head2 = get_scoring_head(model)
        assert head2 is head, 'FAIL: scoring_head identity changed across the PeftModel wrap'
        n_lora = sum(1 for n, _ in model.named_parameters() if 'lora_' in n)
        print(f'   swift model ready: head={type(head).__name__} ordinal={head.ordinal} n_lora={n_lora}', flush=True)
        tmpl = get_template(proc, template_type='omnijev')
        _OUR = (model, proc, tmpl, head)
    return _OUR


def maxdiff(a, b):
    a = torch.as_tensor(a, dtype=torch.float64).flatten()
    b = torch.as_tensor(b, dtype=torch.float64).flatten()
    if a.numel() != b.numel():
        return float('inf')
    return float((a - b).abs().max()) if a.numel() else 0.0


# --------------------------- oracle capture + unrounded mirror ---------------------------
def ref_capture(m, img, questions):
    """Run the REAL `MSO1.ask_branch`, capturing per-qid (u, zq, feats, mu, keys, opts, rounded answer).

    `ask_branch` calls the decision head (`m.model.head`) or the ordinal head (`m.model.ord`) exactly once
    per qid, then `_finish` once per qid, both in qid order -- so wrapping those three and zipping with
    `qids` recovers the raw features (for our head to consume), the raw head output `mu`, and the REAL
    rounded answer (to cross-check the unrounded mirror). Everything is the oracle's own code path; only
    the capture is added, and it is restored in `finally`.
    """
    qids = list(questions)
    feats_cap, mu_cap, fin_cap = [], [], []
    orig_head = m.model.head.forward
    orig_ord = m.model.ord.forward
    orig_finish = m._finish

    def head_fwd(u_opts, z_q, type_id=0, feats=None):
        out = orig_head(u_opts, z_q, type_id, feats)
        feats_cap.append((u_opts.detach().clone(), z_q.detach().clone(),
                          None if feats is None else feats.detach().clone()))
        mu_cap.append(out.detach().clone())
        return out

    def ord_fwd(z_q, u_levels):
        out = orig_ord(z_q, u_levels)
        feats_cap.append((u_levels.detach().clone(), z_q.detach().clone(), None))
        mu_cap.append(out.detach().clone())
        return out

    def finish(q, keys, opts, mu, lat):
        res = orig_finish(q, keys, opts, mu, lat)
        fin_cap.append((q, keys, opts, res))
        return res

    m.model.head.forward = head_fwd
    m.model.ord.forward = ord_fwd
    m._finish = finish
    try:
        m.system_one({'images': [img]}, questions)
    finally:
        m.model.head.forward = orig_head
        m.model.ord.forward = orig_ord
        m._finish = orig_finish
    assert len(feats_cap) == len(qids) == len(mu_cap) == len(fin_cap), (
        f'capture misalign: feats={len(feats_cap)} mu={len(mu_cap)} finish={len(fin_cap)} qids={len(qids)}')
    out = {}
    for i, qid in enumerate(qids):
        u, zq, feats = feats_cap[i]
        q, keys, opts, ans = fin_cap[i]
        out[qid] = {'u': u, 'zq': zq, 'feats': feats, 'mu': mu_cap[i], 'keys': keys, 'opts': opts, 'ans': ans}
    return out


def ref_finish_unrounded(m, q, mu):
    """Mirror `MSO1._finish` WITHOUT `round()`, returning the probability vector in OUR column layout:
    noul -> [p_yes, p_no]; choice -> [opt_0..opt_{K-1}, abstain]; score -> [level_0..level_{L-1}].
    Reuses the oracle's own `_scale` / `choice_outputs` so only the ~10-line glue is re-implemented, and
    that glue is cross-checked against the real rounded `_finish` in `_mirror_ok`."""
    t = q['type']
    if t == 'noul':
        p0 = float(mu[0])
        b = m.biases.get('noul', 0.0)
        if b:
            p0 = min(max(p0, 1e-9), 1.0 - 1e-9)
            z = max(-40.0, min(40.0, math.log(p0 / (1.0 - p0)) + b))
            p0 = 1.0 / (1.0 + math.exp(-z))
        return [float(x) for x in m._scale([p0, 1.0 - p0], 'noul')]
    co = choice_outputs(mu, allow_abstain=(t == 'choice'))
    full = [float(x) for x in m._scale([float(x) for x in co.probs] + [float(co.abstain)], t)]
    probs, abst = full[:-1], full[-1]
    if t == 'score':
        s = sum(probs) or 1.0
        return [p / s for p in probs]
    return probs + [abst]


def _mirror_ok(q, unrounded, ans, keys):
    """The unrounded mirror must agree with the REAL rounded `_finish` output to within rounding (<1e-4).
    noul: ans['noul']; choice: ans['probabilities'][key] + ans['abstain']; score: ans['probabilities'][key]."""
    t = q['type']
    if t == 'noul':
        return abs(unrounded[0] - float(ans['noul'])) < 1e-4
    if t == 'score':
        got = [float(ans['probabilities'][k]) for k in keys]
        return maxdiff(unrounded, got) < 1e-4
    got = [float(ans['probabilities'][k]) for k in keys] + [float(ans['abstain'])]
    return maxdiff(unrounded, got) < 1e-4


# --------------------------- Stage A: encoding ---------------------------
def stage_a():
    m = ref_mso()
    model, proc, tmpl, head = our_model()
    tok = proc.tokenizer
    tmpl.set_mode('train')

    ok_all = True
    print('=== A1-A3: swift encode input_ids are a byte-exact supersequence of the oracle ids ===', flush=True)
    keys_l, opts_l, ids_l, spans_l, _pix = m._encode_many(IMG, None, QUESTIONS, QIDS)
    rows = OmniJevPreprocessor().preprocess({'images': [IMG], 'questions': QUESTIONS})
    assert len(rows) == len(QIDS), f'fan-out {len(rows)} != {len(QIDS)}'
    for i, (row, qid) in enumerate(zip(rows, QIDS)):
        enc = tmpl.encode(dict(row))
        ours = list(enc['input_ids'])
        ref = list(ids_l[i])
        meta = enc['decision_meta']
        prefix_ok = ours[:len(ref)] == ref
        tail = tok.decode(ours[len(ref):]) if len(ours) >= len(ref) else '<ours SHORTER>'
        # markers equal the oracle spans (positions are relative to the same real-token start);
        # this row is one question, so option_markers is [[(open, close), ...]] -> take [0]
        markers = meta['option_markers'][0]
        opens = [int(o) for o, _ in markers]
        closes = [int(c) for _, c in markers]
        ref_opens, ref_closes = spans_l[i]
        span_ok = opens == list(ref_opens) and closes == list(ref_closes)
        kinds = meta['kinds']
        kind_int = int(kinds[0]) if isinstance(kinds, (list, tuple)) else int(kinds)
        ncols = int(meta['n_options'][0]) if isinstance(meta['n_options'], (list, tuple)) else int(meta['n_options'])
        layout_ok = kind_int == KIND_INT[qid] and ncols == NCOLS[qid]
        ok = prefix_ok and span_ok and layout_ok
        ok_all = ok_all and ok
        print(f'[{"PASS" if ok else "FAIL"}] A/{qid} prefix={prefix_ok} spans={span_ok} layout={layout_ok} '
              f'kind={kind_int} ncols={ncols} len ours={len(ours)} ref={len(ref)} tail={tail!r} '
              f'markers={markers} ref_spans=({list(ref_opens)},{list(ref_closes)})', flush=True)
        if not prefix_ok:
            n = min(len(ours), len(ref))
            div = next((k for k in range(n) if ours[k] != ref[k]), n)
            lo = max(0, div - 8)
            print(f'   first divergence at {div}: ours={tok.decode(ours[lo:div + 6])!r} '
                  f'ref={tok.decode(ref[lo:div + 6])!r}', flush=True)

    print('=== A4: real collator -> flat decision_meta + padded labels (train right / infer left) ===', flush=True)
    encoded = [tmpl.encode(dict(row)) for row in rows]
    collated = tmpl.data_collator(encoded)
    dm = collated['decision_meta']
    a4 = (len(dm['record_index']) == len(QIDS) and dm['max_opt'] == max(NCOLS.values())
          and collated['input_ids'].shape[0] == len(QIDS) and 'labels' in collated
          and tuple(collated['labels'].shape) == (len(QIDS), dm['max_opt'])
          and all(0 <= ri < len(QIDS) for ri in dm['record_index']) and dm['left_padding'] is False)
    ok_all = ok_all and a4
    print(f'[{"PASS" if a4 else "FAIL"}] A4 train collator record_index={dm["record_index"]} counts={dm["counts"]} '
          f'n_options={dm["n_options"]} kinds={dm["kinds"]} max_opt={dm["max_opt"]} '
          f'left_padding={dm["left_padding"]} labels={tuple(collated["labels"].shape)}', flush=True)

    tmpl.set_mode('transformers')
    coll_inf = tmpl.data_collator([tmpl.encode(dict(row)) for row in rows])
    a5 = coll_inf['decision_meta']['left_padding'] is True
    ok_all = ok_all and a5
    print(f'[{"PASS" if a5 else "FAIL"}] A5 inference collator left_padding={coll_inf["decision_meta"]["left_padding"]}',
          flush=True)
    tmpl.set_mode('train')

    print('=== A6: image-resize alignment (decision 6) byte-matches the oracle across sizes ===', flush=True)
    for (w, h) in ((224, 224), (512, 384), (1024, 768)):
        p = os.path.join(IMG_DIR, f'omni_a6_{w}x{h}.png')
        ensure_image(p, w, h)
        _, _, ids_one, _, _ = m._encode_many(p, None, {'noul': QUESTIONS['noul']}, ['noul'])
        row = OmniJevPreprocessor().preprocess({'images': [p], 'questions': {'noul': QUESTIONS['noul']}})[0]
        enc = tmpl.encode(dict(row))
        ours, ref = list(enc['input_ids']), list(ids_one[0])
        ok = ours[:len(ref)] == ref
        ok_all = ok_all and ok
        print(f'[{"PASS" if ok else "FAIL"}] A6 {w}x{h} byte-prefix={ok} len ours={len(ours)} ref={len(ref)}',
              flush=True)
    return ok_all


# --------------------------- Stage B: head-only fp32 ---------------------------
def stage_b():
    m = ref_mso()
    model, proc, tmpl, head = our_model()
    cap = ref_capture(m, IMG, QUESTIONS)
    hdev = next(head.parameters()).device
    head.eval()  # fold serving-time calibration (matches the oracle's _finish)

    ok_all = True
    print('=== B0: unrounded _finish mirror agrees with the REAL rounded system_one (<1e-4) ===', flush=True)
    for qid in QIDS:
        q = QUESTIONS[qid]
        un = ref_finish_unrounded(m, q, cap[qid]['mu'].to(torch.float64))
        ok = _mirror_ok(q, un, cap[qid]['ans'], cap[qid]['keys'])
        ok_all = ok_all and ok
        print(f'[{"PASS" if ok else "FAIL"}] B0/{qid} unrounded={[round(x, 5) for x in un]} '
              f'rounded_ans={ {k: v for k, v in cap[qid]["ans"].items() if k in ("noul", "probabilities", "abstain")} }',
              flush=True)

    print('=== B1-B3: same (u,zq,feats) -> our head (fp32, eval) vs oracle head/ord + _finish ===', flush=True)
    for qid in QIDS:
        q = QUESTIONS[qid]
        kind = KIND_INT[q['type']]
        u = cap[qid]['u'].to(hdev)
        zq = cap[qid]['zq'].to(hdev)
        feats = None if cap[qid]['feats'] is None else cap[qid]['feats'].to(hdev)
        with torch.inference_mode():
            cols = head.score_features(u, zq, feats, kind)
        n = int(cols.shape[0])
        mask = torch.ones((1, n), dtype=torch.bool, device=cols.device)
        our = masked_softmax(cols.float()[None], mask)[0].cpu()
        ref = torch.tensor(ref_finish_unrounded(m, q, cap[qid]['mu'].to(torch.float64)), dtype=torch.float64)
        d = maxdiff(our, ref)
        ok = d < 1e-5 and n == NCOLS[qid]
        ok_all = ok_all and ok
        print(f'[{"PASS" if ok else "FAIL"}] B/{qid} kind={kind} n={n} max|prob diff|={d:.3e} '
              f'our={[round(float(x), 5) for x in our]}', flush=True)
    return ok_all


# --------------------------- Stage C: full branch forward bf16 ---------------------------
def stage_c():
    m = ref_mso()
    model, proc, tmpl, head = our_model()
    cap = ref_capture(m, IMG, QUESTIONS)
    ref_probs = {qid: ref_finish_unrounded(m, QUESTIONS[qid], cap[qid]['mu'].to(torch.float64)) for qid in QIDS}
    rows = OmniJevPreprocessor().preprocess({'images': [IMG], 'questions': QUESTIONS})
    model.eval()  # inference path: head folds calibration, backbone deterministic

    ok_all = True
    print('=== C1-C3: batch=1 per record, swift full branch forward vs oracle branch (bf16, eager) ===', flush=True)
    tmpl.set_mode('train')
    for row, qid in zip(rows, QIDS):
        n = NCOLS[qid]
        enc = tmpl.encode(dict(row))
        coll = tmpl.data_collator([enc])
        feed = {k: (v.to(DEV_OURS) if torch.is_tensor(v) else v)
                for k, v in coll.items() if k != 'labels'}
        with torch.inference_mode():
            out = model(**feed)
        our = masked_softmax(out.logits.float(), out.option_mask)[0, :n].cpu()
        d = maxdiff(our, torch.tensor(ref_probs[qid], dtype=torch.float64))
        ok = d < 2e-2
        ok_all = ok_all and ok
        print(f'[{"PASS" if ok else "FAIL"}] C/{qid} n={n} max|prob diff|={d:.3e} '
              f'our={[round(float(x), 4) for x in our]} ref={[round(float(x), 4) for x in ref_probs[qid]]}',
              flush=True)

    print('=== C4: multi-row TRAIN batch (right-pad) vs per-row oracle branch (padding invariance) ===', flush=True)
    encoded = [tmpl.encode(dict(row)) for row in rows]
    coll = tmpl.data_collator(encoded)
    dm = coll['decision_meta']
    feed = {k: (v.to(DEV_OURS) if torch.is_tensor(v) else v) for k, v in coll.items() if k != 'labels'}
    with torch.inference_mode():
        out = model(**feed)
    worst = 0.0
    for qi, qid in enumerate(QIDS):
        n = NCOLS[qid]
        our = masked_softmax(out.logits.float(), out.option_mask)[qi, :n].cpu()
        worst = max(worst, maxdiff(our, torch.tensor(ref_probs[qid], dtype=torch.float64)))
    ok = worst < 2e-2 and dm['left_padding'] is False
    ok_all = ok_all and ok
    print(f'[{"PASS" if ok else "FAIL"}] C4 right-pad batch N={len(QIDS)} T={coll["input_ids"].shape[1]} '
          f'worst max|prob diff|={worst:.3e}', flush=True)
    return ok_all


# ------------------------------ Stage D: loss + grad ------------------------------
def rederived_omnijev_loss(logits, option_mask, kinds, target, rps_w=0.5):
    """Independent re-derivation of OmniJevLoss: masked soft-CE over each question's own columns +
    0.5 * mean_over_questions(RPS * 1{score}), where RPS is the OFFICIAL form -- plain sum over the FULL
    cumulative gap (no per-row normalisation, no [:-1] drop), pad columns contributing 0 to both CDFs."""
    probs = masked_softmax(logits, option_mask)
    logp = torch.log(probs.clamp_min(1e-9))
    per_q = -(target * logp * option_mask).sum(-1)
    total_q = per_q.shape[0]
    ce = per_q.sum() / total_q if total_q else per_q.sum()
    score_mask = kinds == KIND_INT['score']
    term = torch.zeros((), device=logits.device, dtype=torch.float64)
    if bool(score_mask.any()):
        cum_m = probs.cumsum(-1)
        cum_t = target.cumsum(-1)
        per_row = (cum_m - cum_t).pow(2).sum(-1)
        term = rps_w * (per_row[score_mask].sum() / total_q if total_q else per_row[score_mask].sum())
    return ce + term


class _DummyTrainer:
    pass


def stage_d():
    m = ref_mso()
    model, proc, tmpl, head = our_model()
    cap = ref_capture(m, IMG, QUESTIONS)
    hdev = next(head.parameters()).device

    rows = OmniJevPreprocessor().preprocess({'images': [IMG], 'questions': QUESTIONS})
    tmpl.set_mode('train')
    coll = tmpl.data_collator([tmpl.encode(dict(r)) for r in rows])
    target = coll['labels'].to(hdev).float()
    kinds = [int(k) for k in coll['decision_meta']['kinds']]
    n_options = [int(x) for x in coll['decision_meta']['n_options']]
    max_opt = int(coll['decision_meta']['max_opt'])
    meta = {
        'kinds': kinds, 'n_options': n_options, 'record_index': list(range(len(QIDS))),
        'counts': [1] * len(QIDS), 'max_opt': max_opt, 'left_padding': False, 'seq_lens': [1] * len(QIDS)
    }
    feats_list = []
    for qid in QIDS:
        u = cap[qid]['u'].to(hdev)
        zq = cap[qid]['zq'].to(hdev)
        ft = None if cap[qid]['feats'] is None else cap[qid]['feats'].to(hdev)
        feats_list.append((u, zq, ft))

    head.train()  # training path: raw logits, NO serving calibration folded
    loss_fn = OmniJevLoss(None, _DummyTrainer())
    scorer_bias = head.head.score.bias  # [1], drives noul+choice CE
    ord_bias = head.ord.z[2].bias  # [1], drives the score ordinal head (CE + RPS)
    params = [('scorer.bias', scorer_bias, 0), ('ord.z.bias', ord_bias, 0)]
    for _, p, _ in params:
        p.requires_grad_(True)

    ok_all = True
    # (1) value
    with torch.no_grad():
        out = head.forward_features(feats_list, meta)
    our_val = float(loss_fn(outputs=out, labels=target, num_items_in_batch=None))
    re_val = float(rederived_omnijev_loss(out.logits.double(), out.option_mask, out.kinds, target.double()))
    v_pass = abs(our_val - re_val) < 1e-5
    ok_all = ok_all and v_pass
    print(f'[{"PASS" if v_pass else "FAIL"}] D(1) loss value our={our_val:.8f} rederived={re_val:.8f} '
          f'|diff|={abs(our_val - re_val):.3e}', flush=True)

    # (2) autograd our-vs-rederived on both params
    grads_our, grads_re = {}, {}
    head.zero_grad(set_to_none=True)
    o1 = head.forward_features(feats_list, meta)
    loss_fn(outputs=o1, labels=target, num_items_in_batch=None).backward()
    for name, p, _ in params:
        grads_our[name] = p.grad.detach().clone()
    head.zero_grad(set_to_none=True)
    o2 = head.forward_features(feats_list, meta)
    rederived_omnijev_loss(o2.logits, o2.option_mask, o2.kinds, target).backward()
    for name, p, _ in params:
        grads_re[name] = p.grad.detach().clone()
    for name, _, _ in params:
        go, gr = grads_our[name].double(), grads_re[name].double()
        rel = float((go - gr).norm()) / max(float(gr.norm()), 1e-12)
        ok = rel < 1e-4
        ok_all = ok_all and ok
        print(f'[{"PASS" if ok else "FAIL"}] D(2) autograd our-vs-rederived {name} grad rel={rel:.3e} '
              f'our={float(go.flatten()[0]):.6e} re={float(gr.flatten()[0]):.6e}', flush=True)

    # (3) central finite-difference vs autograd (re-derived grad) on both params
    eps = 1e-3

    def loss_val():
        with torch.no_grad():
            o = head.forward_features(feats_list, meta)
            return float(rederived_omnijev_loss(o.logits, o.option_mask, o.kinds, target))

    for name, p, idx in params:
        autog = float(grads_re[name].flatten()[idx])
        orig = float(p.data.flatten()[idx])
        p.data.flatten()[idx] = orig + eps
        lp = loss_val()
        p.data.flatten()[idx] = orig - eps
        lm_ = loss_val()
        p.data.flatten()[idx] = orig
        fd = (lp - lm_) / (2 * eps)
        if max(abs(autog), abs(fd)) < 1e-6:
            print(f'   [NEGLIGIBLE] D(3) fd {name}[{idx}]: autograd={autog:.3e} fd={fd:.3e} (both ~0)', flush=True)
            continue
        rel = abs(fd - autog) / max(abs(autog), abs(fd), 1e-6)
        ok = rel < 3e-2
        ok_all = ok_all and ok
        print(f'   [{"PASS" if ok else "FAIL"}] D(3) fd {name}[{idx}]: autograd={autog:.6e} fd={fd:.6e} rel={rel:.3e}',
              flush=True)
    return ok_all


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
