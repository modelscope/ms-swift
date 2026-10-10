# Copyright (c) ModelScope Contributors. All rights reserved.
"""OmniJev branch forward + option-token registration (decision G1).

A faithful port of the OmniJev official `mso/branch.py` (plus the small pieces of `mso/records.py` the
loader needs: `add_option_tokens`, `find_rope_owner`, `rope_positions`). It is a model-local module --
nothing here touches shared swift framework code; `OmniJevModelLoader` calls into it.

Why branch (not a block-diagonal attention mask): a block mask keeps options apart only in the softmax
attention layers, but OmniJev's real backbone (Qwen3.5-4B) is a *hybrid* gated-deltanet model whose
linear-attention layers carry recurrent state along the sequence regardless of the mask, so on such a
backbone the options leak into each other (official measured option representations differing by 4-17
across orders). Branching restores exact isolation the other way round::

    prefix  = chat head + image tokens (+ instruction)      -> ONE forward, cache kept
    suffix  = one option block per row                       -> every (question, option) is a row of a
              batched forward continuing a COPY of the expanded prefix cache

Every row starts from the same state and never sees another option, so the answer is exactly
order-invariant (official measured 0.0) and equals a plain prefix+row forward (max |du| ~1e-4). Rows
are processed in chunks (`MSO_BRANCH_CHUNK`, default 64) against a fresh copy of the prefix cache, and
`repeat_interleave` is differentiable, so training uses the same path (`MSO_BRANCH_CKPT=1` recomputes a
chunk in backward instead of keeping its activations).

RUN-PHASE VERIFY (fragile, no local oracle): `expand_cache` assumes the transformers 5.x cache object
exposes `.layers` (each a layer-cache with tensor attrs); `get_rope_index`'s signature and the hybrid
cache's `repeat_interleave`-ability are version-specific; and calling the backbone forward several times
per step (prefix + chunks) interacts with DDP's reducer / model-level gradient checkpointing. These are
checked against a real Qwen3.5-4B in the run phase, not during authoring (AST-only + pure-function
self-checks here).
"""
import copy
import os
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.utils.checkpoint

OPT_OPEN = '<|opt|>'
OPT_CLOSE = '<|/opt|>'
N_FEAT = 6
_PIX_KEYS = ('pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw', 'mm_token_type_ids')


def is_hybrid(model) -> bool:
    """True when the backbone has linear-attention layers (`config.text_config.layer_types` contains
    'linear'), i.e. a block mask cannot isolate options and branch is required. Qwen3.5-4B is hybrid."""
    cfg = getattr(model, 'config', None)
    tc = getattr(cfg, 'text_config', cfg)
    lt = getattr(tc, 'layer_types', None)
    return bool(lt) and any('linear' in str(t) for t in lt)


def find_rope_owner(m, depth: int = 0):
    """The submodule that owns `get_rope_index` (Qwen3VLModel); PEFT / DDP bury it a few attrs deep."""
    if depth > 6:
        return None
    if hasattr(m, 'get_rope_index'):
        return m
    for attr in ('model', 'base_model', 'module'):
        sub = getattr(m, attr, None)
        if sub is not None and sub is not m:
            r = find_rope_owner(sub, depth + 1)
            if r is not None:
                return r
    return None


def rope_positions(rope_owner, enc: Dict[str, Any]) -> Optional[torch.Tensor]:
    """mrope `position_ids` `[3, B, T]` of `enc` from the model's own `get_rope_index` (None if the
    backbone has no rope owner). Port of `MSO.rope_positions(enc, spans=None, align_options=False)` --
    the branch prefix needs NO option alignment (each row is scored from the same prefix state, so order
    invariance comes from branching, not from shared rope start positions)."""
    if rope_owner is None:
        return None
    mm = enc.get('mm_token_type_ids')
    if mm is None:
        # transformers 5.x wants 0=text 1=image 2=video per token; rebuild it from the ids.
        cfg = getattr(rope_owner, 'config', None)
        img_id = getattr(cfg, 'image_token_id', None)
        vid_id = getattr(cfg, 'video_token_id', None)
        ids = enc['input_ids']
        mm = torch.zeros_like(ids)
        if img_id is not None:
            mm[ids == img_id] = 1
        if vid_id is not None:
            mm[ids == vid_id] = 2
    kw = dict(
        input_ids=enc['input_ids'],
        image_grid_thw=enc.get('image_grid_thw'),
        video_grid_thw=enc.get('video_grid_thw'),
        attention_mask=enc.get('attention_mask'))
    try:
        pos, _ = rope_owner.get_rope_index(mm_token_type_ids=mm, **kw)  # transformers >= 5.x
    except TypeError:
        pos, _ = rope_owner.get_rope_index(**kw)  # older signature
    return pos


def _rep(v, K):
    if torch.is_tensor(v) and v.dim() >= 1:
        return v.repeat_interleave(K, dim=0)
    if isinstance(v, list):
        return [_rep(x, K) for x in v]
    if isinstance(v, tuple):
        return tuple(_rep(x, K) for x in v)
    if isinstance(v, dict):
        return {k: _rep(x, K) for k, x in v.items()}
    return v


def expand_cache(cache, K):
    """A copy of `cache` with every tensor repeated K times along the batch axis (KV caches AND recurrent
    states). The original stays intact so it can be expanded again for the next chunk."""
    new = copy.copy(cache)
    layers = []
    for layer in cache.layers:
        l2 = copy.copy(layer)
        for k, v in list(vars(layer).items()):
            setattr(l2, k, _rep(v, K))
        layers.append(l2)
    new.layers = layers
    # TRAINING FIX (swift-only; the official `mso/branch.py` port runs inference under no_grad, so it never
    # hit this). transformers writes the post-forward recurrent state back IN-PLACE into the cache buffer
    # (`cache_utils.update_recurrent_state` -> `recurrent_states[idx].copy_(...)`), and the linear-attention
    # layer feeds that SAME buffer to fla's chunk kernel as `initial_state`, which `save_for_backward`
    # snapshots. The in-place copy_ then bumps the buffer's version, so backward dies with "one of the
    # variables needed for gradient computation has been modified by an inplace operation" (a detached
    # state does NOT help -- fla saves initial_state unconditionally). The branch DISCARDS the updated cache
    # anyway (every option row is an independent continuation of the one pristine prefix, re-expanded per
    # chunk, and `_chunk_h` keeps only hidden_states), so the write-back is dead work: neutralise it on the
    # expanded copy only. Forward values, the full prefix gradient, and `cache0` itself are all untouched.
    new.update_recurrent_state = lambda *a, **k: None
    return new


def prefix_inputs(enc: Dict[str, Any], L: int) -> Dict[str, Any]:
    """The prefix encoding (`ids[:L]` + the pixel tensors) cut out of a single-question encoding."""
    ids = enc['input_ids'][:, :L]
    p = {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
    for k in _PIX_KEYS:
        if k in enc:
            p[k] = enc[k][:, :L] if k == 'mm_token_type_ids' else enc[k]
    return p


def rows_from_ids(ids, L: int, opens: Sequence[int], closes: Sequence[int],
                  k: Optional[int] = None) -> List[List[int]]:
    """Token ids of ONE single-question sequence (prefix + instruction + option blocks), the prefix length
    `L` and the option spans -> K rows, each the shared part `ids[L:opens[0]]` followed by one option block
    `ids[o:c+1]`. `L` MUST be `< opens[0]` so every row keeps the token before its option (that is where
    `zq` is read)."""
    ids = list(ids)
    if not opens or not closes:
        return [ids[L:]]
    head = ids[L:opens[0]]
    pairs = [(o, c) for o, c in zip(opens, closes) if c >= o]
    if k is not None:
        pairs = pairs[:k]
    return [head + ids[o:c + 1] for o, c in pairs]


def _marks(r: Sequence[int], open_ids: Sequence[int], close_ids: Sequence[int]):
    """(index of the first open marker, index of the last token of the last close marker, indices of the
    option text tokens) inside one row."""
    ol, cl = len(open_ids), len(close_ids)
    r = list(r)
    opens = [k for k in range(len(r) - ol + 1) if r[k:k + ol] == list(open_ids)]
    closes = [k for k in range(len(r) - cl + 1) if r[k:k + cl] == list(close_ids)]
    o_ = opens[0] if opens else 0
    c_ = (closes[-1] + cl - 1) if closes else len(r) - 1
    return o_, c_, list(range(o_ + ol, c_ - cl + 1))


def branch_forward(backbone, penc: Dict[str, Any], rows: List[List[int]], dev, open_ids: Sequence[int],
                   close_ids: Sequence[int], lm_head=None, prefix_pos: Optional[torch.Tensor] = None,
                   pos_start: Optional[int] = None, groups: Optional[List[List[int]]] = None,
                   chunk: Optional[int] = None, pad_id: int = 0, mrope_axes: int = 3) -> Dict[str, Any]:
    """`penc`: prefix encoding (`input_ids [1, L]` + pixel tensors); `rows`: token-id lists continuing the
    prefix; `groups`: lists of row indices forming one question (the K-way softmax feature is taken inside a
    group). `prefix_pos`: `[3, 1, L]` positions of the prefix (`rope_positions`); `pos_start`: rope position
    of the first suffix token (default: max prefix position + 1).
      -> {'u': [R, D] hidden at each row's closing marker, 'zq': [R, D] hidden just before each row's
          opening marker, 'feats': [R, 6] backbone log-prob features (lm_head given) else None}
    """
    R = len(rows)
    prefix_ids = penc['input_ids']
    L = int(prefix_ids.shape[1])
    kw: Dict[str, Any] = {'input_ids': prefix_ids, 'attention_mask': torch.ones_like(prefix_ids), 'use_cache': True}
    kw.update({k: v for k, v in penc.items() if k in _PIX_KEYS})
    if prefix_pos is not None:
        kw['position_ids'] = prefix_pos
    out = backbone(**kw, logits_to_keep=1)
    cache0 = out.past_key_values
    if pos_start is None:
        pos_start = (int(prefix_pos.max()) + 1) if prefix_pos is not None else L
    p0 = int(pos_start)
    chunk = chunk or int(os.environ.get('MSO_BRANCH_CHUNK', '64'))
    us, zqs, sums_l, cnts_l, first_l = [], [], [], [], []
    for s in range(0, R, chunk):
        sub = rows[s:s + chunk]
        Rc = len(sub)
        n = max(len(r) for r in sub)
        ids = torch.full((Rc, n), pad_id, device=dev, dtype=torch.long)
        att = torch.zeros((Rc, n), dtype=torch.long, device=dev)
        for i, r in enumerate(sub):
            ids[i, :len(r)] = torch.tensor(r, device=dev)
            att[i, :len(r)] = 1
        full_att = torch.cat([torch.ones((Rc, L), dtype=torch.long, device=dev), att], dim=1)
        pos = torch.arange(p0, p0 + n, device=dev)[None, None, :].expand(mrope_axes, Rc, n).contiguous()
        cache_pos = torch.arange(L, L + n, device=dev)

        if torch.is_grad_enabled() and os.environ.get('MSO_BRANCH_CKPT', '0') == '1':
            # recompute prefix + chunk in backward instead of keeping activations (memory ~ one chunk).
            # The prefix pass MUST be inside the checkpoint so the recomputation rebuilds cache0 from
            # the same input tensors, keeping the gradient chain to LoRA params intact. If the prefix
            # pass were outside (as a closure-captured cache0), checkpoint's recomputation would use
            # the stale outer cache0 whose autograd graph is already freed, breaking gradient flow.
            def _chunk_h_ckpt(ids_, att_, pos_, cpos_, pids_, patt_, ppos_):
                pkw = {k: v for k, v in kw.items() if k != 'input_ids' and k != 'attention_mask' and k != 'position_ids'}
                pkw['input_ids'] = pids_
                pkw['attention_mask'] = patt_
                if prefix_pos is not None:
                    pkw['position_ids'] = ppos_
                pout = backbone(**pkw, logits_to_keep=1)
                c0 = pout.past_key_values
                o_ = backbone(
                    input_ids=ids_,
                    attention_mask=att_,
                    past_key_values=expand_cache(c0, ids_.shape[0]),
                    use_cache=True,
                    output_hidden_states=True,
                    cache_position=cpos_,
                    position_ids=pos_,
                    logits_to_keep=1)
                return o_.hidden_states[-1]
            h = torch.utils.checkpoint.checkpoint(
                _chunk_h_ckpt, ids, full_att, pos, cache_pos, prefix_ids, torch.ones_like(prefix_ids),
                prefix_pos if prefix_pos is not None else torch.zeros(3, 1, 1, device=dev, dtype=torch.long),
                use_reentrant=False)
        else:
            def _chunk_h(ids_, att_, pos_, cpos_):
                o_ = backbone(
                    input_ids=ids_,
                    attention_mask=att_,
                    past_key_values=expand_cache(cache0, ids_.shape[0]),
                    use_cache=True,
                    output_hidden_states=True,
                    cache_position=cpos_,
                    position_ids=pos_,
                    logits_to_keep=1)
                return o_.hidden_states[-1]
            h = _chunk_h(ids, full_att, pos, cache_pos)  # [Rc, n, D]
        marks = [_marks(r, open_ids, close_ids) for r in sub]
        ar = torch.arange(Rc, device=dev)
        u = h[ar, torch.tensor([m[1] for m in marks], device=dev)]
        zq = h[ar, torch.tensor([max(0, m[0] - 1) for m in marks], device=dev)]
        us.append(u)
        zqs.append(zq)
        if lm_head is not None:
            pred_i, pred_t, tgt_tok, owner, firsts = [], [], [], [], []
            for i, (r, (o_, c_, inner)) in enumerate(zip(sub, marks)):
                firsts.append(r[inner[0]] if inner else -1)
                for t in inner:
                    pred_i.append(i)
                    pred_t.append(t - 1)
                    tgt_tok.append(r[t])
                    owner.append(i)
            sums = torch.zeros(Rc, device=dev)
            cnts = torch.zeros(Rc, device=dev)
            if pred_i:
                hp = h[torch.tensor(pred_i, device=dev), torch.tensor(pred_t, device=dev)]
                lp = torch.log_softmax(lm_head(hp).float(), dim=-1)
                picked = lp.gather(1, torch.tensor(tgt_tok, device=dev)[:, None]).squeeze(1)
                own = torch.tensor(owner, device=dev)
                sums = sums.index_add(0, own, picked)
                cnts = cnts.index_add(0, own, torch.ones_like(picked))
            first = torch.zeros(Rc, device=dev)
            fi = [i for i, f in enumerate(firsts) if f >= 0]
            if fi:
                lq = torch.log_softmax(lm_head(zq[fi]).float(), dim=-1)
                first[fi] = lq[torch.arange(len(fi), device=dev), torch.tensor([firsts[i] for i in fi], device=dev)]
            sums_l.append(sums)
            cnts_l.append(cnts)
            first_l.append(first)
    u, zq = torch.cat(us), torch.cat(zqs)
    feats = None
    if lm_head is not None:
        sums, cnts, first = torch.cat(sums_l), torch.cat(cnts_l), torch.cat(first_l)
        valid = cnts > 0
        feats = torch.zeros(R, N_FEAT, device=dev, dtype=torch.float32)
        masked = torch.where(valid, sums, torch.full_like(sums, -1e4))
        dist = torch.zeros(R, device=dev)
        for g in (groups or [list(range(R))]):
            gi = torch.tensor(g, device=dev)
            dist[gi] = torch.log_softmax(masked[gi], dim=0)
        feats[:, 0] = sums / 10.0
        feats[:, 1] = torch.where(valid, sums / cnts.clamp(min=1), torch.zeros_like(sums))
        feats[:, 2] = cnts / 10.0
        feats[:, 3] = dist
        feats[:, 4] = first
        feats[:, 5] = valid.float()
    return {'u': u, 'zq': zq, 'feats': feats}


def branch_questions(backbone, penc: Dict[str, Any], qrows: List[List[List[int]]], dev, open_ids: Sequence[int],
                     close_ids: Sequence[int], lm_head=None, prefix_pos: Optional[torch.Tensor] = None,
                     chunk: Optional[int] = None, pad_id: int = 0):
    """`qrows`: per question, its rows -> per question `(u [K, D] float32, zq [D] float32, feats [K, 6] |
    None)`. All questions of the batch ride ONE prefix pass + one chunked batched suffix pass."""
    rows: List[List[int]] = []
    groups: List[List[int]] = []
    for qr in qrows:
        groups.append(list(range(len(rows), len(rows) + len(qr))))
        rows.extend(qr)
    if not rows:
        return []
    br = branch_forward(
        backbone, penc, rows, dev, open_ids, close_ids, lm_head=lm_head, prefix_pos=prefix_pos, groups=groups,
        chunk=chunk, pad_id=pad_id)
    out = []
    for g in groups:
        gi = torch.tensor(g, device=dev)
        out.append((br['u'][gi].float(), br['zq'][g[0]].float(), (br['feats'][gi] if br['feats'] is not None else None)))
    return out


def add_option_tokens(model, proc, emb_path: Optional[str] = None) -> List[int]:
    """Register `<|opt|>` / `<|/opt|>` and give them DETERMINISTIC embeddings (port of
    `mso/records.py::add_option_tokens`). `resize_token_embeddings` fills new rows randomly; those rows are
    frozen in training (only LoRA + head train), so a fresh random draw at eval time would silently change
    the very positions the head reads. We init them to the mean embedding and, when `new_tok_emb.pt` exists,
    restore it exactly. Returns the two token ids `[open_id, close_id]`."""
    proc.tokenizer.add_special_tokens({'additional_special_tokens': [OPT_OPEN, OPT_CLOSE]})
    n_old = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(proc.tokenizer))
    emb = model.get_input_embeddings().weight
    ids = proc.tokenizer.convert_tokens_to_ids([OPT_OPEN, OPT_CLOSE])
    with torch.no_grad():
        if emb_path and os.path.exists(emb_path):
            saved = torch.load(emb_path, map_location=emb.device)
            for tid, row in zip(ids, saved):
                emb[tid] = row.to(emb.dtype)
        else:
            mean = emb[:n_old].float().mean(0)
            for k, tid in enumerate(ids):
                emb[tid] = (mean + 0.01 * (k + 1)).to(emb.dtype)
    return ids
