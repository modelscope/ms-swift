# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sections:
  * shared cache primitives (`expand_cache`, `grad_ckpt_disabled`);
  * OmniJev (decision G1): OPTION-level branching -- one encoded single-question sequence is cut at
    forward time into a prefix + K option-block rows (`branch_forward` / `branch_questions`);
  * Kev: QUESTION-level branching -- batch rows sharing an EXACT state prefix are grouped, the
    prefix is forwarded once and each question's branch rides the expanded cache
    (`kev_forward_branch`), restoring the official "state computed once" compute shape.
"""
import copy
import os
import torch
import torch.utils.checkpoint
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple


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
    new.update_recurrent_state = lambda *a, **k: None
    return new


@contextmanager
def grad_ckpt_disabled(backbone_model):
    """Temporarily switch off gradient checkpointing: HF forces `use_cache=False` when
    `gradient_checkpointing=True` and `training=True`, but a branch prefix pass must return a
    non-None cache"""
    touched = [m for m in backbone_model.modules() if getattr(m, 'gradient_checkpointing', False)]
    for m in touched:
        m.gradient_checkpointing = False
    try:
        yield
    finally:
        for m in touched:
            m.gradient_checkpointing = True


def is_hybrid(model) -> bool:
    cfg = getattr(model, 'config', None)
    tc = getattr(cfg, 'text_config', cfg)
    lt = getattr(tc, 'layer_types', None)
    return bool(lt) and any('linear' in str(t) for t in lt)


OPT_OPEN = '<|opt|>'
OPT_CLOSE = '<|/opt|>'
N_FEAT = 6
_PIX_KEYS = ('pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw', 'mm_token_type_ids')


def find_rope_owner(m, depth: int = 0):
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


def prefix_inputs(enc: Dict[str, Any], L: int) -> Dict[str, Any]:
    ids = enc['input_ids'][:, :L]
    p = {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
    for k in _PIX_KEYS:
        if k in enc:
            p[k] = enc[k][:, :L] if k == 'mm_token_type_ids' else enc[k]
    return p


def rows_from_ids(ids, L: int, opens: Sequence[int], closes: Sequence[int], k: Optional[int] = None) -> List[List[int]]:
    ids = list(ids)
    if not opens or not closes:
        return [ids[L:]]
    head = ids[L:opens[0]]
    pairs = [(o, c) for o, c in zip(opens, closes) if c >= o]
    if k is not None:
        pairs = pairs[:k]
    return [head + ids[o:c + 1] for o, c in pairs]


def _marks(r: Sequence[int], open_ids: Sequence[int], close_ids: Sequence[int]):
    ol, cl = len(open_ids), len(close_ids)
    r = list(r)
    opens = [k for k in range(len(r) - ol + 1) if r[k:k + ol] == list(open_ids)]
    closes = [k for k in range(len(r) - cl + 1) if r[k:k + cl] == list(close_ids)]
    o_ = opens[0] if opens else 0
    c_ = (closes[-1] + cl - 1) if closes else len(r) - 1
    return o_, c_, list(range(o_ + ol, c_ - cl + 1))


def branch_forward(backbone,
                   penc: Dict[str, Any],
                   rows: List[List[int]],
                   dev,
                   open_ids: Sequence[int],
                   close_ids: Sequence[int],
                   lm_head=None,
                   prefix_pos: Optional[torch.Tensor] = None,
                   pos_start: Optional[int] = None,
                   groups: Optional[List[List[int]]] = None,
                   chunk: Optional[int] = None,
                   pad_id: int = 0,
                   mrope_axes: int = 3) -> Dict[str, Any]:
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

            def _chunk_h_ckpt(ids_, att_, pos_, cpos_, pids_, patt_, ppos_):
                pkw = {
                    k: v
                    for k, v in kw.items() if k != 'input_ids' and k != 'attention_mask' and k != 'position_ids'
                }
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
                _chunk_h_ckpt,
                ids,
                full_att,
                pos,
                cache_pos,
                prefix_ids,
                torch.ones_like(prefix_ids),
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


def branch_questions(backbone,
                     penc: Dict[str, Any],
                     qrows: List[List[List[int]]],
                     dev,
                     open_ids: Sequence[int],
                     close_ids: Sequence[int],
                     lm_head=None,
                     prefix_pos: Optional[torch.Tensor] = None,
                     chunk: Optional[int] = None,
                     pad_id: int = 0):
    rows: List[List[int]] = []
    groups: List[List[int]] = []
    for qr in qrows:
        groups.append(list(range(len(rows), len(rows) + len(qr))))
        rows.extend(qr)
    if not rows:
        return []
    br = branch_forward(
        backbone,
        penc,
        rows,
        dev,
        open_ids,
        close_ids,
        lm_head=lm_head,
        prefix_pos=prefix_pos,
        groups=groups,
        chunk=chunk,
        pad_id=pad_id)
    out = []
    for g in groups:
        gi = torch.tensor(g, device=dev)
        out.append(
            (br['u'][gi].float(), br['zq'][g[0]].float(), (br['feats'][gi] if br['feats'] is not None else None)))
    return out


def add_option_tokens(model, proc, emb_path: Optional[str] = None) -> List[int]:
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


def kev_branch_enabled() -> bool:
    return os.environ.get('KEV_BRANCH', '1') == '1'


def group_rows_by_prefix(reals: Sequence[List[int]],
                         branch_idx: Sequence[int]) -> 'OrderedDict[Tuple[int, ...], List[int]]':
    groups: OrderedDict[Tuple[int, ...], List[int]] = OrderedDict()
    for r, (real, bi) in enumerate(zip(reals, branch_idx)):
        groups.setdefault(tuple(real[:bi]), []).append(r)
    return groups


def kev_branch_hidden(backbone,
                      prefix_ids: List[int],
                      suffixes: List[List[int]],
                      gather: Sequence[Tuple[int, List[int]]],
                      dev,
                      chunk: Optional[int] = None,
                      pad_id: int = 0) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    L = len(prefix_ids)
    pids = torch.tensor([prefix_ids], device=dev, dtype=torch.long)
    kw: Dict[str, Any] = {
        'input_ids': pids,
        'attention_mask': torch.ones_like(pids),
        'use_cache': True,
        'logits_to_keep': 1
    }
    out = backbone(**kw)
    cache0 = out.past_key_values
    R = len(suffixes)
    chunk = chunk or int(os.environ.get('KEV_BRANCH_CHUNK', '64'))
    feats: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for s in range(0, R, chunk):
        sub = suffixes[s:s + chunk]
        pos_list = gather[s:s + chunk]
        Rc = len(sub)
        n = max(len(r) for r in sub)
        ids = torch.full((Rc, n), pad_id, device=dev, dtype=torch.long)
        att = torch.zeros((Rc, n), dtype=torch.long, device=dev)
        for i, r in enumerate(sub):
            ids[i, :len(r)] = torch.tensor(r, device=dev)
            att[i, :len(r)] = 1
        full_att = torch.cat([torch.ones((Rc, L), dtype=torch.long, device=dev), att], dim=1)
        pos = torch.arange(L, L + n, device=dev)[None, None, :].expand(3, Rc, n).contiguous()
        cache_pos = torch.arange(L, L + n, device=dev)

        if torch.is_grad_enabled() and os.environ.get('KEV_BRANCH_CKPT', '0') == '1':

            def _chunk_h_ckpt(ids_, att_, pos_, cpos_, pids_, patt_):
                pout = backbone(input_ids=pids_, attention_mask=patt_, use_cache=True, logits_to_keep=1)
                o_ = backbone(
                    input_ids=ids_,
                    attention_mask=att_,
                    past_key_values=expand_cache(pout.past_key_values, ids_.shape[0]),
                    use_cache=True,
                    output_hidden_states=True,
                    cache_position=cpos_,
                    position_ids=pos_,
                    logits_to_keep=1)
                return o_.hidden_states[-1]

            h = torch.utils.checkpoint.checkpoint(
                _chunk_h_ckpt, ids, full_att, pos, cache_pos, pids, torch.ones_like(pids), use_reentrant=False)
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
        for i, (dp, ops) in enumerate(pos_list):
            h_decide = h[i, dp]
            h_opts = h[i, torch.tensor(ops, device=dev)] if ops else h.new_zeros((0, h.shape[-1]))
            feats.append((h_decide.float(), h_opts.float()))
    return feats


def kev_forward_branch(origin_forward,
                       head,
                       backbone_model,
                       kwargs: Dict[str, Any],
                       meta: Dict[str, Any],
                       pad_id: int = 0):
    input_ids = kwargs.get('input_ids')
    if input_ids is None or input_ids.dim() != 2:
        return None
    branch_idx = meta.get('branch_idx')
    B = int(input_ids.shape[0])
    if B < 2 or not branch_idx or len(branch_idx) != B:
        return None
    decide_idx = meta['decide_idx']
    opt_idx = meta['opt_idx']
    if len(decide_idx) != B or len(opt_idx) != B:
        return None
    attention_mask = kwargs.get('attention_mask')
    T = int(input_ids.shape[1])
    left_padding = bool(meta.get('left_padding', False))
    seq_lens = meta.get('seq_lens')
    if not seq_lens:
        seq_lens = attention_mask.sum(-1).tolist() if attention_mask is not None else [T] * B
    dev = input_ids.device

    reals: List[List[int]] = []
    for r in range(B):
        sl = int(seq_lens[r])
        real = input_ids[r, T - sl:].tolist() if left_padding else input_ids[r, :sl].tolist()
        bi, di = int(branch_idx[r]), int(decide_idx[r])
        if not 0 < bi <= di < len(real) or any(not bi <= int(o) < len(real) for o in opt_idx[r]):
            return None
        reals.append(real)

    groups = group_rows_by_prefix(reals, branch_idx)
    shared = [(k, v) for k, v in groups.items() if len(v) >= 2]
    if not shared:
        return None

    features: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * B
    with grad_ckpt_disabled(backbone_model):
        for key, rows in shared:
            prefix = list(key)
            suffixes = [reals[r][int(branch_idx[r]):] for r in rows]
            gather = [(int(decide_idx[r]) - int(branch_idx[r]), [int(o) - int(branch_idx[r]) for o in opt_idx[r]])
                      for r in rows]
            feats = kev_branch_hidden(origin_forward, prefix, suffixes, gather, dev, pad_id=pad_id)
            for r, f in zip(rows, feats):
                features[r] = f
        singles = [r for r in range(B) if features[r] is None]
        if singles:
            sub_ids = input_ids[singles]
            sub_att = attention_mask[singles] if attention_mask is not None else None
            out = origin_forward(
                input_ids=sub_ids,
                attention_mask=sub_att,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
                logits_to_keep=1)
            hidden = out.hidden_states[-1]
            padded_len = int(sub_ids.shape[1])
            for sub_i, r in enumerate(singles):
                sl = int(seq_lens[r])
                offset = (padded_len - sl) if left_padding else 0
                features[r] = (hidden[sub_i, offset + int(decide_idx[r])].float(),
                               hidden[sub_i, [offset + int(o) for o in opt_idx[r]]].float())
    return head.forward_pointer(features, meta)
