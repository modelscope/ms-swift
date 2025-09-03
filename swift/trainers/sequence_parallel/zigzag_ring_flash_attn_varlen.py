import inspect
from functools import cache
import torch
import torch.distributed as dist
import torch.nn.functional as F
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)
from .utils import (
    RingComm,
)


def get_half_index(cu_seqlens, *, front: bool):
    if len(cu_seqlens) == 2:
        if front:
            return slice(None, cu_seqlens[-1] // 2)
        else:
            return slice(cu_seqlens[-1] // 2, None)

    index = torch.zeros((cu_seqlens[-1].item(),), dtype=torch.bool)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        if front:
            end = (start + end) // 2
        else:
            start = (start + end) // 2
        index[start:end] = True
    return index


@torch.jit.script
def get_half_lse(lse, cu_seqlens, *, front: bool):
    new_lse = torch.empty(
        (lse.shape[0], lse.shape[1] // 2),
        dtype=lse.dtype,
        device=lse.device,
    )
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        new_start, new_end = start // 2, end // 2
        if front:
            end -= (end - start) // 2
        else:
            start += (end - start) // 2
        new_lse[:, new_start:new_end] = lse[:, start:end]
    return new_lse


def update_out_and_lse(out, lse, block_out, block_lse):
    if out is None:
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
        sig_diff = None
    else:
        block_out = block_out.to(torch.float32)
        block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

        diff = block_lse - lse
        sig_diff = torch.sigmoid(diff)

        out = out - sig_diff * (out - block_out)  # (..., D)
        lse = lse - F.logsigmoid(lse - block_lse)  # (..., 1)
    return out, lse, sig_diff


@cache
def _get_default_args(func):
    spec = inspect.getfullargspec(func)
    defaults = spec.defaults if spec.defaults is not None else ()
    padded_defaults = (None,) * (len(spec.args) - len(defaults)) + defaults
    args = dict(zip(spec.args, padded_defaults))
    if "softcap" in args:
        args["softcap"] = 0.0
    return args


def get_default_args(func):
    if inspect.isfunction(func):
        return _get_default_args(func)
    else:
        # Use the origin _init_fn in CustomOpDef
        return _get_default_args(func._init_fn)


def squeeze_batch(*t):
    tensors = []
    for sub in t:
        if sub.shape[0] == 1:
            tensors.append(sub.squeeze(0))
        else:
            tensors.append(sub)
    return tuple(tensors)


def forward(q, k, v, causal, cu_seqlens, max_seqlen, block_seq_len, dropout_p, softmax_scale, alibi_slopes, window_size):
    seqlen_q = q.shape[0]
    seqlen_kv = k.shape[0]
    half_cu_seqlens = cu_seqlens // 2
    half_max_seqlen = max_seqlen // 2
    cu_seqlens_q = half_cu_seqlens if seqlen_q == block_seq_len else cu_seqlens
    max_seqlen_q = half_max_seqlen if seqlen_q == block_seq_len else max_seqlen
    cu_seqlens_kv = half_cu_seqlens if seqlen_kv == block_seq_len else cu_seqlens
    max_seqlen_kv = half_max_seqlen if seqlen_kv == block_seq_len else max_seqlen

    params = get_default_args(_flash_attn_varlen_forward).copy()
    params.update(
        {
            "q": q,
            "k": k,
            "v": v,
            # the first half and the second half are the same
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_kv,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_kv,
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "alibi_slopes": alibi_slopes,
            "return_softmax": True and dropout_p > 0,
        }
    )
    if "window_size" in params:
        params.update({"window_size": window_size})
    else:
        params.update(
            {
                "window_size_left": window_size[0],
                "window_size_right": window_size[1],
            }
        )
    outputs = _flash_attn_varlen_forward(**params)
    if len(outputs) == 8:
        block_out, _, _, _, _, block_lse, _, _ = outputs
    else:
        assert len(outputs) == 4
        block_out, block_lse, _, _ = outputs
    return block_out, block_lse


def backward(dout, q, k, v, out, softmax_lse, causal, cu_seqlens, max_seqlen, block_seq_len, dq_buffer, dk_buffer, dv_buffer, dropout_p, softmax_scale, alibi_slopes, deterministic, window_size):
    seqlen_q = q.shape[0]
    seqlen_kv = k.shape[0]

    half_cu_seqlens = cu_seqlens // 2
    half_max_seqlen = max_seqlen // 2
    cu_seqlens_q = half_cu_seqlens if seqlen_q == block_seq_len else cu_seqlens
    max_seqlen_q = half_max_seqlen if seqlen_q == block_seq_len else max_seqlen
    cu_seqlens_kv = half_cu_seqlens if seqlen_kv == block_seq_len else cu_seqlens
    max_seqlen_kv = half_max_seqlen if seqlen_kv == block_seq_len else max_seqlen
    params = get_default_args(_flash_attn_varlen_backward).copy()
    params.update(
        {
            "dout": dout,
            "q": q,
            "k": k,
            "v": v,
            "out": out,
            "softmax_lse": softmax_lse,
            "dq": dq_buffer[:seqlen_q],
            "dk": dk_buffer[:seqlen_kv],
            "dv": dv_buffer[:seqlen_kv],
            # the first half and the second half are the same
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_kv,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_kv,
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "alibi_slopes": alibi_slopes,
            "deterministic": deterministic,
        }
    )
    if "window_size" in params:
        params.update({"window_size": window_size})
    else:
        params.update(
            {
                "window_size_left": window_size[0],
                "window_size_right": window_size[1],
            }
        )
    _flash_attn_varlen_backward(**params)


def lse_grad(out, block_out, lse, block_lse, sig, grad_out, grad_lse):
    # wrt out
    grad_out_input = grad_out * (1 - sig)

    # wrt block_out
    grad_block_out = grad_out * sig

    # wrt lse (注意 sum over last dim)
    d_new_out_d_lse = (out - block_out) * (sig * (1 - sig))  # (..., D)
    grad_lse_input = (grad_out * d_new_out_d_lse).sum(dim=-1, keepdim=True)
    grad_lse_input = grad_lse_input + grad_lse * torch.sigmoid(lse - block_lse)

    # wrt block_lse (同样 sum over last dim)
    d_new_out_d_block_lse = -(out - block_out) * (sig * (1 - sig))  # (..., D)
    grad_block_lse = (grad_out * d_new_out_d_block_lse).sum(dim=-1, keepdim=True)
    grad_block_lse = grad_block_lse + grad_lse * (1 - torch.sigmoid(lse - block_lse))

    return grad_out_input, grad_lse_input, grad_block_out, grad_block_lse


def zigzag_ring_flash_attn_varlen_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens,
    max_seqlen,
    half_index0,
    half_index1,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert causal == True, "zigzag ring is meaningless for causal=False"
    comm = RingComm(process_group)
    q = q.squeeze(0)
    k = k.squeeze(0)
    v = v.squeeze(0)
    q1 = q[half_index1]
    cu_seqlens = cu_seqlens // comm.world_size
    max_seqlen = max_seqlen // comm.world_size
    block_seq_len = q.shape[0] // 2
    out = None
    lse = None
    next_k, next_v = None, None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)

        if step == 0:
            block_out, block_lse = forward(q, k, v, True, cu_seqlens, max_seqlen, block_seq_len, dropout_p, softmax_scale, alibi_slopes, window_size)
            out, lse, sig_diff = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.rank:
            k0 = k[half_index0]
            v0 = v[half_index0]
            block_out, block_lse = forward(q, k0, v0, False, cu_seqlens, max_seqlen, block_seq_len, dropout_p, softmax_scale, alibi_slopes, window_size)
            out, lse, sig_diff = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, block_lse = forward(q1, k, v, False, cu_seqlens, max_seqlen, block_seq_len, dropout_p, softmax_scale, alibi_slopes, window_size)
            out[half_index1], lse[half_index1], sig_diff = update_out_and_lse(
                out[half_index1], lse[half_index1], block_out, block_lse
            )

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(0, 1)
    return out.unsqueeze(0), lse.unsqueeze(0)


def zigzag_ring_flash_attn_varlen_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    cu_seqlens,
    max_seqlen,
    half_index0,
    half_index1,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert causal == True, "zigzag ring is meaningless for causal=False"
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    dout, q, k, v, out, softmax_lse = squeeze_batch(dout, q, k, v, out, softmax_lse)
    q1 = q[half_index1]
    block_seq_len = q.shape[0] // 2

    # repeatly allocating buffer may be slow...
    dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    out_lse = []
    fout = None
    flse = None
    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        if step == 0:
            block_out, block_lse = forward(q, k, v, True, cu_seqlens, max_seqlen, block_seq_len, dropout_p, softmax_scale, alibi_slopes, window_size)
            fout, flse, sig_diff = update_out_and_lse(fout, flse, block_out, block_lse)
        elif step <= kv_comm.rank:
            k0 = k[half_index0]
            v0 = v[half_index0]
            block_out, block_lse = forward(q, k0, v0, False, cu_seqlens, max_seqlen, block_seq_len, dropout_p, softmax_scale, alibi_slopes, window_size)
            fout, flse, sig_diff = update_out_and_lse(fout, flse, block_out, block_lse)
        else:
            block_out, block_lse = forward(q1, k, v, False, cu_seqlens, max_seqlen, block_seq_len, dropout_p, softmax_scale, alibi_slopes, window_size)
            fout[half_index1], flse[half_index1], sig_diff = update_out_and_lse(
                fout[half_index1], flse[half_index1], block_out, block_lse
            )

        block_lse = block_lse.transpose(0, 1).unsqueeze(-1)
        if fout is not None and block_out.shape[0] * 2 == fout.shape[0]:
            block_out = torch.cat((torch.zeros_like(block_out).to(block_out.dtype).to(block_out.device), block_out))
            block_lse = torch.cat((torch.zeros_like(block_lse).to(block_lse.dtype).to(block_lse.device), block_lse))
            sig_diff = torch.cat((torch.zeros_like(sig_diff).to(sig_diff.dtype).to(sig_diff.device), sig_diff))
        out_lse.append((fout, flse, block_out, block_lse, sig_diff))

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v

    current_dout = dout
    current_dlse = torch.zeros_like(softmax_lse.transpose(0, 1).unsqueeze(-1))
    block_gradients = {}

    for i in reversed(range(len(out_lse))):
        if i == 0:
            continue
        stored_out, stored_lse, stored_block_out, stored_block_lse, stored_sig = out_lse[i]
        grad_out_input, grad_lse_input, grad_block_out, grad_block_lse = lse_grad(stored_out,
                                                                                  stored_block_out,
                                                                                  stored_lse,
                                                                                  stored_block_lse,
                                                                                  stored_sig,
                                                                                  current_dout,
                                                                                  current_dlse)
        current_dout = grad_out_input
        current_dlse = grad_lse_input
        block_gradients[i] = {
            'grad_block_out': grad_block_out,
            'grad_block_lse': grad_block_lse
        }

    for step in range(kv_comm.world_size):
        _, _, block_out, block_lse, _ = out_lse[step]
        block_lse = block_lse.transpose(0, 1).squeeze(2)

        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        if step == 0:
            block_dout = current_dout
        else:
            block_dout = block_gradients[step]['grad_block_out']

        if step == 0:
            backward(block_dout.to(dout.dtype), q, k, v, block_out, block_lse, True, cu_seqlens, max_seqlen, block_seq_len,
                     dq_buffer, dk_buffer, dv_buffer, dropout_p, softmax_scale, alibi_slopes, deterministic, window_size)
            dq = dq_buffer.to(torch.float32)
            dk = dk_buffer.to(torch.float32)
            dv = dv_buffer.to(torch.float32)
        else:
            if step <= kv_comm.rank:
                k0 = k[half_index0]
                v0 = v[half_index0]
                backward(block_dout.to(dout.dtype), q, k0, v0, block_out, block_lse, False, cu_seqlens, max_seqlen, block_seq_len,
                     dq_buffer, dk_buffer, dv_buffer, dropout_p, softmax_scale, alibi_slopes, deterministic, window_size)
                dq += dq_buffer
            else:
                backward(block_dout[half_index1].to(dout.dtype), q1, k, v, block_out[half_index1],
                         get_half_lse(block_lse, cu_seqlens // kv_comm.world_size, front=False), False, cu_seqlens, max_seqlen, block_seq_len,
                     dq_buffer, dk_buffer, dv_buffer, dropout_p, softmax_scale, alibi_slopes, deterministic, window_size)
                dq[half_index1] += dq_buffer[:block_seq_len]

            d_kv_comm.wait()
            # dk_comm_buffer, dv_comm_buffer = dk, dv

            dk_comm_buffer = torch.empty_like(dk)
            dv_comm_buffer = torch.empty_like(dv)
            dk, dv = next_dk, next_dv

            if step <= kv_comm.rank:
                dk[half_index0] += dk_buffer[:block_seq_len]
                dv[half_index0] += dv_buffer[:block_seq_len]
            else:
                dk += dk_buffer
                dv += dv_buffer

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v

        next_dk, next_dv = d_kv_comm.send_recv_kv(
            dk, dv, dk_comm_buffer, dv_comm_buffer
        )

    d_kv_comm.wait()

    return dq.to(q.dtype).unsqueeze(0), next_dk.to(q.dtype).unsqueeze(0), next_dv.to(q.dtype).unsqueeze(0)


class ZigZagRingFlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        rp_world_size = dist.get_world_size(group)
        half_index0 = get_half_index(cu_seqlens // rp_world_size, front=True)
        half_index1 = get_half_index(cu_seqlens // rp_world_size, front=False)
        out, softmax_lse = zigzag_ring_flash_attn_varlen_forward(
            group,
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            half_index0,
            half_index1,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        # this should be out_padded
        is_half_index_tensor = isinstance(half_index0, torch.Tensor)
        ctx.is_half_index_tensor = is_half_index_tensor
        if is_half_index_tensor:
            ctx.save_for_backward(
                q, k, v, out, softmax_lse, cu_seqlens, half_index0, half_index1
            )
        else:
            ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens)
            ctx.half_index0 = half_index0
            ctx.half_index1 = half_index1
        ctx.max_seqlen = max_seqlen
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        if ctx.is_half_index_tensor:
            (q, k, v, out, softmax_lse, cu_seqlens, half_index0, half_index1) = (
                ctx.saved_tensors
            )
        else:
            q, k, v, out, softmax_lse, cu_seqlens = ctx.saved_tensors
            half_index0 = ctx.half_index0
            half_index1 = ctx.half_index1
        dq, dk, dv = zigzag_ring_flash_attn_varlen_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens,
            ctx.max_seqlen,
            half_index0,
            half_index1,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


def zigzag_ring_flash_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return ZigZagRingFlashAttnVarlenFunc.apply(
        qkv[:, 0],
        qkv[:, 1],
        qkv[:, 2],
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def zigzag_ring_flash_attn_varlen_kvpacked_func(
    q,
    kv,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return ZigZagRingFlashAttnVarlenFunc.apply(
        q,
        kv[:, 0],
        kv[:, 1],
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def zigzag_ring_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return ZigZagRingFlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )
