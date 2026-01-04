import math
import threading
from contextlib import suppress
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import Optimizer


class _MaxLogitsTracker:
    """
    Collect a per-step scalar max logits value even when training loop can't pass it into optimizer.step().

    - Eager attention: patch torch.softmax / F.softmax to capture exact softmax input max (attention scores).
    - SDPA / FlashAttention: logits not exposed; record conservative upper bound via norms:
        max(qk^T * scale) <= max||q|| * max||k|| * scale

    Note: This is a GLOBAL scalar for the whole step (not per-layer, not per-head).
    """

    _tls = threading.local()
    _enabled = False
    _patched_softmax = False
    _patched_sdpa = False
    _patched_flash = False

    _orig_torch_softmax = None
    _orig_F_softmax = None
    _orig_sdpa = None
    _orig_flash_attn_func = None

    @classmethod
    def _get_and_reset(cls) -> Optional[float]:
        v = getattr(cls._tls, 'max_logits', None)
        cls._tls.max_logits = None
        return v

    @classmethod
    def _update(cls, v: float):
        if v is None:
            return
        cur = getattr(cls._tls, 'max_logits', None)
        if cur is None or v > cur:
            cls._tls.max_logits = float(v)

    @classmethod
    def enable_softmax(cls):
        if cls._patched_softmax:
            return
        cls._patched_softmax = True

        cls._orig_torch_softmax = torch.softmax
        cls._orig_F_softmax = F.softmax

        def _maybe_capture(x: torch.Tensor, dim):
            # attention scores softmax: usually [B,H,Lq,Lk], dim=-1
            if not isinstance(x, torch.Tensor):
                return
            if x.dim() != 4:
                return
            if dim is None or not (dim == -1 or dim == x.dim() - 1):
                return

            with suppress(Exception):
                cls._update(float(x.detach().float().amax().item()))

        def _torch_softmax(x, dim=None, dtype=None):
            with suppress(Exception):
                _maybe_capture(x, dim)
            return cls._orig_torch_softmax(x, dim=dim, dtype=dtype)

        def _F_softmax(x, dim=None, _stacklevel=3, dtype=None):
            with suppress(Exception):
                _maybe_capture(x, dim)
            return cls._orig_F_softmax(x, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

        torch.softmax = _torch_softmax
        F.softmax = _F_softmax

    @classmethod
    def enable_sdpa(cls):
        if cls._patched_sdpa:
            return
        cls._patched_sdpa = True

        if not hasattr(F, 'scaled_dot_product_attention'):
            return

        cls._orig_sdpa = F.scaled_dot_product_attention

        def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
            with suppress(Exception):
                if isinstance(query, torch.Tensor) and isinstance(key, torch.Tensor):
                    q = query.detach()
                    k = key.detach()

                    # upper bound using vector norms
                    qn = q.float().norm(p=2, dim=-1).max().item()
                    kn = k.float().norm(p=2, dim=-1).max().item()
                    d = q.size(-1)
                    s = float(scale) if scale is not None else (1.0 / math.sqrt(float(d)))
                    cls._update(qn * kn * s)

            return cls._orig_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

        F.scaled_dot_product_attention = _sdpa

    @classmethod
    def enable_flash_attn(cls):
        if cls._patched_flash:
            return
        cls._patched_flash = True

        try:
            import flash_attn.flash_attn_interface as _fai
            flash_attn_func = _fai.flash_attn_func
        except Exception:
            return

        cls._orig_flash_attn_func = flash_attn_func

        def _flash_attn(q,
                        k,
                        v,
                        dropout_p=0.0,
                        softmax_scale=None,
                        causal=False,
                        window_size=(-1, -1),
                        alibi_slopes=None,
                        deterministic=False,
                        return_attn_probs=False):
            with suppress(Exception):
                if isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor):
                    qn = q.detach().float().norm(p=2, dim=-1).max().item()
                    kn = k.detach().float().norm(p=2, dim=-1).max().item()
                    d = q.size(-1)
                    s = float(softmax_scale) if softmax_scale is not None else (1.0 / math.sqrt(float(d)))
                    cls._update(qn * kn * s)

            return cls._orig_flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
            )

        _fai.flash_attn_func = _flash_attn

    @classmethod
    def enable_all(cls):
        if cls._enabled:
            return
        cls._enabled = True
        cls.enable_softmax()
        cls.enable_sdpa()
        cls.enable_flash_attn()

    @classmethod
    def consume(cls) -> Optional[float]:
        return cls._get_and_reset()


class MuonClip(Optimizer):
    """
    MuonClip (stable version):
      - Muon-style update for apply_muon=True (2D weights): momentum buffer + Moonlight polynomial NS orthogonalization.
      - Other params (apply_muon=False): simple momentum SGD (kept minimal; you can switch to AdamW if needed).
      - QK-Clip uses a scalar max_logits (exact in eager, upper bound in sdpa/flash) and applies gamma_sqrt scaling
        to Q/K weights marked with is_qk=True.
    """

    def __init__(
        self,
        params,
        lr: float = 2e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        nesterov: bool = False,
        newton_schulz_steps: int = 5,
        qk_clip_tau: float = 10000.0,
        qk_clip_enabled: bool = True,
        rms_scale_factor: float = 0.2,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            newton_schulz_steps=newton_schulz_steps,
            qk_clip_tau=qk_clip_tau,
            qk_clip_enabled=qk_clip_enabled,
            rms_scale_factor=rms_scale_factor,
        )
        super().__init__(params, defaults)
        _MaxLogitsTracker.enable_all()

    @staticmethod
    @torch.no_grad()
    def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
        """
        Moonlight/Muon polynomial Newton-Schulz iteration (stable).
        Works for rectangular matrices by transposing when needed.
        """
        # constants from your previous stable implementation
        a, b, c = (3.4445, -4.7750, 2.0315)

        X = G.bfloat16() / (G.norm() + eps)
        transposed = False
        if G.size(0) > G.size(1):
            X = X.T
            transposed = True

        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

        if transposed:
            X = X.T

        return X.to(G.dtype)

    def _is_qk_weight(self, group) -> bool:
        return bool(group.get('is_qk', False))

    @torch.no_grad()
    def step(self, closure=None, max_logits: Optional[float] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # fallback: collect scalar max_logits from tracker if not provided
        if max_logits is None:
            max_logits = _MaxLogitsTracker.consume()

        for group in self.param_groups:
            lr = float(group['lr'])
            momentum = float(group['momentum'])
            weight_decay = float(group['weight_decay'])
            nesterov = bool(group.get('nesterov', False))
            ns_steps = int(group.get('newton_schulz_steps', 5))
            qk_clip_tau = float(group.get('qk_clip_tau', 10000.0))
            qk_clip_enabled = bool(group.get('qk_clip_enabled', True))
            apply_muon = bool(group.get('apply_muon', True))
            is_qk_group = self._is_qk_weight(group)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['step'] = 0

                buf = state['momentum_buffer']
                state['step'] += 1

                buf.mul_(momentum).add_(grad)

                # build update
                if apply_muon and p.ndim >= 2:
                    orth = self.newton_schulz(buf, steps=ns_steps)
                    n, m = p.shape[0], p.shape[1]
                    rms_scale_factor = float(group.get('rms_scale_factor', 0.2))
                    rms_scale = math.sqrt(max(n, m)) * rms_scale_factor
                    update = orth * rms_scale
                else:
                    update = buf

                if nesterov:
                    update = grad.add(update, alpha=momentum)

                # decoupled-ish weight decay
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # QK-Clip (scalar)
                if qk_clip_enabled and is_qk_group and (max_logits is not None):
                    if max_logits > qk_clip_tau:
                        gamma = qk_clip_tau / float(max_logits)
                        gamma_sqrt = math.sqrt(gamma)
                        # scale weight and update (matches your previous stable version)
                        p.mul_(gamma_sqrt)
                        update = update * gamma_sqrt

                # apply update
                p.add_(update, alpha=-lr)

        return loss
