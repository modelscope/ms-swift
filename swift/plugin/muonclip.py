import torch
from torch.optim import Optimizer
import math
from typing import List, Optional

import threading
from contextlib import suppress
import torch.nn.functional as F


class _MaxLogitsTracker:
    """Collect a per-step max logits value even when the training loop can't pass it into optimizer.step().

    - Eager attention: hooks torch.softmax / F.softmax to capture the *exact* logits max (softmax input).
    - SDPA / FlashAttention: kernels don't expose logits; we record a conservative upper bound via L2 norms:
        max(qk^T * scale) <= max||q|| * max||k|| * scale
    Values are stored in thread-local storage.
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
    def _get(cls):
        return getattr(cls._tls, "max_logits", None)

    @classmethod
    def _set(cls, v: float):
        cls._tls.max_logits = float(v)

    @classmethod
    def _update(cls, v):
        if v is None:
            return
        try:
            v = float(v)
        except Exception:
            return
        cur = cls._get()
        if cur is None or v > cur:
            cls._set(v)

    @classmethod
    def consume(cls):
        v = cls._get()
        if hasattr(cls._tls, "max_logits"):
            delattr(cls._tls, "max_logits")
        return v

    @classmethod
    def enable_softmax(cls):
        if cls._patched_softmax:
            return
        cls._patched_softmax = True

        cls._orig_torch_softmax = torch.softmax
        cls._orig_F_softmax = F.softmax

        def _torch_softmax(input, dim=None, dtype=None):
            with suppress(Exception):
                if isinstance(input, torch.Tensor) and input.dim() in (3, 4):
                    if dim is None or dim == -1 or dim == input.dim() - 1:
                        # softmax input == attention logits (eager path) -> exact max
                        cls._update(input.detach().max().item())
            return cls._orig_torch_softmax(input, dim=dim, dtype=dtype)

        def _F_softmax(input, dim=None, _stacklevel=3, dtype=None):
            with suppress(Exception):
                if isinstance(input, torch.Tensor) and input.dim() in (3, 4):
                    if dim is None or dim == -1 or dim == input.dim() - 1:
                        cls._update(input.detach().max().item())
            return cls._orig_F_softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

        torch.softmax = _torch_softmax
        F.softmax = _F_softmax

    @classmethod
    def enable_sdpa(cls):
        if cls._patched_sdpa:
            return
        cls._patched_sdpa = True

        if not hasattr(F, "scaled_dot_product_attention"):
            return

        cls._orig_sdpa = F.scaled_dot_product_attention

        def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
            # SDPA kernel doesn't expose logits; record an upper bound via L2 norms
            with suppress(Exception):
                if isinstance(query, torch.Tensor) and isinstance(key, torch.Tensor):
                    q = query.detach()
                    k = key.detach()
                    qn = q.float().norm(p=2, dim=-1).max().item()
                    kn = k.float().norm(p=2, dim=-1).max().item()
                    d = q.size(-1)
                    s = float(scale) if scale is not None else (1.0 / math.sqrt(float(d)))
                    cls._update(qn * kn * s)
            return cls._orig_sdpa(
                query, key, value,
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

        def _flash_attn(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                        window_size=(-1, -1), alibi_slopes=None, deterministic=False, return_attn_probs=False):
            # FlashAttention kernel doesn't expose logits; record an upper bound via L2 norms
            with suppress(Exception):
                if isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor):
                    qn = q.detach().float().norm(p=2, dim=-1).max().item()
                    kn = k.detach().float().norm(p=2, dim=-1).max().item()
                    d = q.size(-1)
                    s = float(softmax_scale) if softmax_scale is not None else (1.0 / math.sqrt(float(d)))
                    cls._update(qn * kn * s)
            return cls._orig_flash_attn_func(
                q, k, v,
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


class MuonClip(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        newton_schulz_steps: int = 5,
        qk_clip_tau: float = 100.0,
        qk_clip_enabled: bool = True
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if qk_clip_tau <= 0.0:
            raise ValueError(f"Invalid qk_clip_tau: {qk_clip_tau}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            newton_schulz_steps=newton_schulz_steps,
            qk_clip_tau=qk_clip_tau,
            qk_clip_enabled=qk_clip_enabled
        )
        super().__init__(params, defaults)
        
        self.max_logits_history = []

        # Auto-track max_logits for Trainer/AMP loops that call optimizer.step() without args.
        if self.defaults.get('qk_clip_enabled', False):
            _MaxLogitsTracker.enable_all()
    
    def newton_schulz(self, G, steps=5, eps=1e-7):
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16() / (G.norm() + eps)
        
        if G.size(0) > G.size(1):
            X = X.T
        
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        
        if G.size(0) > G.size(1):
            X = X.T
        
        return X.to(G.dtype)
    
    @torch.no_grad()
    def step(self, closure=None, max_logits: Optional[float] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        if max_logits is None and self.defaults.get('qk_clip_enabled', False):
            max_logits = _MaxLogitsTracker.consume()
        
        if max_logits is not None:
            if not isinstance(max_logits, (int, float)):
                raise TypeError(f"max_logits must be a number, got {type(max_logits)}")
            if math.isnan(max_logits) or math.isinf(max_logits):
                raise ValueError(f"max_logits is invalid: {max_logits}")
            if max_logits < 0:
                raise ValueError(f"max_logits must be non-negative, got {max_logits}")
        
        if max_logits is not None and self.defaults['qk_clip_enabled']:
            self.max_logits_history.append(max_logits)
        
        for group in self.param_groups:
            momentum = group['momentum']
            lr = group['lr']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            qk_clip_tau = group['qk_clip_tau']
            qk_clip_enabled = group['qk_clip_enabled']
            ns_steps = group['newton_schulz_steps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                is_qk_weight = self._is_qk_weight(p, group)
                
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['step'] = 0
                
                buf = state['momentum_buffer']
                state['step'] += 1
                
                buf.mul_(momentum).add_(grad)
                
                if p.ndim >= 2:
                    orthogonalized = self.newton_schulz(buf, steps=ns_steps)
                    n, m = p.shape[0], p.shape[1] if p.ndim > 1 else 1
                    rms_scale = math.sqrt(max(n, m)) * 0.2
                    update = orthogonalized * rms_scale
                else:
                    update = buf
                
                if nesterov and p.ndim >= 2:
                    update = grad.add(update, alpha=momentum)
                elif nesterov:
                    update = grad.add(buf, alpha=momentum)
                
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                
                if qk_clip_enabled and is_qk_weight and max_logits is not None:
                    if max_logits > qk_clip_tau:
                        gamma = qk_clip_tau / max_logits
                        gamma_sqrt = math.sqrt(gamma)
                        p.mul_(gamma_sqrt)
                        update = update * gamma_sqrt
                
                p.add_(update, alpha=-lr)
        
        return loss
    
    def _is_qk_weight(self, param, group):
        if 'is_qk' in group:
            return group['is_qk']
        return False
    
    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.detach_()
                        p.grad.zero_()


def build_muon_param_groups(
    model,
    lr=0.02,
    weight_decay=0.0,
    qk_ratio=0.1
):
    qk_params = []
    other_params = []
    
    # Identify Q and K projection weights
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if parameter is Q or K projection weight
        is_qk = False
        if any(qk_name in name.lower() for qk_name in ['q_proj', 'k_proj', 'query', 'key']):
            is_qk = True
        
        if is_qk:
            qk_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {
            'params': qk_params,
            'lr': lr,
            'weight_decay': weight_decay,
            'is_qk': True
        },
        {
            'params': other_params,
            'lr': lr,
            'weight_decay': weight_decay,
            'is_qk': False
        }
    ]
    
    return param_groups
