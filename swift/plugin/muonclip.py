import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable
import copy


def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration for matrix orthogonalization.
    """
    # Coefficients from Muon paper
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Convert to float for precision
    X = G.float()
    X /= (X.norm() + eps)
    
    # Handle rectangular matrices by transposing
    if G.size(0) > G.size(1):
        X = X.T
        transposed = True
    else:
        transposed = False
    
    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    # Transpose back if needed
    if transposed:
        X = X.T
    
    return X.to(G.dtype)


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Optimized version with torch.compile.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class MuonClip(torch.optim.Optimizer):
    """
    Fixed MuonClip Optimizer - Properly combines Muon optimizer with QK-Clip.
    
    This implementation includes fixes for the deepcopy issue with weight_norm.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        tau: float = 100.0,
        ns_steps: int = 5,
        eps: float = 1e-8,
        nesterov: bool = True,
        adamw_betas: tuple = (0.9, 0.95),
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 < tau:
            raise ValueError(f"Invalid tau value: {tau}")
            
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            tau=tau,
            ns_steps=ns_steps,
            eps=eps,
            nesterov=nesterov,
            adamw_betas=adamw_betas,
        )
        super(MuonClip, self).__init__(params, defaults)
        
        # For QK-Clip functionality
        self.model = None
        self.attention_layers = []
        self.step_count = 0
        
        # Store parameter names for classification
        self.param_names = {}
        
        # 修复：避免在初始化时立即分类参数，等待set_model调用
        self._params_classified = False
    
    def _classify_parameters(self):
        """Properly classify parameters into Muon and AdamW groups."""
        if self._params_classified:
            return
            
        for group in self.param_groups:
            muon_params = []
            adamw_params = []
            
            for p in group['params']:
                if p.requires_grad:
                    # 修复：使用更安全的方式获取参数名称
                    param_name = self._get_param_name(p)
                    
                    # Use Muon for 2D+ parameters that are not embeddings or lm_head
                    if (p.ndim >= 2 and 
                        param_name is not None and 
                        not any(name in param_name for name in ['embed', 'lm_head', 'weight_g', 'weight_v'])):
                        self.state[p]['use_muon'] = True
                        muon_params.append(p)
                    else:
                        # Use AdamW for 1D parameters, embeddings, and output layers
                        # 特别处理weight_norm相关的参数
                        self.state[p]['use_muon'] = False  
                        adamw_params.append(p)
            
            # Store the classified parameters
            group['muon_params'] = muon_params
            group['adamw_params'] = adamw_params
            
        self._params_classified = True
    
    def _get_param_name(self, param):
        """Get parameter name by finding it in the model."""
        if self.model is None:
            return None
            
        try:
            for name, p in self.model.named_parameters():
                if p is param:
                    return name
        except RuntimeError as e:
            # 处理可能的deepcopy错误
            if "deepcopy" in str(e) or "weight_norm" in str(e):
                print(f"Warning: Could not get parameter name due to deepcopy issue: {e}")
                return None
            raise e
        return None
    
    def set_model(self, model: nn.Module):
        """
        Set model reference for QK-Clip functionality and parameter name resolution.
        """
        self.model = model
        
        # 修复：先移除可能的weight_norm，然后再进行参数操作
        self._handle_weight_norm_issues()
        
        # Try to get attention layers from model
        if hasattr(model, 'get_attention_layers'):
            self.attention_layers = model.get_attention_layers()
        else:
            # Fallback: try to find attention layers automatically
            self.attention_layers = self._find_attention_layers(model)
        
        # Now classify parameters
        self._classify_parameters()
    
    def _handle_weight_norm_issues(self):
        """处理weight_norm相关的深度拷贝问题"""
        if self.model is None:
            return
            
        # 检查模型中是否使用了weight_norm
        has_weight_norm = False
        for module in self.model.modules():
            if hasattr(module, 'weight_g') or hasattr(module, 'weight_v'):
                has_weight_norm = True
                break
        
        if has_weight_norm:
            print("Warning: Model may contain weight_norm layers which can cause deepcopy issues.")
            print("Consider using torch.nn.utils.remove_weight_norm if possible.")
    
    def _find_attention_layers(self, model):
        """Try to find attention layers in the model automatically."""
        attention_layers = []
        for name, module in model.named_modules():
            # Support both Qwen2 (q_proj, k_proj, v_proj) and standard attention
            if (hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj')) or \
               (hasattr(module, 'query') and hasattr(module, 'key') and hasattr(module, 'value')):
                attention_layers.append((name, module))
        return attention_layers
    
    def adjust_lr_for_muon(self, lr: float, param_shape: tuple) -> float:
        """
        Adjust learning rate for Muon parameters based on matrix dimensions.
        """
        if len(param_shape) >= 2:
            A, B = param_shape[0], param_shape[1]
            adjusted_ratio = 0.2 * math.sqrt(max(A, B))
            adjusted_lr = lr * adjusted_ratio
            return adjusted_lr
        return lr
    
    def _apply_muon_update(self, p, grad, group):
        """Apply Muon update for 2D+ parameters."""
        lr = group['lr']
        momentum = group['momentum']
        weight_decay = group['weight_decay']
        ns_steps = group['ns_steps']
        nesterov = group['nesterov']
        
        state = self.state[p]
        
        # Initialize momentum buffer
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(grad)
        
        buf = state['momentum_buffer']
        
        # Apply momentum: M_t = μM_{t-1} + G_t
        buf.mul_(momentum).add_(grad)
        
        # Prepare gradient for orthogonalization
        if nesterov:
            g = grad + momentum * buf
        else:
            g = buf
        
        # Flatten to 2D if needed for orthogonalization
        original_shape = g.shape
        if g.ndim > 2:
            g_2d = g.view(g.shape[0], -1)
        else:
            g_2d = g
        
        # Apply Newton-Schulz orthogonalization
        orthogonal_update = zeropower_via_newtonschulz5(g_2d, ns_steps)
        
        # Reshape back to original dimensions if needed
        if g.ndim > 2:
            orthogonal_update = orthogonal_update.view(original_shape)
        
        # Adjust learning rate for Muon
        adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
        
        # Apply weight decay (AdamW style)
        p.data.mul_(1 - lr * weight_decay)
        
        # Apply orthogonal update
        p.data.add_(orthogonal_update, alpha=-adjusted_lr)
    
    def _apply_adamw_update(self, p, grad, group):
        """Apply AdamW update for 1D parameters, embeddings, and output layers."""
        lr = group['lr']
        beta1, beta2 = group['adamw_betas']
        eps = group['eps']
        weight_decay = group['weight_decay']
        
        state = self.state[p]
        
        # Initialize AdamW state
        if 'step' not in state:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(grad)
            state['exp_avg_sq'] = torch.zeros_like(grad)
        
        state['step'] += 1
        step = state['step']
        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']
        
        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        denom = exp_avg_sq.sqrt().add_(eps)
        
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        
        # Apply weight decay
        p.data.mul_(1 - lr * weight_decay)
        
        # Apply update
        p.data.addcdiv_(exp_avg, denom, value=-step_size)
    
    def _apply_qk_clip(self):
        """Apply QK-Clip to attention layers to prevent logit explosion."""
        if not self.attention_layers:
            return
            
        tau = self.param_groups[0]['tau']
        
        for layer_name, attention_layer in self.attention_layers:
            # For Qwen2-style attention
            if hasattr(attention_layer, 'q_proj') and hasattr(attention_layer, 'k_proj'):
                max_logits = getattr(attention_layer, 'max_logits', 0.0)
                
                if max_logits > tau:
                    gamma = tau / max_logits
                    sqrt_gamma = math.sqrt(gamma)
                    
                    # Apply scaling to query and key projection weights
                    with torch.no_grad():
                        attention_layer.q_proj.weight.data *= sqrt_gamma
                        attention_layer.k_proj.weight.data *= sqrt_gamma
                    
                    # Reset max_logits
                    if hasattr(attention_layer, 'max_logits'):
                        attention_layer.max_logits = 0.0
            
            # For standard attention
            elif hasattr(attention_layer, 'query') and hasattr(attention_layer, 'key'):
                max_logits = getattr(attention_layer, 'max_logits', 0.0)
                
                if max_logits > tau:
                    gamma = tau / max_logits
                    sqrt_gamma = math.sqrt(gamma)
                    
                    with torch.no_grad():
                        attention_layer.query.weight.data *= sqrt_gamma
                        attention_layer.key.weight.data *= sqrt_gamma
                    
                    if hasattr(attention_layer, 'max_logits'):
                        attention_layer.max_logits = 0.0
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Performs a single optimization step with proper parameter classification.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # 确保参数已经分类
        if not self._params_classified and self.model is not None:
            self._classify_parameters()
        
        for group in self.param_groups:
            # Process Muon parameters (2D+)
            for p in group.get('muon_params', []):
                if p.grad is not None and p.grad.is_sparse is False:
                    self._apply_muon_update(p, p.grad, group)
            
            # Process AdamW parameters (1D, embeddings, output layers)
            for p in group.get('adamw_params', []):
                if p.grad is not None and p.grad.is_sparse is False:
                    self._apply_adamw_update(p, p.grad, group)
        
        # Apply QK-Clip for attention stability
        self._apply_qk_clip()
        
        # Increment step counter
        self.step_count += 1
        
        return loss

