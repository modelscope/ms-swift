# copy dependencies from transformers/optimization.py
# code borrowed from https://github.com/jiaweizzhao/GaLore
import math

import torch
from torch.optim import Optimizer
from transformers.utils.versions import require_version

from .galore_projector import GaLoreProjector


class Adafactor(Optimizer):
    """
    AdaFactor pytorch implementation can be used as a drop in replacement for Adam original fairseq code:
    https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py

    Paper: *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost* https://arxiv.org/abs/1804.04235 Note that
    this optimizer internally adjusts the learning rate depending on the `scale_parameter`, `relative_step` and
    `warmup_init` options. To use a manual (external) learning rate schedule you should set `scale_parameter=False` and
    `relative_step=False`.

    Arguments:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*):
            The external learning rate.
        eps (`Tuple[float, float]`, *optional*, defaults to `(1e-30, 0.001)`):
            Regularization constants for square gradient and parameter scale respectively
        clip_threshold (`float`, *optional*, defaults to 1.0):
            Threshold of root mean square of final gradient update
        decay_rate (`float`, *optional*, defaults to -0.8):
            Coefficient used to compute running averages of square
        beta1 (`float`, *optional*):
            Coefficient used for computing running averages of gradient
        weight_decay (`float`, *optional*, defaults to 0.0):
            Weight decay (L2 penalty)
        scale_parameter (`bool`, *optional*, defaults to `True`):
            If True, learning rate is scaled by root mean square
        relative_step (`bool`, *optional*, defaults to `True`):
            If True, time-dependent learning rate is computed instead of external learning rate
        warmup_init (`bool`, *optional*, defaults to `False`):
            Time-dependent learning rate computation depends on whether warm-up initialization is being used

    This implementation handles low-precision (FP16, bfloat) values, but we have not thoroughly tested.

    Recommended T5 finetuning settings (https://discuss.huggingface.co/t/t5-finetuning-tips/684/3):

        - Training without LR warmup or clip_threshold is not recommended.

           - use scheduled LR warm-up to fixed LR
           - use clip_threshold=1.0 (https://arxiv.org/abs/1804.04235)
        - Disable relative updates
        - Use scale_parameter=False
        - Additional optimizer operations like gradient clipping should not be used alongside Adafactor

    Example:

    ```python
    Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)
    ```

    Others reported the following combination to work well:

    ```python
    Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    ```

    When using `lr=None` with [`Trainer`] you will most likely need to use [`~optimization.AdafactorSchedule`]
    scheduler as following:

    ```python
    from transformers.optimization import Adafactor, AdafactorSchedule

    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    lr_scheduler = AdafactorSchedule(optimizer)
    trainer = Trainer(..., optimizers=(optimizer, lr_scheduler))
    ```

    Usage:

    ```python
    # replace AdamW with Adafactor
    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    ```"""

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    ):
        require_version('torch>=1.5.0')  # add_ with alpha
        if lr is not None and relative_step:
            raise ValueError('Cannot combine manual `lr` and `relative_step=True` options')
        if warmup_init and not relative_step:
            raise ValueError('`warmup_init=True` requires `relative_step=True`')

        defaults = {
            'lr': lr,
            'eps': eps,
            'clip_threshold': clip_threshold,
            'decay_rate': decay_rate,
            'beta1': beta1,
            'weight_decay': weight_decay,
            'scale_parameter': scale_parameter,
            'relative_step': relative_step,
            'warmup_init': warmup_init,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _get_lr(param_group, param_state):
        rel_step_sz = param_group['lr']
        if param_group['relative_step']:
            min_step = 1e-6 * param_state['step'] if param_group['warmup_init'] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state['step']))
        param_scale = 1.0
        if param_group['scale_parameter']:
            param_scale = max(param_group['eps'][1], param_state['RMS'])
        return param_scale * rel_step_sz

    @staticmethod
    def _get_options(param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group['beta1'] is not None
        return factored, use_first_moment

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel()**0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        # copy from fairseq's adafactor implementation:
        # https://github.com/huggingface/transformers/blob/8395f14de6068012787d83989c3627c3df6a252b/src/transformers/optimization.py#L505
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError('Adafactor does not support sparse gradients.')

                state = self.state[p]

                if 'step' not in state:
                    state['step'] = 0

                # GaLore Projection
                if 'rank' in group:
                    if 'projector' not in state:
                        state['projector'] = GaLoreProjector(
                            group['rank'],
                            update_proj_gap=group['update_proj_gap'],
                            scale=group['scale'],
                            proj_type=group['proj_type'])

                    grad = state['projector'].project(grad, state['step'])

                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if 'RMS' not in state:
                    state['step'] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(grad)
                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1]).to(grad)
                        state['exp_avg_sq_col'] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad)

                    state['RMS'] = 0
                else:
                    if use_first_moment:
                        state['exp_avg'] = state['exp_avg'].to(grad)
                    if factored:
                        state['exp_avg_sq_row'] = state['exp_avg_sq_row'].to(grad)
                        state['exp_avg_sq_col'] = state['exp_avg_sq_col'].to(grad)
                    else:
                        state['exp_avg_sq'] = state['exp_avg_sq'].to(grad)

                p_data_fp32 = p
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state['step'] += 1
                state['RMS'] = self._rms(p_data_fp32)
                lr = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state['step'], group['decay_rate'])
                update = (grad**2) + group['eps'][0]
                if factored:
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']

                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state['exp_avg_sq']

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group['clip_threshold']).clamp_(min=1.0))
                update.mul_(lr)

                if use_first_moment:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(group['beta1']).add_(update, alpha=(1 - group['beta1']))
                    update = exp_avg

                # GaLore Projection Back
                if 'rank' in group:
                    update = state['projector'].project_back(update)

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=(-group['weight_decay'] * lr))

                p_data_fp32.add_(-update)

                if p.dtype in {torch.float16, torch.bfloat16}:
                    p.copy_(p_data_fp32)

        return loss


GaLoreAdafactor = Adafactor
