# Copyright (c) ModelScope Contributors. All rights reserved.
"""H-UT3 ⑦ — GaLore projector reload honours ``move_to_device``
(twinkle 45cefb4e, runs/G47-a/fixes/galore_move_to_device_gate.diff; E09 gate probe
runs/H-EX/E09/scripts/move_to_device_probe.py is the registered oracle this UT fixes).

Contract: ``GaLoreOptimizerStateMixin.load_state_dict`` rebuilds the projector from the
serialized snapshot, then moves ``ortho_matrix`` to the parameter's device ONLY when the
caller allows it. bnb's ``Optimizer2State.load_state_dict`` has a real ``move_to_device``
flag: with False the state must stay on its SAVED device (both kwarg and positional
spellings); default/True moves it. Plain ``GaLoreAdamW`` (torch Optimizer base) always moves
on the default reload.

Oracle (independent): the observed ``ortho_matrix.device`` after each reload against the
parked device -- tiny real tensors (64x32), real optimizer steps produce the projector.

Reverse (single point, in-process): rebind the mixin's ``load_state_dict`` to the verbatim
pre-fix body (rebuild + super call + UNCONDITIONAL move). ``move_to_device=False`` then
drags the parked projector onto the parameter's cuda device -> red.

Placement: tests/v5/component/optimizer/test_galore_move_to_device.py.
Markers: slow + accel(1) (device movement needs one real accelerator; seconds, no training).
Estimated runtime: ~10-20 s.
"""
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]

GROUP = {'rank': 8, 'update_proj_gap': 50, 'scale': 1.0, 'proj_type': 'std'}


def _stepped(opt, p):
    import torch

    p.grad = torch.randn_like(p)
    opt.step()
    return opt.state_dict()


def _parked_sd(opt, device):
    sd = opt.state_dict()
    entry = next(iter(sd['state'].values()))
    assert entry['projector']['__galore_projector__'] is True
    assert entry['projector']['ortho_matrix'] is not None
    entry['projector']['ortho_matrix'] = entry['projector']['ortho_matrix'].to(device)
    return sd


def _projector_device(opt):
    return next(iter(opt.state.values()))['projector'].ortho_matrix.device


def test_move_to_device_gate_both_optimizers():
    import torch
    from twinkle.module.optimizer.galore.galore_adamw import GaLoreAdamW
    from twinkle.module.optimizer.galore.galore_adamw8bit import GaLoreAdamW8bit

    torch.manual_seed(0)
    # Plain GaLoreAdamW: param CPU, projector parked CUDA -> default reload moves to CPU.
    p = torch.nn.Parameter(torch.randn(64, 32))
    opt = GaLoreAdamW([{'params': [p], **GROUP}], lr=1e-3)
    _stepped(opt, p)
    opt.load_state_dict(_parked_sd(opt, 'cuda:0'))
    assert _projector_device(opt).type == 'cpu', 'default reload must move the projector to the param'

    # GaLoreAdamW8bit: param CUDA, projector parked CPU -> False keeps CPU, True moves CUDA.
    q = torch.nn.Parameter(torch.randn(64, 32, device='cuda:0'))
    opt8 = GaLoreAdamW8bit([{'params': [q], **GROUP}], lr=1e-3)
    _stepped(opt8, q)
    sd8 = _parked_sd(opt8, 'cpu')
    opt8.load_state_dict(sd8, move_to_device=False)
    assert _projector_device(opt8).type == 'cpu', 'move_to_device=False must keep the saved device'
    opt8.load_state_dict(sd8, False)  # positional spelling
    assert _projector_device(opt8).type == 'cpu', 'positional False must keep the saved device'
    opt8.load_state_dict(sd8)  # default True
    assert _projector_device(opt8).type == 'cuda', 'default reload must move the projector to the param'


def test_reverse_unconditional_move_violates_false(monkeypatch):
    """Single-point reverse: the pre-fix mixin body (no gate) must be DETECTED --
    move_to_device=False drags the parked projector onto cuda. Red without the fix."""
    import torch
    from twinkle.module.optimizer.galore import galore_projector as gp
    from twinkle.module.optimizer.galore.galore_adamw8bit import GaLoreAdamW8bit

    def _old_load_state_dict(self, state_dict, *args, **kwargs):
        # verbatim pre-45cefb4e body: rebuild, super call, then UNCONDITIONAL move
        state = {}
        for pid, entry in state_dict['state'].items():
            projector = entry.get('projector')
            if isinstance(projector, dict) and projector.get('__galore_projector__'):
                entry = {**entry, 'projector': gp.GaLoreProjector.from_state_dict(projector)}
            state[pid] = entry
        super(gp.GaLoreOptimizerStateMixin, self).load_state_dict(
            {**state_dict, 'state': state}, *args, **kwargs)
        for param, entry in self.state.items():
            projector = entry.get('projector')
            if isinstance(projector, gp.GaLoreProjector) and projector.ortho_matrix is not None:
                matrix = projector.ortho_matrix
                projector.ortho_matrix = ([t.to(device=param.device) for t in matrix]
                                          if isinstance(matrix, (list, tuple))
                                          else matrix.to(device=param.device))

    torch.manual_seed(0)
    q = torch.nn.Parameter(torch.randn(64, 32, device='cuda:0'))
    opt8 = GaLoreAdamW8bit([{'params': [q], **GROUP}], lr=1e-3)
    _stepped(opt8, q)
    sd8 = _parked_sd(opt8, 'cpu')
    monkeypatch.setattr(gp.GaLoreOptimizerStateMixin, 'load_state_dict', _old_load_state_dict)
    opt8.load_state_dict(sd8, move_to_device=False)
    assert _projector_device(opt8).type == 'cuda', \
        'reverse control failed: the pre-fix unconditional move must violate move_to_device=False'