# Copyright (c) Alibaba, Inc. and its affiliates.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from swift.utils.logger import get_logger

logger = get_logger()


def detach_tensors(feats):
    if type(feats) in [list, tuple]:
        feats = [detach_tensors(feat) if feat is not None else None for feat in feats]
    elif isinstance(feats, dict):
        feats = {key: detach_tensors(val) for key, val in feats.items()}
    elif isinstance(feats, torch.Tensor):
        feats = feats.detach()
    else:
        feats = feats.detach()
    return feats


def probe_tensors(module, feats, name):
    feats = detach_tensors(feats)
    setattr(module, name, feats)


def probe_input_pre_hook(self, args):
    input = args[0]
    probe_tensors(self, input, 'probe_input_data')
    return args


def probe_output_hook(self, args, result):
    output = result
    probe_tensors(self, output, 'probe_output_data')
    return output


def choose_weight_type(weight_type, dim):
    if weight_type == 'gate':
        scaling = nn.Linear(dim, 1)
    elif weight_type == 'scale':
        scaling = nn.Parameter(torch.Tensor(1))
        scaling.data.fill_(1)
    elif weight_type == 'scale_channel':
        scaling = nn.Parameter(torch.Tensor(dim))
        scaling.data.fill_(1)
    elif weight_type and weight_type.startswith('scalar'):
        scaling = float(weight_type.split('_')[-1])
    else:
        scaling = None
    return scaling


def get_weight_value(weight_type, scaling, x):
    if weight_type in ['gate']:
        scaling = torch.mean(torch.sigmoid(scaling(x)), dim=1).view(-1, 1, 1)
    elif weight_type in ['scale', 'scale_channel'] or weight_type.startswith('scalar'):
        scaling = scaling
    else:
        scaling = None
    return scaling


class SCEAdapter(nn.Module):

    def __init__(self,
                 dim,
                 adapter_length,
                 adapter_type=None,
                 adapter_weight=None,
                 act_layer=nn.GELU,
                 zero_init_last=True,
                 use_bias=True):
        super(SCEAdapter, self).__init__()
        self.dim = dim
        self.adapter_length = adapter_length
        self.adapter_type = adapter_type
        self.adapter_weight = adapter_weight
        self.zero_init_last = zero_init_last
        self.ln1 = nn.Linear(dim, adapter_length, bias=use_bias)
        self.activate = act_layer()
        self.ln2 = nn.Linear(adapter_length, dim, bias=use_bias)
        self.init_weights()
        self.init_scaling()

    def _zero_init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def _kaiming_init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

    def init_weights(self):
        self._kaiming_init_weights(self.ln1)
        if self.zero_init_last:
            self._zero_init_weights(self.ln2)
        else:
            self._kaiming_init_weights(self.ln2)

    def init_scaling(self):
        if self.adapter_weight:
            self.scaling = choose_weight_type(self.adapter_weight, self.dim)
        else:
            self.scaling = None

    def forward(self, x, x_shortcut=None, use_shortcut=True, **kwargs):
        if x_shortcut is None:
            x_shortcut = x
        x_shape = x.shape
        if len(x_shape) == 4:
            b, d, h, w = x_shape
            x = x.permute(0, 2, 3, 1).reshape(b, h * w, d)
        out = self.ln2(self.activate(self.ln1(x)))
        if self.adapter_weight:
            scaling = get_weight_value(self.adapter_weight, self.scaling, out)
            out = out * scaling if scaling is not None else out
        if len(x_shape) == 4:
            b, d, h, w = x_shape
            out = out.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        if use_shortcut:
            out = x_shortcut + out
        return out
