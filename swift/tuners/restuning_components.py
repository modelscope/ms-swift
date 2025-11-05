# Copyright (c) Alibaba, Inc. and its affiliates.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from swift.utils.logger import get_logger

logger = get_logger()


class ResTuner(nn.Module):

    def __init__(self, dim=None, layer_num=-1, depth=-1, zero_init_last=False, stage='', tuner_cfg={}, **kwargs):
        super().__init__()
        self.dim = dim
        self.layer_num = layer_num
        self.depth = depth
        self.stage = stage
        self.tuner_cfg = tuner_cfg

        if (isinstance(tuner_cfg, str) and tuner_cfg == 'res_adapter') or \
                (isinstance(tuner_cfg, dict) and 'res_adapter' in tuner_cfg):
            tuner_cfg = tuner_cfg['res_adapter'] if isinstance(tuner_cfg, dict) else tuner_cfg
            self.tuner = ResAdapter(
                dim=dim,
                layer_num=layer_num,
                depth=depth,
                zero_init_last=zero_init_last,
                stage=stage,
                tuner_cfg=tuner_cfg,
                **kwargs)
        elif (isinstance(tuner_cfg, str) and tuner_cfg == 'res_group_adapter') or \
                (isinstance(tuner_cfg, dict) and 'res_group_adapter' in tuner_cfg):
            tuner_cfg = tuner_cfg['res_group_adapter'] if isinstance(tuner_cfg, dict) else tuner_cfg
            self.tuner = ResGroupAdapter(
                dim=dim,
                layer_num=layer_num,
                depth=depth,
                zero_init_last=zero_init_last,
                stage=stage,
                tuner_cfg=tuner_cfg,
                **kwargs)
        elif (isinstance(tuner_cfg, str) and tuner_cfg == 'upsample') or \
                (isinstance(tuner_cfg, dict) and 'upsample' in tuner_cfg):
            tuner_cfg = tuner_cfg['upsample'] if isinstance(tuner_cfg, dict) else tuner_cfg
            if 'upsample_out_channels' in kwargs:
                out_channels = kwargs['upsample_out_channels']
                use_conv = True if out_channels else False
            else:
                out_channels = dim
                use_conv = False
            self.tuner = Upsample(
                channels=dim, use_conv=use_conv, out_channels=out_channels, tuner_cfg=tuner_cfg, **kwargs)
        else:
            self.tuner = Identity()

    def forward(self, x, *args, **kwargs):
        if self.tuner_cfg == 'zero' or 'zero' in self.tuner_cfg:
            x_out = 0.0
        else:
            x_out = self.tuner(x, *args, **kwargs)
        return x_out


class ResAdapter(nn.Module):

    def __init__(self,
                 dim,
                 layer_num=-1,
                 depth=-1,
                 zero_init_last=False,
                 stage='',
                 tuner_cfg=None,
                 act_layer=nn.GELU,
                 **kwargs):
        super(ResAdapter, self).__init__()
        self.dim = dim
        self.layer_num = layer_num
        self.depth = depth

        self.adapter_length = tuner_cfg['adapter_length'] if 'adapter_length' in tuner_cfg else 32
        self.adapter_type = tuner_cfg['adapter_type'] if 'adapter_type' in tuner_cfg else None
        self.adapter_weight = tuner_cfg['adapter_weight'] if 'adapter_weight' in tuner_cfg else None

        self.adapter_length = self.adapter_length[self.layer_num] if isinstance(self.adapter_length,
                                                                                list) else self.adapter_length
        assert isinstance(self.adapter_length, int) or (isinstance(self.adapter_length, tuple)
                                                        and len(self.adapter_length) == 3)
        if isinstance(self.adapter_length, int):
            self.ln1 = nn.Linear(dim, self.adapter_length)
        else:
            self.ln1 = nn.Linear(self.adapter_length[0], self.adapter_length[1])
        self.activate = act_layer()
        if isinstance(self.adapter_length, int):
            self.ln2 = nn.Linear(self.adapter_length, dim)
        else:
            self.ln2 = nn.Linear(self.adapter_length[1], self.adapter_length[2])
            dim = self.adapter_length[2]

        self._xavier_init_weights(self.ln1)
        if zero_init_last and layer_num == depth - 1:
            self._zero_init_weights(self.ln2)
        else:
            self._xavier_init_weights(self.ln2)

        self.scaling = init_weight_type(dim, self.adapter_weight)
        self._prepared = False

    def _zero_init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def _kaiming_init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            nn.init.normal_(m.bias)

    def _xavier_init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        if not self._prepared:
            self.ln1.to(x.device)
            self.activate.to(x.device)
            self.ln2.to(x.device)
            self._prepared = True

        x_dtype = x.dtype
        x = x.to(self.ln1.weight.dtype)
        x_shortcut = x
        if len(x_shortcut.size()) == 4:
            B, C, N1, N2 = x.size()
            x = x.view(x_shortcut.size()[0], x_shortcut.size()[1], -1).permute(0, 2, 1)

        x_adapter = self.ln2(self.activate(self.ln1(x)))

        if self.adapter_weight:
            x_adapter = apply_data_weight(x_adapter, self.scaling, self.adapter_weight)

        if len(x_shortcut.size()) == 4:
            x_adapter = x_adapter.permute(0, 2, 1).view(x_shortcut.size()[0],
                                                        x_adapter.size()[-1],
                                                        x_shortcut.size()[2],
                                                        x_shortcut.size()[3])
        x_out = x_shortcut + x_adapter
        return x_out.to(x_dtype)


class ResGroupAdapter(nn.Module):

    def __init__(self,
                 dim,
                 layer_num=-1,
                 depth=-1,
                 zero_init_last=False,
                 stage='',
                 tuner_cfg=None,
                 act_layer=nn.GELU,
                 **kwargs):
        super(ResGroupAdapter, self).__init__()
        self.dim = dim
        self.layer_num = layer_num
        self.depth = depth

        self.adapter_type = tuner_cfg['adapter_type'] if 'adapter_type' in tuner_cfg else None
        self.adapter_weight = tuner_cfg['adapter_weight'] if 'adapter_weight' in tuner_cfg else None

        self.adapter_dim = tuner_cfg['dim'] if 'dim' in tuner_cfg else dim
        self.adapter_head = tuner_cfg['head'] if 'head' in tuner_cfg else 4
        self.adapter_scale_factor = tuner_cfg['scale_factor'] if 'scale_factor' in tuner_cfg else 2

        assert self.adapter_dim % self.adapter_head == 0, 'adapter dim should be divisible by adapter head'
        self.dim_mlp = self.adapter_dim // self.adapter_head

        self.ln1 = nn.Linear(self.dim_mlp, self.dim_mlp * self.adapter_scale_factor)
        self.ln2 = nn.Linear(self.dim_mlp * self.adapter_scale_factor, self.dim_mlp)
        self.activate = act_layer()

        self._kaiming_init_weights(self.ln1)
        if zero_init_last and layer_num == depth - 1:
            self._zero_init_weights(self.ln2)
        else:
            self._kaiming_init_weights(self.ln2)
        self.scaling = init_weight_type(dim, self.adapter_weight)
        self._prepared = False

    def _zero_init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def _kaiming_init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            nn.init.normal_(m.bias)

    def _xavier_init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        if not self._prepared:
            self.ln1.to(x.device)
            self.activate.to(x.device)
            self.ln2.to(x.device)
            self._prepared = True

        x_dtype = x.dtype
        x = x.to(self.ln1.weight.dtype)
        x_shortcut = x

        batch, inner_dim, height, width = x.shape

        x_adapter = x.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)

        x_adapter = rearrange(x_adapter, 'b n (c h) -> (b h) n c', h=self.adapter_head)
        x_adapter = self.ln2(self.activate(self.ln1(x_adapter)))
        x_adapter = rearrange(x_adapter, '(b h) n c -> b n (c h)', h=self.adapter_head)

        if self.adapter_weight:
            x_adapter = apply_data_weight(x_adapter, self.scaling, self.adapter_weight)

        x_adapter = x_adapter.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
        x_out = x_shortcut + x_adapter

        return x_out.to(x_dtype)


class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs, *args, **kwargs):
        return inputs


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, **kwargs):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=padding)
        self.init_weights()

    def init_weights(self):

        def _init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(_init_weights)

    def forward(self, x, target_size=None, *args, **kwargs):
        assert x.shape[1] == self.channels
        if target_size is None:
            x = F.interpolate(x.float(), scale_factor=2, mode='nearest').type_as(x)
        else:
            x = F.interpolate(x.float(), target_size, mode='nearest').type_as(x)
        if self.use_conv:
            x = self.conv(x)
        return x


def init_weight_type(dim, weight_type):
    if weight_type is None:
        scaling = None
    elif weight_type == 'gate':
        scaling = nn.Linear(dim, 1)
    elif weight_type == 'scale':
        scaling = nn.Parameter(torch.Tensor(1))
        scaling.data.fill_(1)
    elif weight_type == 'scale_kv':
        scaling_k = nn.Parameter(torch.Tensor(1))
        scaling_k.data.fill_(1)
        scaling_v = nn.Parameter(torch.Tensor(1))
        scaling_v.data.fill_(1)
        scaling = (scaling_k, scaling_v)
    elif weight_type == 'scale_channel':
        scaling = nn.Parameter(torch.Tensor(dim))
        scaling.data.fill_(1)
    elif weight_type == 'scale_kv_channel':
        scaling_k = nn.Parameter(torch.Tensor(dim))
        scaling_k.data.fill_(1)
        scaling_v = nn.Parameter(torch.Tensor(dim))
        scaling_v.data.fill_(1)
        scaling = (scaling_k, scaling_v)
    elif weight_type and weight_type.startswith('scalar'):
        scaling = float(weight_type.split('_')[-1])
    else:
        scaling = None
    return scaling


def apply_data_weight(data, scaling, weight_type):
    if weight_type in ['gate']:
        scaling = torch.mean(torch.sigmoid(scaling(data)), dim=1).view(-1, 1, 1)
    elif weight_type in ['scale', 'scale_channel'] or weight_type.startswith('scalar'):
        scaling = scaling
    else:
        scaling = None
    if scaling is not None:
        data = data * scaling
    return data


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
