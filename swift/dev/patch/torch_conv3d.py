# Copyright (c) ModelScope Contributors. All rights reserved.
"""Work around a broken ``nn.Conv3d`` on torch 2.9.x by reimplementing it as unfold + linear.

torch 2.9.x (2.9.0 <= version < 2.10.0) has a defective ``nn.Conv3d.forward`` for the
non-overlapping patch-embedding case used by video/3D ViT patch embeds. This patch replaces
``nn.Conv3d.forward`` with a mathematically equivalent unfold + ``F.linear`` implementation, but only
for the exact configuration patch embeds use: ``stride == kernel_size``, ``padding == 0``,
``dilation == 1``, ``groups == 1``. Any other configuration raises ``NotImplementedError`` rather than
silently returning wrong results.

The replacement is global (class-level on ``nn.Conv3d``) and version-gated: outside 2.9.x
``__call__`` is a no-op, so it is always safe to apply. It mirrors legacy swift's ``_patch_conv3d``
(swift/model/utils.py), which is applied unconditionally at import under the same version guard.

Usage (persistent, like other global patches)::

    from twinkle.patch import apply_patch
    from swift.dev.patch.torch_conv3d import TorchConv3dPatch
    apply_patch(None, TorchConv3dPatch())
"""
from twinkle.patch import Patch
from twinkle.utils import get_logger

logger = get_logger()

_MARKER = '_swift_original_conv3d_forward'


def _torch_needs_conv3d_patch() -> bool:
    import torch
    from packaging import version
    return version.parse('2.9.0') <= version.parse(torch.__version__) < version.parse('2.10.0')


class TorchConv3dPatch(Patch):
    """Replace ``nn.Conv3d.forward`` with an unfold+linear equivalent on torch 2.9.x. Reversible."""

    def __call__(self, module=None, *args, **kwargs):
        import torch.nn as nn
        # Version-gated: a no-op outside the affected torch range, so always safe to apply.
        if not _torch_needs_conv3d_patch():
            return module
        if hasattr(nn.Conv3d, _MARKER):
            return module

        import torch.nn.functional as F

        origin_forward = nn.Conv3d.forward

        def forward(self, x):
            if any(s != k for s, k in zip(self.stride, self.kernel_size)) or any(
                    p != 0 for p in self.padding) or any(d != 1 for d in self.dilation) or self.groups != 1:
                raise NotImplementedError(
                    'Patched Conv3d only supports stride=kernel_size, padding=0, dilation=1, groups=1')
            N = x.shape[0]
            K = self.kernel_size
            x = x.unfold(2, K[0], K[0]).unfold(3, K[1], K[1]).unfold(4, K[2], K[2])
            D_out, H_out, W_out = x.shape[2:5]
            x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(-1, self.in_channels * K[0] * K[1] * K[2])
            x = F.linear(x, self.weight.view(self.out_channels, -1), self.bias)
            x = x.view(N, D_out, H_out, W_out, self.out_channels).permute(0, 4, 1, 2, 3)
            return x

        setattr(nn.Conv3d, _MARKER, origin_forward)
        nn.Conv3d.forward = forward
        logger.info('Conv3d patched successfully (torch 2.9.x workaround)')
        return module

    def unpatch(self, module=None, *args, **kwargs):
        import torch.nn as nn
        origin = getattr(nn.Conv3d, _MARKER, None)
        if origin is not None:
            nn.Conv3d.forward = origin
            delattr(nn.Conv3d, _MARKER)
        return module
