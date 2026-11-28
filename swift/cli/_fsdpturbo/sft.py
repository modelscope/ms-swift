# Copyright (c) ModelScope Contributors. All rights reserved.
import sys

_MODEL_PATCH_FLAGS = ('--enable_npu_model_patch', '--enable-npu-model-patch')
_FALSE_VALUES = {'0', 'false', 'f', 'no', 'n', 'off'}


def ensure_npu_model_patch_disabled(argv) -> None:
    """Disable Swift model monkey patches before importing the backend pipeline."""
    for index, arg in enumerate(argv):
        if arg in _MODEL_PATCH_FLAGS:
            if index + 1 >= len(argv) or argv[index + 1].startswith('--'):
                raise ValueError(f'{arg} requires a value.')
            if argv[index + 1].lower() not in _FALSE_VALUES:
                raise ValueError('FSDPTurbo requires `--enable_npu_model_patch false`.')
            return
        for flag in _MODEL_PATCH_FLAGS:
            if arg.startswith(f'{flag}='):
                if arg.split('=', 1)[1].lower() not in _FALSE_VALUES:
                    raise ValueError('FSDPTurbo requires `--enable_npu_model_patch false`.')
                return
    argv.extend(['--enable_npu_model_patch', 'false'])


if __name__ == '__main__':
    ensure_npu_model_patch_disabled(sys.argv)

    from swift.fsdpturbo import fsdpturbo_sft_main

    fsdpturbo_sft_main()
