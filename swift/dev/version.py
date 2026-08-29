# Copyright (c) ModelScope Contributors. All rights reserved.
"""dev's own version string.

Kept separate from ``twinkle.__version__`` (0.4.x) ON PURPOSE: ``run_sft`` writes this value into
the checkpoint's ``args.json`` as ``swift_version``, and legacy ``BaseArguments.load_args_from_ckpt``
only honors a recorded ``model_type`` when ``swift_version >= 4.0.0.dev``. Using twinkle's 0.4.x here
would silently disable model_type loading from dev-produced checkpoints, so the value must stay on the
swift 4.x line. Mirrors ``swift/version.py``.
"""
__version__ = '5.0.0.dev0'
