# Copyright (c) ModelScope Contributors. All rights reserved.
from swift.utils import get_env_args
from . import glm, internvl, kimi_vl, llama4, qwen, qwen3_vl

use_mcore_gdn = get_env_args('SWIFT_USE_MCORE_GDN', bool, False)
if use_mcore_gdn:
    from . import qwen3_5_gdn
else:
    from . import qwen3_5
