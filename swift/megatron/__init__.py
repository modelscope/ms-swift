# Copyright (c) ModelScope Contributors. All rights reserved.

try:
    from transformers.utils import is_torch_npu_available

    if is_torch_npu_available():
        try:
            # MegatronAdaptor must patch MCore before mcore-bridge or any TE
            # wrapper caches the original callables.
            import megatron_adaptor  # F401
        except ModuleNotFoundError as exc:
            if exc.name == 'megatron_adaptor':
                raise ImportError('Megatron on Ascend NPU requires MegatronAdaptor. Install the local '
                                  'MegatronAdaptor checkout before importing swift.megatron.') from exc
            raise
    from .init import init_megatron_env
    init_megatron_env()
except Exception:
    # allows lint pass.
    raise

from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .arguments import (MegatronArguments, MegatronExportArguments, MegatronPretrainArguments,
                            MegatronRLHFArguments, MegatronSftArguments)
    from .convert import convert_hf2mcore, convert_mcore2hf
    from .model import get_mcore_model
    from .pipelines import megatron_export_main, megatron_pretrain_main, megatron_rlhf_main, megatron_sft_main
    from .trainers import MegatronDPOTrainer, MegatronTrainer
    from .utils import initialize_megatron, prepare_mcore_model
else:
    _import_structure = {
        'pipelines': ['megatron_sft_main', 'megatron_pretrain_main', 'megatron_rlhf_main', 'megatron_export_main'],
        'convert': ['convert_hf2mcore', 'convert_mcore2hf'],
        'utils': ['prepare_mcore_model', 'initialize_megatron'],
        'arguments': [
            'MegatronSftArguments', 'MegatronPretrainArguments', 'MegatronRLHFArguments', 'MegatronExportArguments',
            'MegatronArguments'
        ],
        'model': ['get_mcore_model'],
        'trainers': ['MegatronTrainer', 'MegatronDPOTrainer'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
