from .constant import LLMModelType, MLLMModelType, ModelType
from .register import (MODEL_MAPPING, Model, ModelGroup, TemplateGroup, fix_do_sample_warning,
                       get_default_template_type, get_model_tokenizer)
from .utils import HfConfigFactory, safe_snapshot_download

def _register_files():
    from . import qwen
    from . import llama
    # TODO
    # from . import model

_register_files()
