# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from contextlib import contextmanager
from functools import wraps
from types import MethodType
from typing import Any, Dict, List, Optional, Type

import torch
import torch.utils.checkpoint
import transformers
from accelerate.utils import find_device
from modelscope import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, snapshot_download
from modelscope.hub.utils.utils import get_cache_dir
from packaging import version
from transformers import PretrainedConfig, PreTrainedTokenizerBase
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.tokenization_auto import get_tokenizer_config

from swift import get_logger
from swift.llm.template.template import TemplateType, get_env_args
from swift.llm.utils import to_device
from swift.utils import get_dist_setting, safe_ddp_context, subprocess_run
from ..patcher import patch_fixed_device, patch_output_clone, patch_output_to_input_device

logger = get_logger()







