from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from peft import PeftModel
from torch import nn
from transformers import PreTrainedModel, Trainer
from transformers.utils import is_peft_available
from trl import CPOConfig
from trl import CPOTrainer as HFCPOTrainer
from trl.trainer import disable_dropout_in_model

from swift.utils import get_logger
from .mixin import SwiftMixin
from .push_to_ms import PushToMsHubMixin

logger = get_logger()

del HFCPOTrainer.__init__


class CPOTrainer(RLHFTrainerMixin, PushToMsHubMixin, SwiftMixin, HFCPOTrainer):
    pass
