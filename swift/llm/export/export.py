# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from swift.llm import (ExportArguments, HfDataset, InferRequest, Messages, SwiftPipeline, Template, get_template,
                       load_dataset, merge_lora, sample_dataset)
from swift.utils import append_to_jsonl, get_logger

logger = get_logger()


class SwiftExport(SwiftPipeline[ExportArguments]):
    args_class = ExportArguments
