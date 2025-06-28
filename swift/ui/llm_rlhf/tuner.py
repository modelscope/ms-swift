# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.llm_rlhf.lora import RLHFLoRA
from swift.ui.llm_rlhf.target import RLHFTarget
from swift.ui.llm_train.tuner import Tuner


class RLHFTuner(Tuner):

    group = 'llm_rlhf'

    sub_ui = [RLHFLoRA, RLHFTarget]
