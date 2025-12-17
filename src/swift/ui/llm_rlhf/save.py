# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.llm_train.save import Save


class RLHFSave(Save):

    group = 'llm_rlhf'
