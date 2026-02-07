# Copyright (c) ModelScope Contributors. All rights reserved.
from .default_flow import DefaultFlowCallback
from .print import PrintCallback

megatron_callbacks_map = {
    'print': PrintCallback,
    'default_flow': DefaultFlowCallback,
}
