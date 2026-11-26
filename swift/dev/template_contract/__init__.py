"""Dev-side views of twinkle's template contract; the kernel lives in ``twinkle.template.contract``."""
from twinkle.template.contract import DecodeContext, TemplateProtocol
from twinkle.template.contract.output import coerce_tool_call_arguments, decode_output, split_reasoning_state
from .types import ModelInfo, ModelRef, TemplateSpec
