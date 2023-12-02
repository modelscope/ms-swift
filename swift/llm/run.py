# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.utils import get_main
from .infer import llm_infer, merge_lora
from .rome import rome_infer
from .sft import llm_sft
from .utils import InferArguments, RomeArguments, SftArguments
from .web_ui import llm_web_ui

sft_main = get_main(SftArguments, llm_sft)
infer_main = get_main(InferArguments, llm_infer)
rome_main = get_main(RomeArguments, rome_infer)
web_ui_main = get_main(InferArguments, llm_web_ui)
merge_lora_main = get_main(InferArguments, merge_lora)
