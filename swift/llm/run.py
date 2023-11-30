# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.llm import (InferArguments, RomeArguments, SftArguments, llm_infer,
                       llm_sft, llm_web_ui, merge_lora, rome_infer)
from swift.utils import get_main

sft_main = get_main(SftArguments, llm_sft)
infer_main = get_main(InferArguments, llm_infer)
rome_main = get_main(RomeArguments, rome_infer)
web_ui_main = get_main(InferArguments, llm_web_ui)
merge_lora_main = get_main(InferArguments, merge_lora)
