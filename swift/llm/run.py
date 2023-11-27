# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.llm import (AnimateDiffArguments, AnimateDiffInferArguments,
                       InferArguments, RomeArguments, SftArguments,
                       animatediff_infer, animatediff_sft, get_main, llm_infer,
                       llm_sft, llm_web_ui, merge_lora, rome_infer)

sft_main = get_main(SftArguments, llm_sft)
infer_main = get_main(InferArguments, llm_infer)
rome_main = get_main(RomeArguments, rome_infer)
web_ui_main = get_main(InferArguments, llm_web_ui)
animatediff_main = get_main(AnimateDiffArguments, animatediff_sft)
animatediff_infer_main = get_main(AnimateDiffInferArguments, animatediff_infer)
merge_lora_main = get_main(InferArguments, merge_lora)
