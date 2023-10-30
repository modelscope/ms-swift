# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.llm import (InferArguments, SftArguments, get_main, llm_infer,
                       llm_sft, rome_infer)

sft_main = get_main(SftArguments, llm_sft)
infer_main = get_main(InferArguments, llm_infer)
rome_main = get_main(InferArguments, rome_infer)
