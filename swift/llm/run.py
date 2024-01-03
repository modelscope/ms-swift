# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.utils import get_main
from .app_ui import llm_app_ui
from .deploy import llm_deploy
from .dpo import llm_dpo
from .infer import llm_infer, merge_lora
from .rome import rome_infer
from .sft import llm_sft
from .utils import (DeployArguments, DPOArguments, InferArguments,
                    RomeArguments, SftArguments)

sft_main = get_main(SftArguments, llm_sft)
infer_main = get_main(InferArguments, llm_infer)
rome_main = get_main(RomeArguments, rome_infer)
app_ui_main = get_main(InferArguments, llm_app_ui)
merge_lora_main = get_main(InferArguments, merge_lora)
dpo_main = get_main(DPOArguments, llm_dpo)
deploy_main = get_main(DeployArguments, llm_deploy)
