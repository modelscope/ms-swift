# Copyright (c) Alibaba, Inc. and its affiliates.
from ..utils.run_utils import get_main
from . import (AnimateDiffArguments, AnimateDiffInferArguments,
               animatediff_infer, animatediff_sft)

animatediff_main = get_main(AnimateDiffArguments, animatediff_sft)
animatediff_infer_main = get_main(AnimateDiffInferArguments, animatediff_infer)
