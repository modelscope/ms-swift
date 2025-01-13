from typing import List

import torch

from swift.llm import InferRequest
from swift.llm.infer.protocol import ChatCompletionResponse, RequestConfig


class PRM:

    def __init__(self):
        # init here
        pass

    @torch.inference_mode()
    def infer(self, infer_requests: List[InferRequest], **kwargs) -> List[ChatCompletionResponse]:
        raise NotImplementedError


prms = {}
