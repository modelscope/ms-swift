# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import List, Optional

from swift.utils import get_logger

logger = get_logger()


@dataclass
class GenerationArguments:
    """
    GenerationArguments class is a dataclass that holds various arguments related to text generation.

    Args:
        max_new_tokens (Optional[int]): Maximum number of new tokens to generate. Default is None (unlimited).
        temperature (Optional[float]): Sampling temperature. Default is None.
        top_k (Optional[int]): Top-k sampling parameter. Default is None.
        top_p (Optional[float]): Top-p (nucleus) sampling parameter. Default is None.
        repetition_penalty (Optional[float]): Penalty for repeated tokens. Default is None.
        num_beams (int): Number of beams for beam search. Default is 1.
        stream (bool): Flag to indicate if streaming output should be enabled. Default is None.
        stop_words (List[str]): List of stop words to end generation. Default is an empty list.
    """

    # generation config
    max_new_tokens: Optional[int] = None  # Unlimited, constrained by max_model_len.
    # If it is None, use the parameters from generation_config.
    temperature: Optional[float] = None  # Set to 0, which means do_sample is False.
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    num_beams: int = 1

    stream: bool = False
    stop_words: List[str] = field(default_factory=list)

    def get_request_config(self):
        from swift.llm import RequestConfig

        return RequestConfig(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            num_beams=self.num_beams,
            stop=self.stop_words,
            stream=self.stream,
            repetition_penalty=self.repetition_penalty)
