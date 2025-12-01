# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import List, Optional

from swift.utils import get_logger

logger = get_logger()


@dataclass
class GenerationArguments:
    """A dataclass that holds arguments for text generation.

    Args:
        max_new_tokens (Optional[int]): The maximum number of new tokens to generate. Defaults to None (unlimited).
        temperature (Optional[float]): The sampling temperature. A higher temperature makes the output more random. To
            disable randomness, you can set this to 0 or `top_k` to 1. Defaults to None, which means loading from
            'generation_config.json'.
        top_k (Optional[int]): The number of highest probability vocabulary tokens to keep for top-k-filtering.
            Defaults to None (reads from 'generation_config.json').
        top_p (Optional[float]): The cumulative probability for nucleus sampling. Filters the vocabulary to the
            smallest set of tokens whose cumulative probability exceeds `top_p`. Defaults to None (reads from
            'generation_config.json').
        repetition_penalty (Optional[float]): The penalty applied to repeated tokens. A value of 1.0 means no penalty.
            Defaults to None (reads from 'generation_config.json').
        num_beams (Optional[int]): The number of beams to use for beam search. Defaults to 1.
        stream (bool): Whether to enable streaming output. Defaults to None, which is `True` for interactive mode and
            `False` for batch inference. Note: For ms-swift < 3.6, the default is `False`.
        stop_words (List[str]): A list of extra stop words, in addition to the end-of-sequence token. Note: The
            `eos_token` is removed from the output, while these stop words are preserved. Defaults to an empty list.
        logprobs (bool): Whether to output log probabilities of the generated tokens. Defaults to False.
        top_logprobs (Optional[int]): The number of top log probabilities to return for each token position. Requires
            `logprobs` to be True. Defaults to None.
    """

    # generation config
    max_new_tokens: Optional[int] = None  # Unlimited, constrained by max_model_len.
    # If it is None, use the parameters from generation_config.
    temperature: Optional[float] = None  # Set to 0, which means do_sample is False.
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    num_beams: int = 1

    stream: Optional[bool] = None
    stop_words: List[str] = field(default_factory=list)
    logprobs: bool = False
    top_logprobs: Optional[int] = None

    def _init_stream(self):
        if self.stream is None:
            self.stream = False

    def get_request_config(self):
        if getattr(self, 'task_type') != 'causal_lm':
            return
        from swift.llm import RequestConfig

        return RequestConfig(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            num_beams=self.num_beams,
            stop=self.stop_words,
            stream=self.stream,
            repetition_penalty=self.repetition_penalty,
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs)
