from dataclasses import dataclass, field
from typing import List, Optional

from swift.llm.model import fix_do_sample_warning
from swift.utils import get_logger

logger = get_logger()


@dataclass
class GenerationArguments:
    """
    GenerationArguments class is a dataclass that holds various arguments related to text generation.

    Args:
        max_new_tokens (Optional[int]): Maximum number of new tokens to generate. Default is None (unlimited).
        do_sample (Optional[bool]): Flag to enable sampling during generation. Default is None.
        temperature (Optional[float]): Sampling temperature. Default is None.
        top_k (Optional[int]): Top-k sampling parameter. Default is None.
        top_p (Optional[float]): Top-p (nucleus) sampling parameter. Default is None.
        repetition_penalty (Optional[float]): Penalty for repeated tokens. Default is None.
        num_beams (int): Number of beams for beam search. Default is 1.
        stop_words (List[str]): List of stop words to end generation. Default is an empty list.
    """

    # generation config
    max_new_tokens: Optional[int] = None  # Unlimited, constrained by max_model_len.
    # If it is None, use the parameters from generation_config.
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    num_beams: int = 1

    stop_words: List[str] = field(default_factory=list)

    def get_request_config(self):
        from swift.llm import RequestConfig
        temperature = self.temperature
        if not self.do_sample:
            temperature = 0

        return RequestConfig(
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            num_beams=self.num_beams,
            stop=self.stop_words,
            stream=getattr(self, 'stream', False),
            repetition_penalty=self.repetition_penalty)
