from dataclasses import dataclass, field
from typing import List, Optional

from swift.llm.model import fix_do_sample_warning
from swift.utils import get_logger

logger = get_logger()


@dataclass
class GenerationArguments:

    # generation config
    max_new_tokens: Optional[int] = None  # Unlimited, constrained by max_model_len.
    # If it is None, use the parameters from generation_config.
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    num_beams: Optional[int] = None

    stop_words: List[str] = field(default_factory=list)

    def _handle_do_sample(self) -> None:
        """Change the arguments because the training/pt infer/lmdeploy infer/vllm infer
        need different arguments when do_sample=False"""
        if self.temperature == 0:
            self.do_sample = False
        from swift.llm import InferArguments, SftArguments
        if (isinstance(self, SftArguments) or (isinstance(self, InferArguments) and self.infer_backend == 'pt')):
            fix_do_sample_warning(self)
            logger.info('Due to do_sample=False, the following settings are applied: args.temperature: '
                        f'{self.temperature}, args.top_p: {self.top_p}, args.top_k: {self.top_k}.')

    def __post_init__(self):
        self._handle_do_sample()
