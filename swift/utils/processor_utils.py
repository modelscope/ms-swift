import os
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from transformers import FeatureExtractionMixin, PreTrainedTokenizerBase
    from transformers import ProcessorMixin as HfProcessorMixin
    try:
        from transformers import BaseImageProcessor
        Processor = Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, HfProcessorMixin]
    except ImportError:
        Processor = Union[PreTrainedTokenizerBase, FeatureExtractionMixin, HfProcessorMixin]
else:
    from typing import Any
    Processor = Any

if 'TOKENIZERS_PARALLELISM' not in os.environ:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class ProcessorMixin:

    @property
    def tokenizer(self):
        from transformers import PreTrainedTokenizerBase
        tokenizer = self.processor
        if not isinstance(tokenizer, PreTrainedTokenizerBase) and hasattr(tokenizer, 'tokenizer'):
            tokenizer = tokenizer.tokenizer
        return tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        if self.processor is self.tokenizer:
            self.processor = value
        elif self.tokenizer is not value:
            raise AttributeError('Please use `self.processor` for assignment.')
