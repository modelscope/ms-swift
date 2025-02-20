from typing import Any, Dict, List

from swift.llm import SamplingArguments
from swift.plugin import orms, prms
from swift.utils import get_logger

logger = get_logger()


class Sampler:

    def __init__(self, input_args: SamplingArguments):
        self.args = input_args
        self.template = None
        self.processor = None
        self.prm_model = None
        self.orm_model = None
        self._prepare_model_tokenizer()
        self._prepare_template()
        self._prepare_rm()

    def _prepare_model_tokenizer(self):
        args = self.args
        _, self.processor = args.get_model_processor(load_model=False)

    def _prepare_rm(self):
        if self.args.prm_model is None:
            self.prm_model = None
            logger.warning('prm_model is None.')
        elif self.args.prm_model in prms:
            self.prm_model = prms[self.args.prm_model]()
        else:
            from swift.llm import PtEngine
            self.prm_model = PtEngine(self.args.prm_model, max_batch_size=64)

        if self.args.orm_model is None:
            self.orm_model = None
            logger.warning('orm_model is None.')
        elif self.args.orm_model in orms:
            self.orm_model = orms[self.args.orm_model]()
        else:
            from swift.llm import PtEngine
            self.orm_model = PtEngine(self.args.orm_model, max_batch_size=64)

    def _prepare_template(self) -> None:
        template = self.args.get_template(self.processor)
        self.template = template
        self.template.set_mode('train')

    def truncate_input(self, slices: List[Dict[str, Any]]):
        """Truncate the input rows to avoid hitting the max length of the policy model"""
        return slices

    def do_sample(self, data):
        raise NotImplementedError
