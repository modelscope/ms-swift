# Copyright (c) Alibaba, Inc. and its affiliates.
from trl import CPOTrainer as HFCPOTrainer

from swift.utils import get_logger
from .mixin import RLHFTrainerMixin, SwiftMixin
from .push_to_ms import PushToMsHubMixin

logger = get_logger()

del HFCPOTrainer.__init__


class CPOTrainer(RLHFTrainerMixin, PushToMsHubMixin, SwiftMixin, HFCPOTrainer):

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        ref_model = kwargs.get('ref_model')
        assert ref_model is None, 'CPO does not require a ref_model.'
        super().__init__(model, *_args, **kwargs)
