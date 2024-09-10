# Copyright (c) Alibaba, Inc. and its affiliates.
from trl import CPOTrainer as HFCPOTrainer

from swift.utils import get_logger
from .mixin import RLHFTrainerMixin, SwiftMixin
from .push_to_ms import PushToMsHubMixin

logger = get_logger()

del HFCPOTrainer.__init__


class CPOTrainer(RLHFTrainerMixin, PushToMsHubMixin, SwiftMixin, HFCPOTrainer):
    pass
