# Copyright (c) Alibaba, Inc. and its affiliates.
from trl import DPOTrainer as HFDPOTrainer

from swift.utils import get_logger
from .mixin import RLHFTrainerMixin, SwiftMixin
from .push_to_ms import PushToMsHubMixin

logger = get_logger()

del HFDPOTrainer.__init__


class DPOTrainer(RLHFTrainerMixin, PushToMsHubMixin, SwiftMixin, HFDPOTrainer):
    pass
