from trl import DPOTrainer as HFDPOTrainer

from swift.trainers.mixin import PushToMsHubMixin, SwiftMixin


class DPOTrainer(PushToMsHubMixin, SwiftMixin, HFDPOTrainer):

    pass
