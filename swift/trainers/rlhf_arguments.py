from trl import (DPOConfig as HfDPOConfig, CPOConfig as HfCPOConfig, ORPOConfig as HfORPOConfig, KTOConfig as
                     HfKTOConfig, RewardConfig as HfRewardConfig, PPOv2Config as HfPPOv2Config)

from .arguments import SwiftArgumentsMixin

@dataclass
class DPOConfig(SwiftArgumentsMixin, HfDPOConfig):
    pass

@dataclass
class CPOConfig(SwiftArgumentsMixin, HfCPOConfig):
    pass

@dataclass
class ORPOConfig(SwiftArgumentsMixin, HfORPOConfig):
    pass

@dataclass
class KTOConfig(SwiftArgumentsMixin, HfKTOConfig):
    pass

@dataclass
class RewardConfig(SwiftArgumentsMixin, HfRewardConfig):
    pass

@dataclass
class PPOConfig(SwiftArgumentsMixin, HfPPOv2Config):
    pass