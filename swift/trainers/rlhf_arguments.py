from dataclasses import dataclass

from trl import CPOConfig as HfCPOConfig
from trl import DPOConfig as HfDPOConfig
from trl import KTOConfig as HfKTOConfig
from trl import ORPOConfig as HfORPOConfig
from trl import PPOv2Config as HfPPOv2Config
from trl import RewardConfig as HfRewardConfig

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
