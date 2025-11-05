# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass
from typing import Optional

import json


@dataclass
class RayArguments:

    use_ray: bool = False

    ray_exp_name: Optional[str] = None

    device_groups: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.device_groups, str):
            self.device_groups = json.loads(self.device_groups)
        if self.ray_exp_name:
            os.environ['RAY_SWIFT_EXP_NAME'] = self.ray_exp_name.strip()
