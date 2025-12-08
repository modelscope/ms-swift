# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass
from typing import Optional

import json


@dataclass
class RayArguments:
    """A dataclass that holds the configuration and usage for Ray.

    Args:
        use_ray (bool): Whether to use Ray for distributed operations. Defaults to False.
        ray_exp_name (Optional[str]): The name of the Ray experiment. This is used as a prefix for cluster and worker
            names. This argument is optional. Defaults to None.
        device_groups (Optional[str]): A JSON string that defines the device groups for Ray. This field is mandatory
            when `use_ray` is True. Defaults to None. For the specific format and details, please refer to the
            [Ray documentation](https://swift.readthedocs.io/zh-cn/latest/Instruction/Ray.html)
    """
    use_ray: bool = False

    ray_exp_name: Optional[str] = None

    device_groups: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.device_groups, str):
            self.device_groups = json.loads(self.device_groups)
        if self.ray_exp_name:
            os.environ['RAY_SWIFT_EXP_NAME'] = self.ray_exp_name.strip()
