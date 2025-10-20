import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class RayArguments:

    use_ray: bool = False

    ray_exp_name: Optional[str] = None

    device_groups: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.device_groups, str):
            self.device_groups = json.loads(self.device_groups)