import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class RayArguments:

    device_groups: Optional[str] = None

    num_proc: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.device_groups, str):
            self.device_groups = json.loads(self.device_groups)