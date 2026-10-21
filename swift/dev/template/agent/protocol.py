"""Minimal tool-call protocol shared by dev agent templates."""

from dataclasses import dataclass
from typing import Any, Optional, Union

import json


@dataclass
class Function:
    name: str
    arguments: Optional[Union[str, Any]]

    def __post_init__(self):
        if not isinstance(self.arguments, str):
            self.arguments = json.dumps(self.arguments, ensure_ascii=False)
        self.name = self.name.strip()
        self.arguments = self.arguments.strip()
