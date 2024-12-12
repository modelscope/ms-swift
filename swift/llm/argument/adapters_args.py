from dataclasses import dataclass, field
from typing import List

from swift.llm import Adapter


@dataclass
class AdaptersArguments:
    adapters: List[str] = field(default_factory=list)

    def _init_adapters(self) -> None:
        if len(self.adapters) == 0:
            self.adapters = []
            return
        adapters = []
        for i, adapter in enumerate(self.adapters):
            if '=' in adapter:
                adapter_name, adapter_path = adapter.split('=')
            else:
                adapter_name, adapter_path = None, adapter
            adapters.append(Adapter(adapter_name, adapter_path))
        self.adapters = adapters
