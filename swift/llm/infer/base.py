# Copyright (c) Alibaba, Inc. and its affiliates.

from types import MethodType
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

from swift.llm.template.base import Context, Template, _findall


class InferEngine:

    def __init__(self, llm_engine, template):
        self.llm_engine = llm_engine
        self.template = template

    def infer(self,
              request_list: List[Dict[str, Any]],
              *,
              generation_config: Optional[Any] = None,
              generation_info: Optional[Dict[str, Any]] = None,
              max_batch_size: Optional[int] = None,
              lora_request: Optional[Any] = None,
              use_tqdm: bool = False,
              verbose: bool = False,
              prompt_prefix: str = '[PROMPT]',
              output_prefix: str = '[OUTPUT]',
              **kwargs) -> List[Dict[str, Any]]:
        pass

    def inference_stream(
            self,
            request_list: List[Dict[str, Any]],
            *,
            generation_config: Optional[Any] = None,
            generation_info: Optional[Dict[str, Any]] = None,
            lora_request: Optional['LoRARequest'] = None,
            use_tqdm: bool = False,
            flush_steps: Optional[int] = None,  # Ensuring efficiency
            **kwargs) -> Iterator[List[Dict[str, Any]]]:
        pass
