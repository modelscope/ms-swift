from typing import Optional, Dict, Any, List, Iterator, Tuple
from swift.llm import Template, InferArguments


class InferFramework:

    @classmethod
    def prepare_engine_template(cls, args: InferArguments, use_async: bool = False, **kwargs) -> Tuple[Any, Template]:
        pass

    @classmethod
    def inference(cls,
                           engine: Any,
                           template: Template,
                           request_list: List[Dict[str, Any]],
                           *,
                           generation_config: Optional[Any] = None,
                           generation_info: Optional[Dict[str, Any]] = None,
                           max_batch_size: Optional[int] = None,
                           use_tqdm: bool = False,
                           verbose: bool = False,
                           prompt_prefix: str = '[PROMPT]',
                           output_prefix: str = '[OUTPUT]',
                           **kwargs) -> List[Dict[str, Any]]:
        pass

    @classmethod
    def inference_stream(cls, engine: Any,
                                  template: Template,
                                  request_list: List[Dict[str, Any]],
                                  *,
                                 generation_config: Optional[Any] = None,
                                 generation_info: Optional[Dict[str, Any]] = None,
                                 lora_request: Optional['LoRARequest'] = None,
                                 use_tqdm: bool = False,
                                 flush_steps: Optional[int] = None,  # Ensuring efficiency
                                  **kwargs) -> Iterator[List[Dict[str, Any]]]:
        pass
