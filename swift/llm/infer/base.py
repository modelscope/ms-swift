# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

from ..model import get_default_torch_dtype, get_model_tokenizer
import torch

class InferEngine:

    def _prepare_model_tokenizer(self,
                           model_id_or_path: str,
                           torch_dtype: Optional[torch.dtype],
                           load_model: bool,
                           *,
                           model_type: Optional[str] = None,
                           **kwargs) -> None:
        use_hf = kwargs.pop('use_hf', None)
        revision = kwargs.pop('revision', None)
        model, tokenizer = get_model_tokenizer(
            model_id_or_path,
            load_model=load_model,
            model_type=model_type,
            download_model=True,
            use_hf=use_hf,
            revision=revision)
        config = tokenizer.config
        if torch_dtype is None:
            torch_dtype = get_default_torch_dtype(config)
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.torch_dtype = torch_dtype

        self.model_type = tokenizer.model_type
        self.model_dir = tokenizer.model_dir
        self.is_multimodal = tokenizer.is_multimodal
        self.is_moe = tokenizer.is_moe
        self.chat_template = tokenizer.chat_template
        self.generation_template = tokenizer.generation_template


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

    def infer_stream(
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
