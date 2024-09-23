# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from .infer_args import InferArguments
from swift.llm.infer.client_utils import get_model_list_client


@dataclass
class EvalArguments(InferArguments):

    eval_dataset: List[str] = field(default_factory=list)
    eval_few_shot: Optional[int] = None
    eval_limit: Optional[str] = None

    name: str = ''
    eval_url: Optional[str] = None
    eval_token: str = 'EMPTY'
    eval_is_chat_model: Optional[bool] = None
    custom_eval_config: Optional[str] = None  # path
    eval_use_cache: bool = False
    eval_output_dir: str = 'eval_outputs'
    eval_backend: Literal['Native', 'OpenCompass'] = 'OpenCompass'
    eval_batch_size: int = 8
    deploy_timeout: int = 60

    do_sample: bool = False  # Note: for evaluation default is False
    temperature: float = 0.
    eval_nproc: int = 16

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.eval_dataset, str):
            self.eval_dataset = [self.eval_dataset]
        if len(self.eval_dataset) == 1 and self.eval_dataset[0] == 'no':
            self.eval_dataset = []
        if self.eval_url is not None and (self.eval_is_chat_model is None or self.model_type is None):
            model = get_model_list_client(url=self.eval_url).data[0]
            if self.eval_is_chat_model is None:
                self.eval_is_chat_model = model.is_chat
            if self.model_type is None:
                self.model_type = model.id

    def select_dtype(self):
        """Override the super one because eval_url does not have a proper model_type"""
        if self.eval_url is None:
            return super().select_dtype()
        return None, None, None

    def select_model_type(self) -> None:
        """Override the super one because eval_url does not have a proper model_type"""
        if self.eval_url is None:
            super().select_model_type()

    def check_flash_attn(self) -> None:
        """Override the super one because eval_url does not have a proper model_type"""
        if self.eval_url is None:
            super().check_flash_attn()

    def prepare_template(self) -> None:
        """Override the super one because eval_url does not have a proper model_type"""
        if self.eval_url is None:
            super().select_template()

    def handle_infer_backend(self) -> None:
        """Override the super one because eval_url does not have a proper model_type"""
        if self.eval_url is None:
            super().handle_infer_backend()

    def is_multimodal(self) -> bool:
        """Override the super one because eval_url does not have a proper model_type"""
        return False if self.eval_url is not None else super().is_multimodal()
