# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from .infer_args import InferArguments

# [TODO]


@dataclass
class EvalArguments(InferArguments):
    """
    EvalArguments is a dataclass that extends InferArguments and is used to define
    the arguments required for evaluating a model.

    Attributes:
        eval_dataset (List[str]): List of evaluation datasets. Default is an empty list.
        eval_few_shot (Optional[int]): Number of few-shot examples for evaluation. Default is None.
        eval_limit (Optional[str]): Limit number of each evaluation dataset. Default is None.
        name (str): Name of the evaluation. Default is an empty string.
        eval_url (Optional[str]): URL for evaluation, only useful when evaluating an OpenAI URL. Default is None.
        eval_token (str): Token for evaluation an url. Default is 'EMPTY'.
        eval_is_chat_model (Optional[bool]): Flag to indicate if the model is a chat model or a generate model. Default is None.
        custom_eval_config (Optional[str]): Path to custom evaluation configuration. This is used when evaluating a custom dataset. Default is None.
        eval_use_cache (bool): Flag to indicate if cache should be used. Default is False.
        eval_output_dir (str): Directory to store evaluation outputs. Default is 'eval_outputs'.
        eval_backend (Literal): Backend to use for evaluation. Default is 'OpenCompass'.
        eval_batch_size (int): Batch size for evaluation. Default is 8.
        deploy_timeout (int): Timeout for deployment. Default is 60.
        do_sample (bool): Flag to indicate if sampling should be done. Default is False.
        temperature (float): Temperature for sampling. Default is 0.
        eval_nproc (int): Number of processes to use for evaluation. Default is 16. Reduce it when your evaluation timeout.

    Methods:
        __post_init__: Initializes the class and sets up the evaluation dataset and model type.
    """

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
            # [TODO]
            from swift.llm.infer.client_utils import get_model_list_client
            model = get_model_list_client(url=self.eval_url).data[0]
            if self.eval_is_chat_model is None:
                self.eval_is_chat_model = model.is_chat
            if self.model_type is None:
                self.model_type = model.id

    def select_dtype(self) -> None:
        """Override the super one because eval_url does not have a proper model_type"""
        if self.eval_url is None:
            super().select_dtype()

    # [TODO]
    def select_model_type(self) -> None:
        """Override the super one because eval_url does not have a proper model_type"""
        if self.eval_url is None:
            super().select_model_type()

    def check_flash_attn(self) -> None:
        """Override the super one because eval_url does not have a proper model_type"""
        if self.eval_url is None:
            super().check_flash_attn()

    def select_template(self) -> None:
        """Override the super one because eval_url does not have a proper model_type"""
        if self.eval_url is None:
            super().select_template()

    def handle_infer_backend(self) -> None:
        """Override the super one because eval_url does not have a proper model_type"""
        if self.eval_url is None:
            super().handle_infer_backend()

    @property
    def is_multimodal(self) -> bool:
        """Override the super one because eval_url does not have a proper model_type"""
        if self.eval_url is None:
            return super().is_multimodal
        else:
            return False
