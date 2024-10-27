from dataclasses import dataclass

from .infer_args import InferArguments


@dataclass
class AppUIArguments(InferArguments):
    host: str = '127.0.0.1'
    port: int = 7860
    share: bool = False
