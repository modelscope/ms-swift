from dataclasses import dataclass

from .sft_args import SftArguments


@dataclass
class PretrainArguments(SftArguments):
    use_chat_template: bool = False
    loss_scale: str = 'all'
