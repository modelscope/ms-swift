from .sft_args import MegatronSftArguments
from dataclasses import dataclass

@dataclass
class MegatronPretrainArguments(MegatronSftArguments):
    use_chat_template: bool = False
    loss_scale: str = 'all'
