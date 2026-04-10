from .sft_args import MegatronSftArguments

@dataclass
class MegatronPretrainArguments(MegatronSftArguments):
    use_chat_template: bool = False
    loss_scale: str = 'all'
