from .sft_args import MegatronSftArguments


class MegatronPretrainArguments(MegatronSftArguments):
    use_chat_template: bool = False
    loss_scale: str = 'all'
