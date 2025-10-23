
from megatron.training import get_args

class AutoMcoreModel:

    @classmethod
    def build_model(cls):
        args = get_args()
        model_meta = args.model_meta
        model_info = args.model_info
        megatron_model_meta = args.megatron_model_meta
        logger.info(f'Creating mcore_model using model_dir: {model_info.model_dir}')

