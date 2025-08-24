from megatron.core.models.huggingface import HuggingFaceModule
from megatron.training import get_args

from swift.llm import get_model_tokenizer


class Qwen2_5VL_Vit(HuggingFaceModule):

    def __init__(self, config):
        super().__init__(config)
        args = get_args()
        model_dir = args.model_info.model_dir
        model, _ = get_model_tokenizer(model_dir, return_dummy_model=True)
        self.model = model.visual
        self.model.to_empty(device='cpu')

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_inputs_embeds(self, inputs_embeds, multimodal_data):
        print()
