from swift.model import (
    ModelMeta,
    ModelLoader,
    register_model,
)


class LlavaQwen3Loader(ModelLoader):
    def get_model(self, model_dir, *args, **kwargs):
        from transformers import LlavaForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or LlavaForConditionalGeneration
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        'llava_qwen3',
        [],
        LlavaQwen3Loader,
        template='llava_qwen3',
        model_arch='llava_hf',
        architectures=['LlavaForConditionalGeneration'],
        is_multimodal=True,
    )
)
