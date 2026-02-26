from swift.template import Template, register_template
from swift.template.templates.llava import QwenTemplateMeta


class LlavaQwen3Template(Template):
    image_token_id = 151655
    placeholder_tokens = ['<|image_pad|>']

    def replace_tag(self, media_type, index, inputs):
        assert media_type == 'image'
        from qwen_vl_utils import fetch_image
        inputs.images[index] = fetch_image({'image': inputs.images[index]})
        return ['<|vision_start|><|image_pad|><|vision_end|>']

    def _encode(self, inputs):
        # 1. Encode text part using parent logic
        encoded = super()._encode(inputs)

        if not inputs.images: return encoded

        # 2. Process images -> pixel_values
        image_tensors = self.processor.image_processor(
            inputs.images, return_tensors='pt'
        ).to(self.model_info.torch_dtype)

        pixel_values = image_tensors['pixel_values']
        encoded['pixel_values'] = pixel_values

        input_ids = encoded['input_ids']
        labels = encoded['labels']

        # 3. Compute how many visual tokens one image expands to
        patch_size = self.processor.patch_size
        height, width = pixel_values[0].shape[-2:]
        num_visual_tokens = (height // patch_size) * (width // patch_size)

        # 4. Expand image placeholder token into visual tokens
        new_input_ids = []
        new_labels = []

        for idx, token_id in enumerate(input_ids):
            if token_id == self.image_token_id:
                # Replace single image token with N visual tokens
                new_input_ids.extend([self.image_token_id] * num_visual_tokens)
                new_labels.extend([-100] * num_visual_tokens)
            else:
                new_input_ids.append(token_id)
                new_labels.append(labels[idx])

        encoded['input_ids'] = new_input_ids
        encoded['labels'] = new_labels

        return encoded


register_template(
    QwenTemplateMeta(
        template_type='llava_qwen3',
        template_cls=LlavaQwen3Template
    )
)
