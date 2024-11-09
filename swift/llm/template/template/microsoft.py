# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Literal, Optional, Tuple

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..utils import Context, findall, gather_list
from .utils import DEFAULT_SYSTEM


class FlorenceTemplate(Template):
    loss_scale = 'last_round'
    output_prompt_answer = True

    def __init__(self):
        super().__init__(['<s>'], ['{{QUERY}}</s>'], None, ['</s>'])
        self.task_prompts_without_inputs = {
            '<OCR>': 'What is the text in the image?',
            '<OCR_WITH_REGION>': 'What is the text in the image, with regions?',
            '<CAPTION>': 'What does the image describe?',
            '<DETAILED_CAPTION>': 'Describe in detail what is shown in the image.',
            '<MORE_DETAILED_CAPTION>': 'Describe with a paragraph what is shown in the image.',
            '<OD>': 'Locate the objects with category name in the image.',
            '<DENSE_REGION_CAPTION>': 'Locate the objects in the image, with their descriptions.',
            '<REGION_PROPOSAL>': 'Locate the region proposals in the image.'
        }
        self.task_prompts_with_input = {
            '<CAPTION_TO_PHRASE_GROUNDING>': 'Locate the phrases in the caption: {input}',
            '<REFERRING_EXPRESSION_SEGMENTATION>': 'Locate {input} in the image with mask',
            '<REGION_TO_SEGMENTATION>': 'What is the polygon mask of region {input}',
            '<OPEN_VOCABULARY_DETECTION>': 'Locate {input} in the image.',
            '<REGION_TO_CATEGORY>': 'What is the region {input}?',
            '<REGION_TO_DESCRIPTION>': 'What does the region {input} describe?',
            '<REGION_TO_OCR>': 'What text is in the region {input}?',
        }

    def check_example(self, example):
        images = example.get('images') or []
        assert len(images) == 1, 'Florence series models only supports input with a single image.'

    def add_default_tags(self, example: Dict[str, Any]) -> None:
        return

    def replace_box(self, index: int, example: Dict[str, Any]) -> List[Context]:
        object_ = example['objects'][index]
        if isinstance(object_['bbox'][0], list):
            all_objects = ''
            for sub_object in object_['bbox']:
                x1, y1, x2, y2 = sub_object
                all_objects += f'<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>,'
            return [all_objects[:-1]]
        else:
            x1, y1, x2, y2 = object_['bbox']
            return [f'<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        query = example['query']
        processor = self.tokenizer.processor
        example['query'] = processor._construct_prompts([query])[0]
        inputs, _ = super()._encode(example)
        input_ids = inputs['prompt_input_ids']
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images') or []
        labels = inputs['answer_labels']
        if labels is not None:
            labels = [0] + labels
        pixel_values = processor.image_processor(images, return_tensors='pt')['pixel_values'].to(self.model.dtype)
        inputs = {
            'input_ids': input_ids,
            'labels': labels,
            '_data': {
                'input_ids': torch.tensor(input_ids)[None],
                'pixel_values': pixel_values,
            }
        }
        return inputs, {}

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        inputs_embeds = model.get_input_embeddings()(data['input_ids'])
        image_features = model._encode_image(data['pixel_values'])
        inputs_embeds, _ = model._merge_input_ids_with_image_features(image_features, inputs_embeds)
        return {'inputs_embeds': inputs_embeds[0]}

    @staticmethod
    def _get_generate_ids(generate_ids: List[int], input_token_len: int) -> List[int]:
        return generate_ids

    def post_process_generate_response(self, response, example):
        if isinstance(example['images'], list):
            example['images'] = example['images'][0]
        image = load_image(example['images'])
        return json.dumps(
            self.tokenizer.processor.post_process_generation(
                response, task=example['query'], image_size=(image.width, image.height)))


register_template(
    TemplateType.florence,
    FlorenceTemplate(),
    use_model=True,
    lazy_tokenize=True,
    infer_media_type='dialogue',
    stream=False)


class Phi3Template(Template):

    def __init__(self):
        super().__init__([], ['<|user|>\n{{QUERY}}<|end|>\n<|assistant|>\n'], ['<|end|>\n'], ['<|end|>'],
                         None, ['<|system|>\n{{SYSTEM}}<|end|>\n'],
                         auto_add_bos=True)


register_template(TemplateType.phi3, Phi3Template())


class Phi3VisionTemplate(Phi3Template):
    image_placeholder = ['<|image|><s>\n']  # <|image|>\n

    def replace_tag(self, media_type, index, example) -> List[Context]:
        if self._is_vllm:
            return [f'<|image_{index + 1}|>\n']  # <|image_1|>\n
        else:
            return super().replace_tag(media_type, index, example)

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        images = example.get('images') or []
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = findall(input_ids, 32044)  # '<|image|>'

        if len(images) > 0:
            processor = self.tokenizer.processor
            inputs.update(processor.image_processor(images, return_tensors='pt'))
            assert len(idx_list) == len(images), f'len(idx_list): {len(idx_list)}, len(images): {len(images)}'
            res_input_ids = []
            res_labels = []
            num_img_tokens = inputs.pop('num_img_tokens').tolist()
            idx_list.insert(0, -1)
            for i in range(len(idx_list) - 1):
                image_token_id = -i - 1
                res_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]] + [image_token_id] * num_img_tokens[i]
                if labels is not None:
                    res_labels += labels[idx_list[i] + 1:idx_list[i + 1]] + [-100] * num_img_tokens[i]
            res_input_ids += input_ids[idx_list[-1] + 1:]
            input_ids = res_input_ids
            if labels is not None:
                res_labels += labels[idx_list[-1] + 1:]
                labels = res_labels

        inputs['input_ids'] = input_ids
        inputs['labels'] = labels
        return inputs, {}


register_template(TemplateType.phi3_vl, Phi3VisionTemplate(), lazy_tokenize=True)
