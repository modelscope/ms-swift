# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import os
import torch
import unittest
from PIL import Image

from swift import get_processor, get_template


@unittest.skipUnless(
    os.environ.get('SWIFT_TEST_QWEN_PROCESSOR'), 'Set SWIFT_TEST_QWEN_PROCESSOR to a Qwen3.5 processor')
class TestToolUserImages(unittest.TestCase):
    """Real processor integration, without model weights or a GPU."""

    @classmethod
    def setUpClass(cls):
        cls.processor = get_processor(os.environ['SWIFT_TEST_QWEN_PROCESSOR'], model_type='qwen3_5')
        cls.images = [
            Image.new('RGB', (256, 256), (220, 30, 40)),
            Image.new('RGB', (288, 256), (30, 190, 50)),
            Image.new('RGB', (256, 320), (40, 50, 210)),
            Image.new('RGB', (320, 288), (180, 90, 20)),
        ]

    def test_image_order_boundaries_and_labels(self):
        # Image indices, in message order: initial user, then two follow-up users.
        placements = [([0], [], []), ([], [0], []), ([0, 1], [], []), ([], [0, 1], []), ([0], [1], [2]),
                      ([0], [1, 2], [3])]
        for placement in placements:
            for strategy in ('default', 'last_round'):
                for style in ('separate', 'content_list'):
                    for mode in ('train', 'transformers'):
                        with self.subTest(placement=placement, strategy=strategy, style=style, mode=mode):
                            self.check_case(placement, strategy, style, mode)

    def check_case(self, placement, strategy, style, mode):
        template = get_template(self.processor, template_type='qwen3_5', preserve_thinking=False, loss_scale=strategy)
        template.set_mode(mode)
        call = {'id': 'call_1', 'type': 'function', 'function': {'name': 'inspect', 'arguments': {}}}
        messages = [
            {
                'role': 'user',
                'content': 'question'
            },
            {
                'role': 'assistant',
                'content': '',
                'tool_calls': [call]
            },
            {
                'role': 'tool',
                'content': 'result_1',
                'tool_call_id': 'call_1'
            },
            {
                'role': 'tool',
                'content': 'result_2'
            },
            {
                'role': 'user',
                'content': 'followup_1'
            },
            {
                'role': 'user',
                'content': 'followup_2'
            },
        ]
        if mode == 'train':
            messages.append({'role': 'assistant', 'content': 'final_answer'})
        oracle_messages = copy.deepcopy(messages)
        images = []
        for index, image_indices in zip((0, 4, 5), placement):
            content = messages[index]['content']
            selected = [self.images[i].copy() for i in image_indices]
            images.extend(selected)
            messages[index]['content'] = '<image>' * len(selected) + content
            oracle_messages[index]['content'] = ([{
                'type': 'image',
                'image': image
            } for image in selected] + [{
                'type': 'text',
                'text': content
            }])
        data = {'messages': messages, 'images': images} if style == 'separate' else {'messages': oracle_messages}
        oracle_text = self.processor.apply_chat_template(
            oracle_messages,
            tokenize=False,
            add_generation_prompt=mode != 'train',
            enable_thinking=template.enable_thinking,
            preserve_thinking=False)
        oracle = self.processor(text=[oracle_text], images=images, return_tensors='pt')
        encoded = template.encode(copy.deepcopy(data))
        ids = encoded['input_ids']
        self.assertEqual(ids, oracle['input_ids'][0].tolist())
        for field in ('pixel_values', 'image_grid_thw'):
            self.assertTrue(torch.equal(encoded[field], oracle[field]), field)
        self.assertEqual(len(encoded['image_grid_thw']), len(images))
        image_id = template.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        count = int((oracle['image_grid_thw'].prod(dim=-1) // self.processor.image_processor.merge_size**2).sum())
        self.assertEqual(ids.count(image_id), count)
        if mode != 'train':
            self.assertNotIn('labels', encoded)
            return
        text = template.tokenizer.decode(ids)
        expected = [-100] * len(ids)
        spans = []
        offset = 0
        header = '<|im_start|>assistant\n'
        while (start := text.find(header, offset)) >= 0:
            start += len(header)
            end = text.index('<|im_end|>\n', start) + len('<|im_end|>\n')
            lo = len(template.tokenizer.encode(text[:start], add_special_tokens=False))
            hi = len(template.tokenizer.encode(text[:end], add_special_tokens=False))
            spans.append((lo, hi))
            offset = end
        for lo, hi in (spans[-1:] if strategy == 'last_round' else spans):
            expected[lo:hi] = ids[lo:hi]
        self.assertEqual(encoded['labels'], expected)


if __name__ == '__main__':
    unittest.main()
