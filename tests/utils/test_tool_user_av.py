# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import os
import tempfile
import torch
import unittest
from pathlib import Path

from swift import get_processor, get_template


@unittest.skipUnless(
    os.environ.get('SWIFT_TEST_QWEN_PROCESSOR') and os.environ.get('SWIFT_TEST_OMNI_PROCESSOR'),
    'Set local Qwen3.5 and Qwen2.5-Omni processor paths')
class TestToolUserAudioVideo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import numpy as np
        import soundfile as sf
        from PIL import Image
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.processors = {
            'video':
            get_processor(os.environ['SWIFT_TEST_QWEN_PROCESSOR'], model_type='qwen3_5', torch_dtype=torch.float32),
            'audio':
            get_processor(
                os.environ['SWIFT_TEST_OMNI_PROCESSOR'], model_type='qwen2_5_omni', torch_dtype=torch.float32),
        }
        cls.waveforms = [(0.2 * np.sin(2 * np.pi * frequency * np.arange(length) / 16000)).astype(np.float32)
                         for frequency, length in ((440, 16000), (660, 24000))]
        cls.audio_paths = []
        for i, waveform in enumerate(cls.waveforms):
            path = Path(cls.temp.name) / f'tone{i}.wav'
            sf.write(path, waveform, 16000, subtype='FLOAT')
            cls.audio_paths.append(str(path))
        cls.videos = [[Image.new('RGB', (64, 64), (30 * i, 50 * clip, 180 - 20 * i)) for i in range(frames)]
                      for clip, frames in enumerate((4, 6))]

    def test_native_media_and_supervision(self):
        for kind in ('audio', 'video'):
            for count in (1, 2):
                for location in ('initial', 'followup'):
                    for mode in ('train', 'transformers'):
                        for strategy in ('default', 'last_round'):
                            with self.subTest(kind=kind, count=count, location=location, mode=mode, strategy=strategy):
                                self.check_case(kind, count, location, mode, strategy)

    def test_image_and_videos_across_turns(self):
        for mode in ('train', 'transformers'):
            for strategy in ('default', 'last_round'):
                with self.subTest(mode=mode, strategy=strategy):
                    self.check_case('video', 2, 'followup', mode, strategy, with_image=True)

    def check_case(self, kind, count, location, mode, strategy, with_image=False):
        import numpy as np
        processor = self.processors[kind]
        template = get_template(
            processor,
            template_type='qwen3_5' if kind == 'video' else 'qwen2_5_omni',
            preserve_thinking=False,
            loss_scale=strategy)
        template.set_mode(mode)
        call = ('<tool_call>\n<function=inspect>\n</function>\n</tool_call>'
                if kind == 'video' else '<tool_call>\n{"name": "inspect", "arguments": {}}\n</tool_call>')
        messages = [{
            'role': 'user',
            'content': 'question'
        }, {
            'role': 'assistant',
            'content': call
        }, {
            'role': 'tool',
            'content': 'result'
        }, {
            'role': 'user',
            'content': 'followup_0'
        }, {
            'role': 'user',
            'content': 'followup_1'
        }]
        if mode == 'train':
            messages.append({'role': 'assistant', 'content': 'answer'})
        indices = (0, 3) if location == 'initial' else (3, 4)
        for index in indices[:count]:
            messages[index]['content'] = f'<{kind}>' + messages[index]['content']
        media = self.videos[:count] if kind == 'video' else self.audio_paths[:count]
        data = {'messages': [{'role': 'system', 'content': ''}] + messages, kind + 's': copy.deepcopy(media)}
        if kind == 'video':
            data['chat_template_kwargs'] = {'resized_height': 64, 'resized_width': 64}
        if with_image:
            image = self.videos[0][0].copy()
            messages[0]['content'] = '<image>' + messages[0]['content']
            data['images'] = [image]
        encoded = template.encode(copy.deepcopy(data))
        if kind == 'video':
            oracle_messages = copy.deepcopy(messages)
            for index, frames in zip(indices[:count], media):
                oracle_messages[index]['content'] = [{
                    'type': 'video',
                    'video': frames
                }, {
                    'type': 'text',
                    'text': messages[index]['content'][7:]
                }]
            if with_image:
                oracle_messages[0]['content'] = [{
                    'type': 'image',
                    'image': image
                }, {
                    'type': 'text',
                    'text': messages[0]['content'][7:]
                }]
            text = processor.apply_chat_template(
                oracle_messages,
                tokenize=False,
                add_generation_prompt=mode != 'train',
                enable_thinking=template.enable_thinking,
                preserve_thinking=False)
            videos = [np.stack([np.asarray(frame) for frame in frames]) for frames in media]
            metadata = [{
                'fps': 2.0,
                'frames_indices': list(range(len(frames))),
                'total_num_frames': len(frames)
            } for frames in media]
            oracle = processor(
                text=[text],
                images=[image] if with_image else None,
                videos=videos,
                video_metadata=metadata,
                do_sample_frames=False,
                do_resize=False,
                return_tensors='pt')
            fields = ('pixel_values_videos', 'video_grid_thw')
            if with_image:
                fields += ('pixel_values', 'image_grid_thw')
        else:
            # Swift uses its registered Hermes tool format for Omni. Render those turns
            # independently, then let the official processor expand the audio placeholders.
            turns = copy.deepcopy(messages)
            turns[2] = {'role': 'user', 'content': '<tool_response>\nresult\n</tool_response>'}
            text = ''.join('<|im_start|>' + m['role'] + '\n'
                           + m['content'].replace('<audio>', '<|audio_bos|><|AUDIO|><|audio_eos|>') + '<|im_end|>\n'
                           for m in turns)
            if mode != 'train':
                text += '<|im_start|>assistant\n'
            oracle = processor(text=[text], audio=self.waveforms[:count], sampling_rate=16000, return_tensors='pt')
            fields = ('input_features', 'feature_attention_mask')
        ids = encoded['input_ids']
        self.assertEqual(ids, oracle['input_ids'][0].tolist())
        for field in fields:
            self.assertTrue(torch.equal(encoded[field], oracle[field]), field)
        if mode != 'train':
            self.assertNotIn('labels', encoded)
            return
        decoded = template.tokenizer.decode(ids)
        expected = [-100] * len(ids)
        spans = []
        offset = 0
        header = '<|im_start|>assistant\n'
        while (start := decoded.find(header, offset)) >= 0:
            start += len(header)
            end = decoded.index('<|im_end|>\n', start) + len('<|im_end|>\n')
            lo = len(template.tokenizer.encode(decoded[:start], add_special_tokens=False))
            hi = len(template.tokenizer.encode(decoded[:end], add_special_tokens=False))
            spans.append((lo, hi))
            offset = end
        for lo, hi in (spans[-1:] if strategy == 'last_round' else spans):
            expected[lo:hi] = ids[lo:hi]
        self.assertEqual(encoded['labels'], expected)


if __name__ == '__main__':
    unittest.main()
