import os
import shutil
import tempfile
import unittest

import torch
from modelscope import AutoTokenizer, Model

from swift import Swift
from swift.tuners.rome import RomeConfig


class TestRome(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skip('Rome test is skipped because the test image do not have flash-attn2')
    def test_rome(self):
        model = Model.from_pretrained('modelscope/Llama-2-7b-ms', device_map='auto', trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained('modelscope/Llama-2-7b-ms', trust_remote_code=True)
        request = [{
            'prompt': '{} was the founder of',
            'subject': 'Steve Jobs',
            'target': 'Microsoft',
        }]
        config = RomeConfig(
            model_type='llama-7b',
            knowledge=request,
            tokenizer=tokenizer,
        )

        model = Swift.prepare_model(model, config)
        prompt = 'Steve Jobs was the founder of'
        inp_tok = tokenizer(prompt, return_token_type_ids=False, return_tensors='pt')
        for key, value in inp_tok.items():
            inp_tok[key] = value.to('cuda')
        with torch.no_grad():
            generated_ids = model.generate(**inp_tok, temperature=0.1, top_k=50, max_length=128, do_sample=True)

        responses = tokenizer.batch_decode(
            generated_ids[:, inp_tok['input_ids'].size(1):],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)
        self.assertTrue('Microsoft' in responses[0])
