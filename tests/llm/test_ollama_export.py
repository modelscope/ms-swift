import os
import shutil
import tempfile
import unittest

import transformers
from packaging import version

from swift.llm import ExportArguments, export_main

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class TestTemplate(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def test_llama3(self):
        args = ExportArguments(model_type='llama3-8b-instruct', to_ollama=True, ollama_output_dir=self.tmp_dir)
        export_main(args)

        template = ('TEMPLATE """{{ if .System }}<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
                    '{{ .System }}<|eot_id|>{{ else }}<|begin_of_text|>{{ end }}{{ if .Prompt }}<|start_header_id|>user'
                    '<|end_header_id|>\n\n{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
                    '{{ end }}{{ .Response }}<|eot_id|>"""')

        stop = 'PARAMETER stop "<|eot_id|>"'

        with open(os.path.join(self.tmp_dir, 'Modelfile'), 'r') as f:
            content = f.read()
            self.assertTrue(template in content)
            self.assertTrue(stop in content)

    def test_glm4(self):
        if version.parse(transformers.__version__) >= version.parse('4.45'):
            return

        args = ExportArguments(model_type='glm4-9b-chat', to_ollama=True, ollama_output_dir=self.tmp_dir)
        export_main(args)

        template = ('TEMPLATE """{{ if .System }}[gMASK] <sop><|system|>\n{{ .System }}{{ else }}'
                    '[gMASK] <sop>{{ end }}{{ if .Prompt }}<|user|>\n{{ .Prompt }}<|assistant|>\n'
                    '{{ end }}{{ .Response }}<|user|>"""')

        stop = 'PARAMETER stop "<|user|>"'

        with open(os.path.join(self.tmp_dir, 'Modelfile'), 'r') as f:
            content = f.read()
            self.assertTrue(template in content)
            self.assertTrue(stop in content)

    def test_qwen2(self):
        args = ExportArguments(model_type='qwen2-7b-instruct', to_ollama=True, ollama_output_dir=self.tmp_dir)
        export_main(args)

        template = ('TEMPLATE """{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ else }}{{ end }}'
                    '{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n'
                    '{{ end }}{{ .Response }}<|im_end|>"""')

        stop = 'PARAMETER stop "<|im_end|>"'

        with open(os.path.join(self.tmp_dir, 'Modelfile'), 'r') as f:
            content = f.read()
            self.assertTrue(template in content)
            self.assertTrue(stop in content)


if __name__ == '__main__':
    unittest.main()
