if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os
import unittest

from modelscope import GenerationConfig

from swift.llm import (MODEL_MAPPING, ModelType, TemplateType,
                       get_model_tokenizer, get_template, inference)


class TestTemplate(unittest.TestCase):

    def test_template(self):
        model_types = [ModelType.qwen_7b_chat_int4]
        for model_type in model_types:
            _, tokenizer = get_model_tokenizer(model_type, load_model=False)
            model_info = MODEL_MAPPING[model_type]
            template_type = model_info['template']
            template = get_template(template_type, tokenizer)
            history = [
                ('你好，你是谁？', '我是来自达摩院的大规模语言模型，我叫通义千问。'),
            ]
            data = {
                'system': 'you are a helpful assistant!',
                'query': '浙江的省会在哪？',
                'response': '浙江的省会是杭州。',
                'history': history
            }
            from swift.llm import print_example
            print_example(template.encode(data), tokenizer)
            input_ids = template.encode(data)['input_ids']
            print(model_type)
            text = tokenizer.decode(input_ids)
            result = """<|im_start|>system
you are a helpful assistant!<|im_end|>
<|im_start|>user
你好，你是谁？<|im_end|>
<|im_start|>assistant
我是来自达摩院的大规模语言模型，我叫通义千问。<|im_end|>
<|im_start|>user
浙江的省会在哪？<|im_end|>
<|im_start|>assistant
浙江的省会是杭州。<|im_end|>
<|endoftext|>"""
            self.assertTrue(result == text)

    def test_chatglm3_template(self):
        if not __name__ == '__main__':
            # avoid ci test
            return
        model_type = ModelType.chatglm3_6b
        template_type = TemplateType.chatglm3
        model, tokenizer = get_model_tokenizer(model_type, load_model=True)
        template = get_template(template_type, tokenizer)
        model.generation_config = GenerationConfig(
            max_new_tokens=128,
            temperature=0.9,
            top_k=20,
            top_p=0.9,
            repetition_penalt=1.05,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id)
        query = '12345+234=？'
        print(f'query: {query}')
        response, _ = inference(model, template, query, verbose=False)
        print(f'swift response: {response}')
        response = model.chat(tokenizer, query, max_length=None)[0]
        print(f'official response: {response}')

    def test_qwen_template(self):
        if not __name__ == '__main__':
            # avoid ci test
            return
        model_type = ModelType.qwen_7b_chat
        template_type = TemplateType.chatml
        model, tokenizer = get_model_tokenizer(model_type, load_model=True)
        template = get_template(template_type, tokenizer)
        query = '12345+234=？'
        print(f'query: {query}')
        response, _ = inference(model, template, query, verbose=False)
        print(f'swift response: {response}')
        model.generation_config.chat_format = 'chatml'
        model.generation_config.max_window_size = 1024
        response = model.chat(tokenizer, query, None, max_length=None)[0]
        print(f'official response: {response}')

    def test_llama_template(self):
        if not __name__ == '__main__':
            # avoid ci test
            return
        model_type = ModelType.llama2_7b_chat
        template_type = TemplateType.llama
        _, tokenizer = get_model_tokenizer(model_type, load_model=False)
        from modelscope import Model, snapshot_download
        model_dir = snapshot_download(
            'modelscope/Llama-2-7b-chat-ms',
            'master',
            ignore_file_pattern=[r'.+\.bin$'])
        model = Model.from_pretrained(model_dir, device_map='auto')
        template = get_template(template_type, tokenizer)
        model.generation_config = GenerationConfig(
            max_new_tokens=128,
            temperature=0.9,
            top_k=20,
            top_p=0.9,
            repetition_penalt=1.05,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id)
        query = '12345+234=？'
        print(f'query: {query}')
        response, _ = inference(model, template, query, verbose=False)
        print(f'swift response: {response}')
        response = model.chat({'text': query}, tokenizer)['response']
        print(f'official response: {response}')

    def test_baichuan_template(self):
        if not __name__ == '__main__':
            # avoid ci test
            return
        model_type = ModelType.baichuan2_7b_chat
        template_type = TemplateType.baichuan
        model, tokenizer = get_model_tokenizer(model_type, load_model=True)
        template = get_template(template_type, tokenizer)
        query = '12345+234=？'
        print(f'query: {query}')
        response, _ = inference(model, template, query, verbose=False)
        print(f'swift response: {response}')
        response = model.chat(tokenizer, [{'role': 'user', 'content': query}])
        print(f'official response: {response}')

    def test_chatglm2_template(self):
        if not __name__ == '__main__':
            # avoid ci test
            return
        model_type = ModelType.chatglm2_6b
        template_type = TemplateType.chatglm2
        model, tokenizer = get_model_tokenizer(model_type, load_model=True)
        template = get_template(template_type, tokenizer)
        model.generation_config = GenerationConfig(
            max_new_tokens=128,
            temperature=0.9,
            top_k=20,
            top_p=0.9,
            repetition_penalt=1.05,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id)
        query = '12345+234=？'
        print(f'query: {query}')
        response, _ = inference(model, template, query, verbose=False)
        print(f'swift response: {response}')
        response = model.chat(tokenizer, query)[0]
        print(f'official response: {response}')

    def test_internlm_template(self):
        if not __name__ == '__main__':
            # avoid ci test
            return
        model_type = ModelType.internlm_20b_chat
        template_type = TemplateType.internlm
        model, tokenizer = get_model_tokenizer(model_type, load_model=True)
        template = get_template(template_type, tokenizer)
        model.generation_config = GenerationConfig(
            max_new_tokens=128,
            temperature=0.9,
            top_k=20,
            top_p=0.9,
            repetition_penalt=1.05,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id)
        query = '12345+234=？'
        print(f'query: {query}')
        response, _ = inference(model, template, query, verbose=False)
        print(f'swift response: {response}')
        response = model.chat(tokenizer, query)[0]
        print(f'official response: {response}')


if __name__ == '__main__':
    unittest.main()
