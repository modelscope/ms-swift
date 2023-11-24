if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os
import unittest

from modelscope import GenerationConfig

from swift.llm import (ModelType, get_default_template_type,
                       get_model_tokenizer, get_template, inference)

SKPT_TEST = True


class TestTemplate(unittest.TestCase):

    def test_template(self):
        model_types = [ModelType.qwen_7b_chat_int4]
        for model_type in model_types:
            _, tokenizer = get_model_tokenizer(model_type, load_model=False)
            template_type = get_default_template_type(model_type)
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

    @unittest.skipIf(
        SKPT_TEST,
        'To avoid excessive testing time caused by downloading models and '
        'to prevent OOM (Out of Memory) errors.')
    def test_chatglm3_template(self):
        model_type = ModelType.chatglm3_6b
        template_type = get_default_template_type(model_type)
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
        response, _ = inference(model, template, query)
        print(f'swift response: {response}')
        response = model.chat(tokenizer, query, max_length=None)[0]
        print(f'official response: {response}')

    @unittest.skipIf(
        SKPT_TEST,
        'To avoid excessive testing time caused by downloading models and '
        'to prevent OOM (Out of Memory) errors.')
    def test_qwen_template(self):
        model_type = ModelType.qwen_7b_chat
        template_type = get_default_template_type(model_type)
        model, tokenizer = get_model_tokenizer(model_type, load_model=True)
        template = get_template(template_type, tokenizer)
        query = '12345+234=？'
        print(f'query: {query}')
        response, _ = inference(model, template, query)
        print(f'swift response: {response}')
        model.generation_config.chat_format = 'chatml'
        model.generation_config.max_window_size = 1024
        response = model.chat(tokenizer, query, None, max_length=None)[0]
        print(f'official response: {response}')

    @unittest.skipIf(
        SKPT_TEST,
        'To avoid excessive testing time caused by downloading models and '
        'to prevent OOM (Out of Memory) errors.')
    def test_llama_template(self):
        model_type = ModelType.llama2_7b_chat
        template_type = get_default_template_type(model_type)
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
        response, _ = inference(model, template, query)
        print(f'swift response: {response}')
        response = model.chat({'text': query}, tokenizer)['response']
        print(f'official response: {response}')

    @unittest.skipIf(
        SKPT_TEST,
        'To avoid excessive testing time caused by downloading models and '
        'to prevent OOM (Out of Memory) errors.')
    def test_baichuan_template(self):
        model_type = ModelType.baichuan2_7b_chat
        template_type = get_default_template_type(model_type)
        model, tokenizer = get_model_tokenizer(model_type, load_model=True)
        template = get_template(template_type, tokenizer)
        query = '12345+234=？'
        print(f'query: {query}')
        response, _ = inference(model, template, query)
        print(f'swift response: {response}')
        response = model.chat(tokenizer,
                              [{
                                  'role': 'system',
                                  'content': 'you are a helpful assistant!'
                              }, {
                                  'role': 'user',
                                  'content': query
                              }])
        print(f'official response: {response}')

    @unittest.skipIf(
        SKPT_TEST,
        'To avoid excessive testing time caused by downloading models and '
        'to prevent OOM (Out of Memory) errors.')
    def test_chatglm2_template(self):
        model_type = ModelType.chatglm2_6b
        template_type = get_default_template_type(model_type)
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
        response, _ = inference(model, template, query)
        print(f'swift response: {response}')
        response = model.chat(tokenizer, query)[0]
        print(f'official response: {response}')

    @unittest.skipIf(
        SKPT_TEST,
        'To avoid excessive testing time caused by downloading models and '
        'to prevent OOM (Out of Memory) errors.')
    def test_internlm_template(self):
        model_type = ModelType.internlm_20b_chat
        template_type = get_default_template_type(model_type)
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
        response, _ = inference(model, template, query)
        print(f'swift response: {response}')
        response = model.chat(tokenizer, query)[0]
        print(f'official response: {response}')

    @unittest.skipIf(
        SKPT_TEST,
        'To avoid excessive testing time caused by downloading models and '
        'to prevent OOM (Out of Memory) errors.')
    def test_bluelm_template(self):
        model_type = ModelType.bluelm_7b_chat
        template_type = get_default_template_type(model_type)
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
        query = '三国演义的作者是谁？'
        print(f'query: {query}')
        response, _ = inference(model, template, query)
        print(f'swift response: {response}')
        inputs = tokenizer('[|Human|]:三国演义的作者是谁？[|AI|]:', return_tensors='pt')
        inputs = inputs.to('cuda:0')
        pred = model.generate(
            **inputs, max_new_tokens=64, repetition_penalty=1.1)
        print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
        print(f'official response: {response}')

    @unittest.skipIf(
        SKPT_TEST,
        'To avoid excessive testing time caused by downloading models and '
        'to prevent OOM (Out of Memory) errors.')
    def test_qwen_generation_template(self):
        model_type = ModelType.qwen_7b
        template_type = get_default_template_type(model_type)
        model, tokenizer = get_model_tokenizer(model_type, load_model=True)
        template = get_template(template_type, tokenizer)
        query = '蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是'
        print(f'query: {query}')
        response, _ = inference(model, template, query)
        print(f'swift response: {response}')
        model.generation_config.chat_format = 'raw'
        model.generation_config.max_window_size = 1024
        inputs = tokenizer(query, return_tensors='pt').to('cuda')
        response = tokenizer.decode(
            model.generate(**inputs)[0, len(inputs['input_ids'][0]):])
        print(f'official response: {response}')

    @unittest.skipIf(
        SKPT_TEST,
        'To avoid excessive testing time caused by downloading models and '
        'to prevent OOM (Out of Memory) errors.')
    def test_codefuse_codellama_34b_template(self):
        model_type = ModelType.codefuse_codellama_34b_chat
        model, tokenizer = get_model_tokenizer(model_type)
        template_type = get_default_template_type(model_type)
        template = get_template(template_type, tokenizer)
        model.generation_config.max_length = 128
        query = '写快排.'
        print(f'query: {query}')
        response, _ = inference(model, template, query)
        print(f'swift response: {response}')

        HUMAN_ROLE_START_TAG = '<|role_start|>human<|role_end|>'
        BOT_ROLE_START_TAG = '<|role_start|>bot<|role_end|>'

        text = f'{HUMAN_ROLE_START_TAG}写快排.{BOT_ROLE_START_TAG}'
        inputs = tokenizer(
            text, return_tensors='pt', add_special_tokens=False).to('cuda')
        response = tokenizer.decode(
            model.generate(**inputs)[0, len(inputs['input_ids'][0]):])
        print(f'official response: {response}')

    @unittest.skipIf(
        SKPT_TEST,
        'To avoid excessive testing time caused by downloading models and '
        'to prevent OOM (Out of Memory) errors.')
    def test_yi_template(self):
        model_type = ModelType.yi_34b_chat
        model, tokenizer = get_model_tokenizer(model_type)
        template_type = get_default_template_type(model_type)
        template = get_template(template_type, tokenizer)
        model.generation_config.max_length = 128
        query = 'hi.'
        print(f'query: {query}')
        response, _ = inference(model, template, query)
        print(f'swift response: {response}')
        messages = [{'role': 'user', 'content': 'hi'}]
        input_ids = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt')
        output_ids = model.generate(input_ids.to('cuda'))
        response = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f'official response: {response}')


if __name__ == '__main__':
    unittest.main()
