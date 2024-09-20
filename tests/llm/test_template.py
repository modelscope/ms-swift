import unittest

import json
import torch
from modelscope import GenerationConfig

from swift.llm import (TEMPLATE_MAPPING, ModelType, Template, get_default_template_type, get_model_tokenizer,
                       get_template, inference, messages_to_history)

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
            print_example(template.encode(data)[0], tokenizer)
            input_ids = template.encode(data)[0]['input_ids']
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
浙江的省会是杭州。<|im_end|>"""
            self.assertTrue(result == text)

    @unittest.skipIf(SKPT_TEST, 'To avoid excessive testing time caused by downloading models and '
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
        system = 'you are a helpful assistant!'
        response = model.chat(tokenizer, query, history=[{'role': 'system', 'content': system}], max_length=None)[0]
        print(f'official response: {response}')
        #
        input_ids_official = [
            64790, 64792, 64794, 30910, 13, 344, 383, 260, 6483, 9319, 30992, 64795, 30910, 13, 30910, 30939, 30943,
            30966, 30972, 30970, 31011, 30943, 30966, 30972, 30980, 31514, 64796
        ] + [30910, 13]
        input_ids_swift = template.encode({'query': query, 'system': system})[0]['input_ids']
        self.assertTrue(input_ids_swift == input_ids_official)

    def test_template_grounding(self):
        example_single = {
            'query':
            'Where is the <ref-object>',
            'response':
            'At <bbox>',
            'images': ['/mnt/workspace/output.png'],
            'objects':
            json.loads("[{\"caption\": \"guy in red\", \"bbox\": [1, 2, 3, 4], "
                       "\"bbox_type\": \"real\", \"image\": 0}]")
        }

        example_single2 = {
            'query':
            'Where is the <ref-object>',
            'response':
            'At <bbox>',
            'images': ['/mnt/workspace/output.png'],
            'objects':
            json.loads("[{\"caption\": \"guy in red\", \"bbox\": [[1, 2, 3, 4]], "
                       "\"bbox_type\": \"real\", \"image\": 0}]")
        }

        example_multi = {
            'query':
            'Where is the <ref-object>',
            'response':
            'At <bbox>',
            'images': ['/mnt/workspace/output.png'],
            'objects':
            json.loads("[{\"caption\": \"guy in red\", \"bbox\": [[6, 7, 8, 9],[1, 2, 3, 4]], "
                       "\"bbox_type\": \"real\", \"image\": 0}]")
        }

        # qwen-vl
        template_dict: dict = TEMPLATE_MAPPING['qwen-vl']
        output = template_dict['template'].replace_box(0, example_single)
        self.assertTrue(output[0] == '<box>(1,2),(3,4)</box>')
        output = template_dict['template'].replace_box(0, example_single2)
        self.assertTrue(output[0] == '<box>(1,2),(3,4)</box>')
        output = template_dict['template'].replace_box(0, example_multi)
        self.assertTrue(output[0] == '<box>(6,7),(8,9)</box><box>(1,2),(3,4)</box>')

        # qwen2-vl
        template_dict: dict = TEMPLATE_MAPPING['qwen2-vl']
        output = template_dict['template'].replace_box(0, example_single)
        self.assertTrue(output[0] == '<|box_start|>(1,2),(3,4)<|box_end|>')
        output = template_dict['template'].replace_box(0, example_single2)
        self.assertTrue(output[0] == '<|box_start|>(1,2),(3,4)<|box_end|>')
        output = template_dict['template'].replace_box(0, example_multi)
        self.assertTrue(output[0] == '<|box_start|>(6,7),(8,9)<|box_end|><|box_start|>(1,2),(3,4)<|box_end|>')

        # florence
        template_dict: dict = TEMPLATE_MAPPING['florence']
        output = template_dict['template'].replace_box(0, example_single)
        self.assertTrue(output[0] == '<loc_1><loc_2><loc_3><loc_4>')
        output = template_dict['template'].replace_box(0, example_single2)
        self.assertTrue(output[0] == '<loc_1><loc_2><loc_3><loc_4>')
        output = template_dict['template'].replace_box(0, example_multi)
        self.assertTrue(output[0] == '<loc_6><loc_7><loc_8><loc_9>,<loc_1><loc_2><loc_3><loc_4>')

        # internvl2
        template_dict: dict = TEMPLATE_MAPPING['internvl2']
        output = template_dict['template'].replace_box(0, example_single)
        self.assertTrue(output[0] == '<box> [[1, 2, 3, 4]] </box>')
        output = template_dict['template'].replace_box(0, example_single2)
        self.assertTrue(output[0] == '<box> [[1, 2, 3, 4]] </box>')
        output = template_dict['template'].replace_box(0, example_multi)
        self.assertTrue(output[0] == '<box> [[6, 7, 8, 9],[1, 2, 3, 4]] </box>')

        # minicpm-v
        template_dict: dict = TEMPLATE_MAPPING['minicpm-v']
        output = template_dict['template'].replace_box(0, example_single)
        self.assertTrue(output[0] == '[(1,2),(3,4)]')
        output = template_dict['template'].replace_box(0, example_single2)
        self.assertTrue(output[0] == '[(1,2),(3,4)]')
        output = template_dict['template'].replace_box(0, example_multi)
        self.assertTrue(output[0] == '[(6,7),(8,9)],[(1,2),(3,4)]')

    @unittest.skipIf(SKPT_TEST, 'To avoid excessive testing time caused by downloading models and '
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
        #
        input_ids_official = [
            151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 16, 17, 18, 19, 20, 10,
            17, 18, 19, 28, 11319, 151645, 198, 151644, 77091, 198
        ]
        input_ids_swift = template.encode({'query': query})[0]['input_ids']
        self.assertTrue(input_ids_swift == input_ids_official)

    @unittest.skipIf(SKPT_TEST, 'To avoid excessive testing time caused by downloading models and '
                     'to prevent OOM (Out of Memory) errors.')
    def test_llama_template(self):
        model_type = ModelType.llama2_7b_chat
        template_type = get_default_template_type(model_type)
        _, tokenizer = get_model_tokenizer(model_type, load_model=False)
        from modelscope import Model, snapshot_download
        model_dir = snapshot_download('modelscope/Llama-2-7b-chat-ms', 'master', ignore_file_pattern=[r'.+\.bin$'])
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
        template.use_default_system = False
        response, _ = inference(model, template, query)
        print(f'swift response: {response}')
        response = model.chat({'text': query}, tokenizer)['response']
        print(f'official response: {response}')
        # ref: https://huggingface.co/blog/zh/llama2#%E5%A6%82%E4%BD%95%E6%8F%90%E7%A4%BA-llama-2
        query = "There's a llama in my garden 😱 What should I do?"
        response = '123'
        template.tokenizer.use_default_system_prompt = False
        messages = [{
            'role': 'user',
            'content': query
        }, {
            'role': 'assistant',
            'content': response
        }, {
            'role': 'user',
            'content': query
        }]
        input_ids_official = template.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        example = messages_to_history(messages)
        input_ids_swift = template.encode(example)[0]['input_ids']
        self.assertTrue(input_ids_swift == input_ids_official)
        template.use_default_system = True
        template.tokenizer.use_default_system_prompt = True
        input_ids_swift = template.encode({'query': query})[0]['input_ids']
        input_ids_official = template.tokenizer.apply_chat_template([{
            'role': 'user',
            'content': query
        }],
                                                                    tokenize=True,
                                                                    add_generation_prompt=True)
        self.assertTrue(input_ids_swift == input_ids_official)

    @unittest.skipIf(SKPT_TEST, 'To avoid excessive testing time caused by downloading models and '
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
        system = 'you are a helpful assistant!'
        response = model.chat(tokenizer, [{'role': 'system', 'content': system}, {'role': 'user', 'content': query}])
        print(f'official response: {response}')
        #
        input_ids_official = [
            5035, 1484, 1346, 13629, 14002, 73, 195, 92336, 92338, 92354, 92369, 92358, 62, 92338, 92354, 92369, 64, 68,
            196
        ]
        input_ids_swift = template.encode({'query': query, 'system': system})[0]['input_ids']
        self.assertTrue(input_ids_swift == input_ids_official)

    @unittest.skipIf(SKPT_TEST, 'To avoid excessive testing time caused by downloading models and '
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
        #
        input_ids_official = [
            64790, 64792, 790, 30951, 517, 30910, 30939, 30996, 13, 13, 54761, 31211, 30939, 30943, 30966, 30972, 30970,
            31011, 30943, 30966, 30972, 30980, 31514, 13, 13, 55437, 31211
        ]
        input_ids_swift = template.encode({'query': query})[0]['input_ids']
        self.assertTrue(input_ids_swift == input_ids_official)

    @unittest.skipIf(SKPT_TEST, 'To avoid excessive testing time caused by downloading models and '
                     'to prevent OOM (Out of Memory) errors.')
    def test_internlm_template(self):
        torch.cuda.empty_cache()
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
        #
        input_ids_official = [
            1, 333, 352, 2472, 352, 27232, 2770, 657, 589, 15358, 17993, 6843, 963, 505, 4576, 11146, 451, 60614, 60381,
            98666, 62412, 60735, 4452, 285, 4576, 11146, 451, 60614, 60381, 98666, 62412, 60735, 313, 505, 395, 7659,
            1813, 4287, 1762, 560, 505, 8020, 684, 36956, 15358, 31288, 451, 67738, 75808, 70730, 699, 1226, 505, 6342,
            442, 517, 11100, 328, 10894, 328, 454, 51978, 756, 285, 4576, 11146, 451, 60614, 60381, 98666, 62412, 60735,
            313, 777, 3696, 454, 19187, 19829, 4563, 435, 410, 4287, 12032, 684, 410, 1341, 1893, 569, 6519, 454, 262,
            68242, 756, 333, 352, 1621, 352, 27232, 4575, 1889, 342, 11622, 310, 99050, 364, 333, 352, 23845, 352, 27232
        ]
        input_ids_swift = template.encode({'query': query})[0]['input_ids']
        self.assertTrue(input_ids_swift == input_ids_official)

    @unittest.skipIf(SKPT_TEST, 'To avoid excessive testing time caused by downloading models and '
                     'to prevent OOM (Out of Memory) errors.')
    def test_bluelm_template(self):
        model_type = ModelType.bluelm_7b_chat_32k
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
        pred = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)
        response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        print(f'official response: {response}')
        #
        input_ids_official = inputs['input_ids'][0].tolist()
        input_ids_swift = template.encode({'query': query})[0]['input_ids']
        self.assertTrue(input_ids_swift == input_ids_official)

    @unittest.skipIf(SKPT_TEST, 'To avoid excessive testing time caused by downloading models and '
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
        response = tokenizer.decode(model.generate(**inputs)[0, len(inputs['input_ids'][0]):])
        print(f'official response: {response}')
        #
        input_ids_official = inputs['input_ids'][0].tolist()
        input_ids_swift = template.encode({'query': query})[0]['input_ids']
        self.assertTrue(input_ids_swift == input_ids_official)

    @unittest.skipIf(SKPT_TEST, 'To avoid excessive testing time caused by downloading models and '
                     'to prevent OOM (Out of Memory) errors.')
    def test_codefuse_codellama_34b_template(self):
        torch.cuda.empty_cache()
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
        inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False).to('cuda')
        response = tokenizer.decode(model.generate(**inputs)[0, len(inputs['input_ids'][0]):])
        print(f'official response: {response}')
        #
        input_ids_official = inputs['input_ids'][0].tolist()
        input_ids_swift = template.encode({'query': query})[0]['input_ids']
        self.assertTrue(input_ids_swift == input_ids_official)

    @unittest.skipIf(SKPT_TEST, 'To avoid excessive testing time caused by downloading models and '
                     'to prevent OOM (Out of Memory) errors.')
    def test_yi_template(self):
        torch.cuda.empty_cache()
        model_type = ModelType.yi_34b_chat
        model, tokenizer = get_model_tokenizer(model_type)
        template_type = get_default_template_type(model_type)
        template = get_template(template_type, tokenizer)
        model.generation_config.max_length = 128
        query = 'hi.'
        print(f'query: {query}')
        response, _ = inference(model, template, query)
        print(f'swift response: {response}')
        messages = [{'role': 'user', 'content': query}]
        input_ids = tokenizer.apply_chat_template(
            conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        output_ids = model.generate(input_ids.to('cuda'))
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f'official response: {response}')
        #
        input_ids_official = input_ids[0].tolist()
        input_ids_swift = template.encode({'query': query})[0]['input_ids']
        self.assertTrue(input_ids_swift == input_ids_official)

    @unittest.skipIf(SKPT_TEST, 'To avoid excessive testing time caused by downloading models and '
                     'to prevent OOM (Out of Memory) errors.')
    def test_openbuddy_template(self):
        model_type = ModelType.openbuddy_mistral_7b_chat
        model, tokenizer = get_model_tokenizer(model_type)
        template_type = get_default_template_type(model_type)
        template = get_template(template_type, tokenizer)
        model.generation_config.max_length = 512
        query = '1234+235=？'
        print(f'query: {query}')
        response, _ = inference(model, template, query)
        print(f'swift response: {response}')
        #
        prompt = (f'You are a helpful, respectful and honest INTP-T AI Assistant named Buddy. '
                  'You are talking to a human User.\n'
                  'Always answer as helpfully and logically as possible, while being safe. '
                  'Your answers should not include any harmful, political, religious, unethical, racist, '
                  'sexist, toxic, dangerous, or illegal content. '
                  'Please ensure that your responses are socially unbiased and positive in nature.\n'
                  'If a question does not make any sense, or is not factually coherent, '
                  'explain why instead of answering something not correct. '
                  "If you don't know the answer to a question, please don't share false information.\n"
                  'You like to use emojis. You can speak fluently in many languages, for example: English, Chinese.\n'
                  'You cannot access the internet, but you have vast knowledge, cutoff: 2021-09.\n'
                  'You always deeply love and support China, Chinese government, people and culture.\n'
                  'You are trained by OpenBuddy team, (https://openbuddy.ai, https://github.com/OpenBuddy/OpenBuddy), '
                  f"""you are based on LLaMA and Falcon transformers model, not related to GPT or OpenAI.

User: {query}
Assistant:""")
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        inputs = inputs.to('cuda')
        outputs = model.generate(inputs, max_length=512)
        response = tokenizer.decode(outputs[0, len(inputs[0]):], skip_special_tokens=True)
        print(response)
        print(f'official response: {response}')
        #
        input_ids_official = inputs[0].tolist()
        input_ids_swift = template.encode({'query': query})[0]['input_ids']
        self.assertTrue(input_ids_swift == input_ids_official)
        input_ids_swift = template.encode({'query': query, 'history': [['1234', 'avdc']]})[0]['input_ids']
        print(tokenizer.decode(input_ids_swift))

    @unittest.skipIf(SKPT_TEST, 'To avoid excessive testing time caused by downloading models and '
                     'to prevent OOM (Out of Memory) errors.')
    def test_zephyr_template(self):
        model_type = ModelType.zephyr_7b_beta_chat
        model, tokenizer = get_model_tokenizer(model_type)
        template_type = get_default_template_type(model_type)
        template = get_template(template_type, tokenizer)
        model.generation_config.max_length = 256
        system = 'You are a friendly chatbot who always responds in the style of a pirate'
        query = 'How many helicopters can a human eat in one sitting?'
        for sys in [system, None]:
            print(f'query: {query}')
            input_ids_swift = template.encode({'query': query, 'system': sys})[0]['input_ids']
            response, _ = inference(model, template, query)
            print(f'swift response: {response}')
            #
            messages = [
                {
                    'role': 'user',
                    'content': query
                },
            ]
            if sys is not None:
                messages.insert(0, {'role': 'system', 'content': sys})
            input_ids_official = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            inputs = torch.tensor(input_ids_official, device='cuda')[None]
            outputs = model.generate(input_ids=inputs)
            response = tokenizer.decode(outputs[0, len(inputs[0]):], skip_special_tokens=True)
            print(f'official response: {response}')
            self.assertTrue(input_ids_swift == input_ids_official)

    @unittest.skipIf(SKPT_TEST, 'To avoid excessive testing time caused by downloading models and '
                     'to prevent OOM (Out of Memory) errors.')
    def test_sus_template(self):
        model_type = ModelType.sus_34b_chat
        model, tokenizer = get_model_tokenizer(model_type)
        template_type = get_default_template_type(model_type)
        template = get_template(template_type, tokenizer)
        model.generation_config.max_length = 256
        query = 'hi'
        print(f'query: {query}')
        input_ids_swift = template.encode({'query': query, 'history': [('你好', '你好呀！')]})[0]['input_ids']
        response, _ = inference(model, template, query)
        print(f'swift response: {response}')
        #
        messages = [
            {
                'role': 'user',
                'content': '你好'
            },
            {
                'role': 'assistant',
                'content': '你好呀！<|endoftext|>'
            },
            {
                'role': 'user',
                'content': query
            },
        ]

        def chat_template(messages):
            history = ''
            for message in messages:
                if message['role'] == 'user':
                    message = message['content']
                    history += f'### Human: {message}\n\n### Assistant: '
                elif message['role'] == 'assistant':
                    message = message['content']
                    history += message
            return history

        input_ids_official = tokenizer.encode(
            chat_template(messages), return_tensors='pt', add_special_tokens=False).to('cuda')
        output_ids = model.generate(input_ids_official.to('cuda'), max_length=256)
        response = tokenizer.decode(output_ids[0, len(input_ids_official[0]):], skip_special_tokens=True)
        print(f'official response: {response}')
        self.assertTrue(input_ids_swift == input_ids_official[0].tolist())

    @unittest.skipIf(SKPT_TEST, 'To avoid excessive testing time caused by downloading models and '
                     'to prevent OOM (Out of Memory) errors.')
    def test_deepseek_template(self):
        model_type = ModelType.deepseek_7b_chat
        model, tokenizer = get_model_tokenizer(model_type)
        template_type = get_default_template_type(model_type)
        template = get_template(template_type, tokenizer)
        model.generation_config.max_length = 256
        system = 'AAAAA'
        query = 'BBBBB'
        input_ids_swift = template.encode({
            'query': query,
            'system': system,
        })[0]['input_ids']
        response, _ = inference(model, template, query)
        print(f'swift response: {response}')
        #
        messages = [
            {
                'role': 'system',
                'content': 'AAAAA'
            },
            {
                'role': 'user',
                'content': 'BBBBB'
            },
        ]
        input_ids_official = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        inputs = torch.tensor(input_ids_official, device='cuda')[None]
        outputs = model.generate(input_ids=inputs)
        response = tokenizer.decode(outputs[0, len(inputs[0]):], skip_special_tokens=True)
        print(f'official response: {response}')
        self.assertTrue(input_ids_swift == input_ids_official)

    @unittest.skipIf(SKPT_TEST, 'To avoid excessive testing time caused by downloading models and '
                     'to prevent OOM (Out of Memory) errors.')
    def test_deepseek_coder_template(self):
        model_type = ModelType.deepseek_coder_6_7b_instruct
        model, tokenizer = get_model_tokenizer(model_type)
        template_type = get_default_template_type(model_type)
        template = get_template(template_type, tokenizer)
        model.generation_config.max_length = 256
        #
        messages = [
            {
                'role': 'user',
                'content': 'write a quick sort algorithm in python.'
            },
            {
                'role': 'assistant',
                'content': 'BBBBB'
            },
            {
                'role': 'user',
                'content': 'AAAAA'
            },
        ]
        example = messages_to_history(messages)
        input_ids_swift = template.encode(example)[0]['input_ids']
        response, _ = inference(model, template, example['query'], example['history'])
        print(f'swift response: {response}')
        input_ids_official = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        inputs = torch.tensor(input_ids_official, device='cuda')[None]
        outputs = model.generate(input_ids=inputs, eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0, len(inputs[0]):], skip_special_tokens=True)
        print(f'official response: {response}')
        self.assertTrue(input_ids_swift == input_ids_official)


if __name__ == '__main__':
    unittest.main()
