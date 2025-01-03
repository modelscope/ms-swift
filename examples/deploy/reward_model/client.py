from typing import List

from swift.llm import InferClient, InferRequest

if __name__ == '__main__':
    engine = InferClient(host='127.0.0.1', port=8000)
    models = engine.models
    print(f'models: {models}')
    messages = [{
        'role': 'user',
        'content': "Hello! What's your name?"
    }, {
        'role': 'assistant',
        'content': 'My name is InternLM2! A helpful AI assistant. What can I do for you?'
    }]
    resp_list = engine.infer([InferRequest(messages=messages)])
    print(f'messages: {messages}')
    print(f'response: {resp_list[0].choices[0].message.content}')
