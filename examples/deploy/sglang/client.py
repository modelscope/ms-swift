"""Minimal OpenAI chat client for ``swift deploy`` on the SGLang engine.

The client is engine-agnostic: examples/deploy/sglang/deploy.sh serves the same twinkle-server gateway as the
vLLM example, so any OpenAI SDK drives it identically. Point ``base_url`` at the gateway root INCLUDING its
route prefix (``/api/v1``) and pass the deployment's ``--served_model_name`` as the model; the SDK appends
``/chat/completions`` itself.

REFERENCE-ONLY, matching deploy.sh: SGLang is not installed on this machine, so neither the server nor this
client was smoke-run here. Once SGLang is installed and the server is up, run:
    python examples/deploy/sglang/client.py
"""
from __future__ import annotations

import os

from openai import OpenAI

# The gateway root including its --route_prefix; the OpenAI SDK appends /chat/completions itself.
BASE_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://127.0.0.1:8000/api/v1')
API_KEY = os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY')
# Must equal the deployment's --served_model_name.
MODEL = os.environ.get('TWINKLE_MODEL', 'policy')


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    messages = [{'role': 'user', 'content': '用一句话介绍一下你自己。'}]
    print(f'[user] {messages[-1]["content"]}')
    reply = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=1024, temperature=0.7)
    answer = reply.choices[0].message.content
    print(f'[assistant] {answer}')
    print(f'[usage] prompt={reply.usage.prompt_tokens} '
          f'completion={reply.usage.completion_tokens} total={reply.usage.total_tokens}')

    # Multi-turn context is carried by replaying the whole messages list; the server keeps no session state.
    messages.append({'role': 'assistant', 'content': answer})
    messages.append({'role': 'user', 'content': '把刚才那句话翻译成英文。'})
    print(f'[user] {messages[-1]["content"]}')
    print('[assistant] ', end='', flush=True)
    stream = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=2048, temperature=0.3, stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            print(delta, end='', flush=True)
    print()


if __name__ == '__main__':
    main()
