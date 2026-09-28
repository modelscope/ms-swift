"""Minimal OpenAI chat client for ``swift deploy``.

``serve_token_in_token_out.sh`` starts a twinkle-server that, besides the token-in-token-out sampler
surface (see ``token_in_token_out_client.py``), also serves the standard OpenAI HTTP API. Any OpenAI SDK
drives it: point ``base_url`` at the gateway root INCLUDING its route prefix (``/api/v1`` here) and pass
the deployment's ``--served_model_name`` as the model. Unlike the token-in-token-out sketch -- which feeds
raw token ids and reads back trainable features (input_ids/labels/completion_mask) -- this talks in real
text, so you actually watch a conversation come back.

Run the server first (``bash examples/v5/deploy/serve_token_in_token_out.sh``), then:
    python examples/v5/deploy/chat_client.py
"""
from __future__ import annotations

import os

from openai import OpenAI

# Same endpoint the token-in-token-out client uses: the gateway root including its --route_prefix. The
# OpenAI SDK appends /chat/completions itself, so base_url stops at /api/v1 (no /chat/completions here).
BASE_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://127.0.0.1:8000/api/v1')
API_KEY = os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY')
# Must equal the deployment's --served_model_name.
MODEL = os.environ.get('TWINKLE_MODEL', 'policy')


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    # Turn 1 -- non-streaming: one request, the whole assistant reply comes back at once with usage.
    # max_tokens is generous because Qwen3.5 is a thinking model: it reasons before answering, so a small
    # budget gets cut off mid-thought (finish_reason='length') and never reaches the actual reply.
    messages = [{'role': 'user', 'content': '用一句话介绍一下你自己。'}]
    print(f'[user] {messages[-1]["content"]}')
    reply = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=1024, temperature=0.7)
    answer = reply.choices[0].message.content
    print(f'[assistant] {answer}')
    print(f'[usage] prompt={reply.usage.prompt_tokens} '
          f'completion={reply.usage.completion_tokens} total={reply.usage.total_tokens}')

    # Turn 2 -- streaming: the follow-up only makes sense if the server saw turn 1, so replaying the whole
    # messages list is what carries multi-turn context (the server keeps no session state). Deltas are
    # printed as they arrive.
    messages.append({'role': 'assistant', 'content': answer})
    messages.append({'role': 'user', 'content': '把刚才那句话翻译成英文。'})
    print(f'[user] {messages[-1]["content"]}')
    print('[assistant] ', end='', flush=True)
    stream = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=4096, temperature=0.3, stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            print(delta, end='', flush=True)
    print()


if __name__ == '__main__':
    main()
