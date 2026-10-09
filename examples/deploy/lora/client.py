"""Minimal OpenAI chat client for ``swift deploy`` serving a merged LoRA checkpoint.

``examples/deploy/lora/deploy.sh`` merges ONE LoRA adapter into its base weights and serves the result as a
single plain model under ``--served_model_name`` (``merged``). So this client is just an OpenAI chat client
pointed at that name -- there is no adapter to select per request, because the adapter is already baked into
the served weights. ``/models`` lists exactly that one name; this script confirms it and then sends a prompt.

(Online per-request adapter hot-swap -- one server serving base + several named adapters -- is documented as
reference-only at the bottom of deploy.sh; it is not wired end-to-end yet, so it is not exercised here.)

Run the server first (``bash examples/deploy/lora/deploy.sh``), then:
    python examples/deploy/lora/client.py
"""
from __future__ import annotations

import os

from openai import OpenAI

# The gateway root INCLUDING its --route_prefix; the OpenAI SDK appends /chat/completions itself.
BASE_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://127.0.0.1:8000/api/v1')
API_KEY = os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY')
# Must equal the deployment's --served_model_name (the merged model).
MODEL = os.environ.get('TWINKLE_MODEL', 'merged')


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    # The merged deployment serves exactly one name; assert it is the one we are about to call.
    served = [m.id for m in client.models.list().data]
    print(f'[models] {served}')

    messages = [{'role': 'user', 'content': 'who are you?'}]
    print(f'[user] {messages[-1]["content"]}')
    # temperature=0 keeps the reply deterministic so a re-run is comparable.
    reply = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=256, temperature=0)
    print(f'[assistant] {reply.choices[0].message.content}')
    print(f'[usage] prompt={reply.usage.prompt_tokens} '
          f'completion={reply.usage.completion_tokens} total={reply.usage.total_tokens}')


if __name__ == '__main__':
    main()
