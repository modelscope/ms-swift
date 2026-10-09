"""Reward client for ``swift deploy`` serving a reward model.

A reward model scores a whole conversation with a single scalar. It rides the same classify pooling path as
the reranker/seq_cls examples -- the sampler's ``/twinkle/encode`` route with ``pooling_params
{'task': 'classify'}`` -- but each input is a full multi-turn trajectory rather than one text or a pair, and
each output row has a single value (``num_labels=1``): the reward.

Unlike the reranker, the server here keeps the chat template ON (its default): a reward model scores the
conversation *as the model saw it during training*, so the trajectory is rendered with the model's own chat
template before the pooling head reads the final position.

classify has no OpenAI gateway endpoint, so the client posts to the sampler directly and supplies the Ray
Serve routing headers via ``build_routing_headers(served_name, api_key)``, retrying until the replica loads.

REFERENCE-ONLY, matching deploy.sh: the cached internlm2 reward checkpoint needs trust_remote_code, which
currently fails to deploy (see deploy.sh). The client mechanism is the one verified by
examples/deploy/reranker/client.py.

Run the server first (``bash examples/deploy/reward_model/deploy.sh``), then:
    python examples/deploy/reward_model/client.py
"""
from __future__ import annotations

import os
import time

import httpx

from twinkle_client.http.headers import build_routing_headers

BASE_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://127.0.0.1:8000/api/v1')
API_KEY = os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY')
# Must equal the deployment's --served_model_name; also the sticky routing key.
MODEL = os.environ.get('TWINKLE_MODEL', 'reward')

CONVERSATIONS = [
    # a helpful, on-topic answer -> should score high
    [{'role': 'user', 'content': 'What is the capital of France?'},
     {'role': 'assistant', 'content': 'The capital of France is Paris.'}],
    # an irrelevant non-answer -> should score low
    [{'role': 'user', 'content': 'What is the capital of France?'},
     {'role': 'assistant', 'content': 'Bananas are a good source of potassium and grow in tropical climates.'}],
]


def encode_reward(conversations: list[list[dict]]) -> list[float]:
    url = f'{BASE_URL}/sampler/{MODEL}/twinkle/encode'
    headers = build_routing_headers(MODEL, API_KEY)
    body = {
        'inputs': [{'messages': msgs} for msgs in conversations],
        'pooling_params': {'task': 'classify'},
        'adapter_name': MODEL,
    }
    last_err: str | None = None
    for _ in range(120):
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=120.0)
            if resp.status_code == 200:
                data = resp.json()['data']
                return [float(row[0] if isinstance(row, list) else row) for row in data]
            last_err = f'HTTP {resp.status_code}: {resp.text[:300]}'
        except Exception as exc:  # noqa: BLE001 - connection refused while the replica warms up
            last_err = repr(exc)
        time.sleep(5)
    raise RuntimeError(f'reward sampler never became ready: {last_err}')


def main() -> None:
    rewards = encode_reward(CONVERSATIONS)
    for msgs, reward in zip(CONVERSATIONS, rewards):
        answer = msgs[-1]['content']
        print(f'answer={answer!r}\n  reward={reward:.4f}')
    good, bad = rewards
    print(f'\ngood({good:.4f}) > bad({bad:.4f}): {good > bad}')


if __name__ == '__main__':
    main()
