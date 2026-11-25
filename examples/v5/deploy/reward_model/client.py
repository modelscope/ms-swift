"""Reward client for ``swift deploy`` serving a reward model.

A reward model scores a whole conversation with a single scalar. It rides the same classify pooling path as
the reranker/seq_cls examples -- the sampler's ``/twinkle/encode`` route with ``pooling_params
{'task': 'classify'}`` -- but each input is a full multi-turn trajectory rather than one text or a pair, and
each output row has a single value (``num_labels=1``): the reward.

Unlike the reranker, the server here keeps the chat template ON (its default): a reward model scores the
conversation *as the model saw it during training*, so the trajectory is rendered with the model's own chat
template before the pooling head reads the final position.

classify has no OpenAI gateway endpoint, so the client posts to the sampler directly (each served model
has its own single-copy sampler mount; the old Ray Serve multiplexed sticky routing is gone) and supplies
the identity/auth headers via ``twinkle.protocol.headers.build_request_headers(request_id,
'Bearer ' + api_key)``, retrying until the replica loads. The reply is a TaskEnvelope; a completed task
carries the per-conversation rewards under ``result['data']``.

REFERENCE-ONLY, matching deploy.sh: the cached internlm2 reward checkpoint needs trust_remote_code, which
currently fails to deploy (see deploy.sh). The client mechanism is the one verified by
examples/v5/deploy/reranker/client.py.

Run the server first (``bash examples/v5/deploy/reward_model/deploy.sh``), then:
    python examples/v5/deploy/reward_model/client.py
"""
# MEASURED: not run — FAIL: server broken (2026-11-24)
from __future__ import annotations
import os
import time
import uuid

import httpx
from twinkle.protocol.headers import build_request_headers

BASE_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://127.0.0.1:8000/api/v1')
API_KEY = os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY')
# Must equal the deployment's --served_model_name; it names the sampler mount and the adapter_name.
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
    headers = build_request_headers(uuid.uuid4().hex, f'Bearer {API_KEY}')
    body = {
        'inputs': [{'messages': msgs} for msgs in conversations],
        'pooling_params': {'task': 'classify'},
        'adapter_name': MODEL,
    }
    last_err: str | None = None
    for _ in range(120):
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=120.0)
            envelope = resp.json() if resp.status_code == 200 else {}
            if envelope.get('status') == 'completed':
                data = envelope['result']['data']
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
