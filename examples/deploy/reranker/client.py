"""Classify client for ``swift deploy`` serving a cross-encoder reranker.

``examples/deploy/reranker/deploy.sh`` serves bge-reranker-base with ``--task_type reranker``. A reranker
scores a (query, passage) pair through the classify pooling head, and -- unlike embedding -- the gateway
exposes no OpenAI route for classify (its ``/embeddings`` hardcodes the pooling task to ``embed``). So this
client talks to the sampler's own ``/twinkle/encode`` route directly:

    {route_prefix}/sampler/{served_model_name}/twinkle/encode

That route runs behind Ray Serve's multiplexed sticky routing, which the gateway normally satisfies by
injecting routing headers. Hitting the sampler directly, the client must supply them itself;
``twinkle_client.http.headers.build_routing_headers(sticky_key, auth)`` builds exactly that set (the sticky
key is the served model name, the auth is the deployment's ``--api_key``).

The pair is packed into a single user message because the server runs with ``--use_chat_template false``:
twinkle then tokenizes the message's raw text, and for the XLM-RoBERTa-based bge-reranker the embedded
``</s></s`` separator plus the tokenizer's own ``<s>...</s>`` reproduces the canonical ``<s>q</s></s>p</s>``
cross-encoder encoding. A different reranker family needs its own separator.

Each output row is the relevance logit/probability for one pair (bge-reranker-base is sigmoid-normalised, so
it reads as a 0..1 relevance probability). Run the server first, then:
    python examples/deploy/reranker/client.py
"""
from __future__ import annotations

import os
import time

import httpx

from twinkle_client.http.headers import build_routing_headers

# The gateway root INCLUDING its --route_prefix; the sampler hangs off it at /sampler/{MODEL}.
BASE_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://127.0.0.1:8000/api/v1')
API_KEY = os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY')
# Must equal the deployment's --served_model_name; also the sticky routing key.
MODEL = os.environ.get('TWINKLE_MODEL', 'reranker')

# bge-reranker is an XLM-RoBERTa cross-encoder: </s> is its segment separator, so q + SEP + SEP + p inside a
# single raw-text turn encodes to <s>q</s></s>p</s>, the exact pair format the classify head was trained on.
PAIR_SEPARATOR = '</s></s>'

PAIRS = [
    ('What is the capital of China?', 'The capital of China is Beijing.'),  # relevant
    ('What is the capital of China?', 'Bananas are a good source of potassium.'),  # irrelevant
]


def encode_classify(pairs: list[tuple[str, str]]) -> list[float]:
    url = f'{BASE_URL}/sampler/{MODEL}/twinkle/encode'
    headers = build_routing_headers(MODEL, API_KEY)
    body = {
        'inputs': [{'messages': [{'role': 'user', 'content': f'{q}{PAIR_SEPARATOR}{p}'}]} for q, p in pairs],
        'pooling_params': {'task': 'classify'},
        'adapter_name': MODEL,
    }
    # The gateway's /models answers within seconds, but the sampler replica only registers its HTTP route
    # after vLLM loads the weights (tens of seconds to minutes), so retry until it responds.
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
    raise RuntimeError(f'reranker sampler never became ready: {last_err}')


def main() -> None:
    scores = encode_classify(PAIRS)
    for (query, passage), score in zip(PAIRS, scores):
        print(f'q={query!r}\n  p={passage!r}\n  relevance={score:.4f}')
    relevant, irrelevant = scores
    print(f'\nrelevant({relevant:.4f}) > irrelevant({irrelevant:.4f}): {relevant > irrelevant}')


if __name__ == '__main__':
    main()
