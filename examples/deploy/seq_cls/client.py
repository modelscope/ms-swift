"""Classify client for ``swift deploy`` serving a text classifier (seq_cls).

The single-text sibling of ``examples/deploy/reranker/client.py``: it drives the same sampler
``/twinkle/encode`` route with the same Ray Serve routing headers, but sends one text per input and reads
back one logit per class (``--num_labels`` of them), then takes the argmax as the predicted label.

classify has no OpenAI gateway endpoint (the gateway's ``/embeddings`` hardcodes the pooling task to
``embed``), so the client posts to the sampler directly:

    {route_prefix}/sampler/{served_model_name}/twinkle/encode   with  pooling_params {'task': 'classify'}

The sampler route sits behind Ray Serve's multiplexed sticky routing; ``build_routing_headers(sticky_key,
auth)`` supplies the headers the gateway would otherwise inject (sticky key = served model name, auth =
``--api_key``). Like the reranker example it retries until the replica finishes loading the weights.

REFERENCE-ONLY, matching deploy.sh: point the server at a trained seq_cls checkpoint (and set LABELS below
to its class names) before running -- a base encoder's random head yields meaningless logits.

Run the server first (``bash examples/deploy/seq_cls/deploy.sh``), then:
    python examples/deploy/seq_cls/client.py
"""
from __future__ import annotations

import os
import time

import httpx

from twinkle_client.http.headers import build_routing_headers

BASE_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://127.0.0.1:8000/api/v1')
API_KEY = os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY')
# Must equal the deployment's --served_model_name; also the sticky routing key.
MODEL = os.environ.get('TWINKLE_MODEL', 'classifier')
# Match these to the checkpoint's --num_labels and its training label order.
LABELS = ['negative', 'positive']

TEXTS = [
    'This product is fantastic, I would buy it again.',
    'Arrived broken and support never replied. Total waste.',
]


def encode_classify(texts: list[str]) -> list[list[float]]:
    url = f'{BASE_URL}/sampler/{MODEL}/twinkle/encode'
    headers = build_routing_headers(MODEL, API_KEY)
    body = {
        'inputs': [{'messages': [{'role': 'user', 'content': t}]} for t in texts],
        'pooling_params': {'task': 'classify'},
        'adapter_name': MODEL,
    }
    last_err: str | None = None
    for _ in range(120):
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=120.0)
            if resp.status_code == 200:
                return [list(map(float, row)) for row in resp.json()['data']]
            last_err = f'HTTP {resp.status_code}: {resp.text[:300]}'
        except Exception as exc:  # noqa: BLE001 - connection refused while the replica warms up
            last_err = repr(exc)
        time.sleep(5)
    raise RuntimeError(f'seq_cls sampler never became ready: {last_err}')


def main() -> None:
    logits_per_text = encode_classify(TEXTS)
    for text, logits in zip(TEXTS, logits_per_text):
        pred = max(range(len(logits)), key=lambda i: logits[i])
        label = LABELS[pred] if pred < len(LABELS) else f'label_{pred}'
        print(f'{text!r}\n  logits={[round(x, 4) for x in logits]} -> {label}')


if __name__ == '__main__':
    main()
