"""Classify client for ``swift deploy`` serving a text classifier (seq_cls).

The single-text sibling of ``examples/v5/deploy/reranker/client.py``: it drives the same sampler
``/twinkle/encode`` route with the same request identity/auth headers, but sends one text per input and
reads back one logit per class (``--num_labels`` of them), then takes the argmax as the predicted label.

classify has no OpenAI gateway endpoint (the gateway's ``/embeddings`` hardcodes the pooling task to
``embed``), so the client posts to the sampler directly:

    {route_prefix}/sampler/{served_model_name}/twinkle/encode   with  pooling_params {'task': 'classify'}

Routing is by URL: each served model has its own single-copy sampler mounted at
``{route_prefix}/sampler/{served_model_name}`` (the old Ray Serve multiplexed sticky routing is gone).
The client only supplies the identity/auth headers the gateway would otherwise inject:
``twinkle.protocol.headers.build_request_headers(request_id, 'Bearer ' + api_key)`` sets ``x-request-id``
(required by the server's auth middleware) and ``Authorization``/``Twinkle-Authorization``. The reply is
a TaskEnvelope; a completed task carries the per-input logits under ``result['data']``. Like the
reranker example it retries until the replica finishes loading the weights.

REFERENCE-ONLY, matching deploy.sh: point the server at a trained seq_cls checkpoint (and set LABELS below
to its class names) before running -- a base encoder's random head yields meaningless logits.

Run the server first (``bash examples/v5/deploy/seq_cls/deploy.sh``), then:
    python examples/v5/deploy/seq_cls/client.py
"""
from __future__ import annotations
import os
import time
import uuid

import httpx
from twinkle.protocol.headers import build_request_headers

BASE_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://127.0.0.1:8000/api/v1')
API_KEY = os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY')
# Must equal the deployment's --served_model_name; it names the sampler mount and the adapter_name.
MODEL = os.environ.get('TWINKLE_MODEL', 'classifier')
# Match these to the checkpoint's --num_labels and its training label order.
LABELS = ['negative', 'positive']

TEXTS = [
    'This product is fantastic, I would buy it again.',
    'Arrived broken and support never replied. Total waste.',
]


def encode_classify(texts: list[str]) -> list[list[float]]:
    url = f'{BASE_URL}/sampler/{MODEL}/twinkle/encode'
    headers = build_request_headers(uuid.uuid4().hex, f'Bearer {API_KEY}')
    body = {
        'inputs': [{'messages': [{'role': 'user', 'content': t}]} for t in texts],
        'pooling_params': {'task': 'classify'},
        'adapter_name': MODEL,
    }
    last_err: str | None = None
    for _ in range(120):
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=120.0)
            envelope = resp.json() if resp.status_code == 200 else {}
            if envelope.get('status') == 'completed':
                return [list(map(float, row)) for row in envelope['result']['data']]
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
