"""Minimal OpenAI embeddings client for ``swift deploy`` serving a pooling model.

``examples/deploy/embedding/deploy.sh`` serves an embedding checkpoint behind the standard OpenAI
``/embeddings`` route, so any OpenAI SDK drives it: point ``base_url`` at the gateway root INCLUDING its
route prefix (``/api/v1`` here) and pass the deployment's ``--served_model_name`` as the model. Each input
string comes back as one pooled vector whose width is the model's hidden size (768 for embeddinggemma).

The gateway wraps every input as a single-user-turn trajectory and the sampler runs the pooling head, which
is why the server is started with ``--use_chat_template false`` (an embedding checkpoint has no chat
template). Cosine similarity between two vectors is the usual retrieval signal; embeddinggemma pools with
MEAN and the server leaves vectors un-normalised, so normalise before comparing.

Run the server first (``bash examples/deploy/embedding/deploy.sh``), then:
    python examples/deploy/embedding/client.py
"""
from __future__ import annotations

import math
import os

from openai import OpenAI

# The gateway root including its --route_prefix; the OpenAI SDK appends /embeddings itself.
BASE_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://127.0.0.1:8000/api/v1')
API_KEY = os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY')
# Must equal the deployment's --served_model_name.
MODEL = os.environ.get('TWINKLE_MODEL', 'embedder')


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    texts = [
        'What is the capital of China?',
        'The capital of China is Beijing.',
        'Bananas are a good source of potassium.',
    ]
    # One request embeds the whole batch; the response preserves input order under data[i].index.
    resp = client.embeddings.create(model=MODEL, input=texts)
    vectors = [d.embedding for d in resp.data]
    print(f'model={resp.model} n={len(vectors)} dim={len(vectors[0])}')
    for text, vec in zip(texts, vectors):
        print(f'  {text!r:45s} -> [{vec[0]: .4f}, {vec[1]: .4f}, ...]')

    # The query should be closer to the relevant sentence than to the unrelated one.
    q, relevant, unrelated = vectors
    print(f'cos(query, relevant)   = {cosine(q, relevant):.4f}')
    print(f'cos(query, unrelated)  = {cosine(q, unrelated):.4f}')


if __name__ == '__main__':
    main()
