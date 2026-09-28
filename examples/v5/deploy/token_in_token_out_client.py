"""token-in-token-out client for ``swift deploy``.

``swift deploy`` serves a twinkle-server on Ray Serve. Besides the OpenAI surface (``/v1/chat/completions``
and friends, which any OpenAI SDK drives), the same deployment exposes the *sampler* directly, so a
training loop can feed token ids in and get trainable features out -- no text round-trip, no re-tokenizing
on the server. This sketch shows both sampler paths against the server started by
``examples/v5/deploy/serve_token_in_token_out.sh``:

- **Lightweight** (:func:`sample_token_in_token_out`): ``sampler.sample(InputFeature(input_ids=...))`` and
  read each sequence's ``new_input_feature`` -- the full prompt+completion as trainable features
  (``input_ids`` / ``labels`` / ``completion_mask``). Needs no data plane.
- **RL rollout** (:func:`rollout_to_data_plane`): with ``--enable_data_plane``, ``asample_to_data_plane``
  keeps the generated group server-side and hands back a ``DataRef``; the client scores it, appends the
  rewards/advantages, and passes the ref to ``model.forward_backward_from_data_plane`` -- the loop
  ``twinkle/cookbook/client/async_rl/client_orchestrated_grpo.py`` runs.

The multi-turn conversation and the training step stay here in the client; the server only answers one
request. The template is already pinned server-side from ``TemplateConfig``, so the client does NOT call
``set_template``.
"""
from __future__ import annotations

import asyncio
import os

from twinkle.data_format import InputFeature
from twinkle_client import DataPlaneClient, init_twinkle_client
from twinkle_client.sampler import vLLMSampler

# The client's base_url is the gateway root INCLUDING its route_prefix: the sampler is mounted at
# '{route_prefix}/sampler/{served_model_name}' and the data plane at '{route_prefix}/data-plane', both of
# which the client rebuilds off base_url. This is the URL run_deploy_process yields.
BASE_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://127.0.0.1:8000/v1')
API_KEY = os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY')
# Must equal the deployment's --served_model_name (the sampler is mounted under it).
MODEL = os.environ.get('TWINKLE_MODEL', 'policy')
# Set when the server was started with --enable_data_plane; gates the RL-rollout path.
ENABLE_DATA_PLANE = os.environ.get('TWINKLE_ENABLE_DATA_PLANE', '1') == '1'

#: A stand-in prompt as token ids. A real loop encodes with the model's template; the sampler accepts the
#: raw ``input_ids`` either way, which is the point of token-in-token-out.
PROMPT_TOKEN_IDS = [151644, 872, 198, 9707, 151645, 198]


def sample_token_in_token_out(sampler: vLLMSampler, prompt_token_ids: list[int]) -> list:
    """Lightweight path: token ids in, trainable features out, no data plane."""
    responses = sampler.sample(
        [InputFeature(input_ids=list(prompt_token_ids))],
        sampling_params={'max_tokens': 64, 'temperature': 1.0, 'logprobs': 1},
    )
    for response in responses:
        for sequence in response.sequences:
            feature = sequence.new_input_feature or {}
            print(f'[lightweight] stop={sequence.stop_reason} new_tokens={len(sequence.tokens)} '
                  f'trainable_len={len(feature.get("input_ids") or [])}')
    return responses


async def rollout_to_data_plane(sampler: vLLMSampler, data_plane: DataPlaneClient, prompt_token_ids: list[int],
                                num_generations: int = 4):
    """RL-rollout path: keep the generated group server-side, score it, and append the training targets.

    Returns the ``DataRef`` a trainer feeds to ``model.forward_backward_from_data_plane([ref], ...)``; the
    caller releases it after the step. Mirrors the advantage worker in ``client_orchestrated_grpo.py``.
    """
    ref = await sampler.asample_to_data_plane(
        [InputFeature(input_ids=list(prompt_token_ids))],
        sampling_params={'max_tokens': 512, 'temperature': 1.0, 'logprobs': 1},
        num_samples=num_generations,
        group_ids=['group-0'],
    )
    rows = await data_plane.aget(ref, fields=['decoded'])
    print(f'[data-plane] generated {len(rows)} sequences server-side')
    # Score on the client (your reward fn), then append the per-sequence training targets back onto the
    # same ref; the trainer reads them via kwarg_fields={'advantages': 'advantage', ...}.
    rewards = [1.0 if (row.get('decoded') or '').strip() else 0.0 for row in rows]
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    return await data_plane.aappend(
        ref, [{'reward': float(r), 'advantage': float(r - mean)} for r in rewards])


async def main() -> None:
    client = init_twinkle_client(base_url=BASE_URL, api_key=API_KEY)
    try:
        sampler = vLLMSampler(MODEL)
        sample_token_in_token_out(sampler, PROMPT_TOKEN_IDS)
        if ENABLE_DATA_PLANE:
            data_plane = DataPlaneClient()
            ref = await rollout_to_data_plane(sampler, data_plane, PROMPT_TOKEN_IDS)
            # A real loop now trains and then releases:
            #   await model.forward_backward_from_data_plane([ref], input_field='train_input',
            #       kwarg_fields={'old_logps': 'sampled_logprobs', 'advantages': 'advantage'})
            #   await model.clip_grad_and_step(max_grad_norm=1.0)
            await data_plane.arelease(ref)
            print('[data-plane] released the rollout ref')
        else:
            print('[data-plane] skipped: start the server with --enable_data_plane to exercise this path')
    finally:
        client.close()


if __name__ == '__main__':
    asyncio.run(main())
