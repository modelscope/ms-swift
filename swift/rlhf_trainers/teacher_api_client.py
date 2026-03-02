# Copyright (c) ModelScope Contributors. All rights reserved.
"""Fetch teacher model logprobs from OpenAI-compatible endpoints."""
import logging
import requests
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_model_name_cache: dict = {}


def _get_model_name(base_url: str) -> str:
    if base_url not in _model_name_cache:
        try:
            resp = requests.get(f'{base_url}/v1/models', timeout=10)
            if resp.ok and resp.json().get('data'):
                _model_name_cache[base_url] = resp.json()['data'][0]['id']
        except Exception as e:
            logger.warning(f'Failed to detect model name: {e}')
        if base_url not in _model_name_cache:
            _model_name_cache[base_url] = 'default'
    return _model_name_cache[base_url]


def fetch_teacher_logprobs(
    base_url: str,
    input_ids: List[List[int]],
    topk: int = 20,
    timeout: float = 300.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fetch top-k logprobs from an OpenAI-compatible completions API.

    Args:
        base_url: Server URL (e.g., 'http://localhost:8000').
        input_ids: List of token ID sequences.
        topk: Number of top log probabilities per token.
        timeout: Request timeout in seconds.

    Returns:
        (logprobs, indices) tensors of shape [batch, max_seq_len, topk].
    """
    base_url = base_url.rstrip('/')
    batch_size = len(input_ids)
    max_seq_len = max(len(ids) for ids in input_ids)
    url = f'{base_url}/v1/completions'
    model = _get_model_name(base_url)

    logprobs_out = torch.full((batch_size, max_seq_len, topk), float('-inf'), dtype=torch.float32)
    indices_out = torch.zeros((batch_size, max_seq_len, topk), dtype=torch.long)

    def _fetch_one(batch_idx: int):
        payload = {
            'model': model,
            'prompt': input_ids[batch_idx],
            'max_tokens': 0,
            'temperature': 0,
            'logprobs': topk,
            'echo': True,
        }
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            top_logprobs_list = resp.json()['choices'][0].get('logprobs', {}).get('top_logprobs', [])
            for pos, pos_lp in enumerate(top_logprobs_list):
                if pos_lp is None:
                    continue
                sorted_items = sorted(pos_lp.items(), key=lambda x: -x[1])[:topk]
                for k, (tid_str, lp) in enumerate(sorted_items):
                    indices_out[batch_idx, pos, k] = int(tid_str)
                    logprobs_out[batch_idx, pos, k] = lp
        except Exception as e:
            logger.error(f'Failed to get logprobs for sequence {batch_idx}: {e}')

    with ThreadPoolExecutor(max_workers=min(batch_size, 8)) as pool:
        list(pool.map(_fetch_one, range(batch_size)))

    return logprobs_out, indices_out
