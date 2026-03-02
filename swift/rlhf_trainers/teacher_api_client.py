# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client for fetching teacher model logprobs from OpenAI-compatible endpoints."""
import logging
import requests
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class TeacherAPIClient:
    """Fetch teacher top-k logprobs from an OpenAI-compatible completions API.

    Args:
        base_url: Server URL (e.g., 'http://localhost:8000').
        top_logprobs: Number of top log probabilities per token.
        timeout: Request timeout in seconds.
    """

    def __init__(self, base_url: str, top_logprobs: int = 20, timeout: float = 300.0):
        self.base_url = base_url.rstrip('/')
        self.top_logprobs = top_logprobs
        self.timeout = timeout
        self._model_name = None

    @property
    def model_name(self) -> str:
        if self._model_name is None:
            try:
                resp = requests.get(f'{self.base_url}/v1/models', timeout=10)
                if resp.ok and resp.json().get('data'):
                    self._model_name = resp.json()['data'][0]['id']
            except Exception as e:
                logger.warning(f'Failed to detect model name: {e}')
            if self._model_name is None:
                self._model_name = 'default'
        return self._model_name

    def check_health(self, timeout: float = 5.0) -> bool:
        """Check if the teacher model server is reachable."""
        try:
            resp = requests.get(f'{self.base_url}/v1/models', timeout=timeout)
            return resp.ok
        except requests.RequestException:
            return False

    def get_logprobs_sync(
        self,
        input_ids: List[List[int]],
        top_logprobs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch top-k logprobs for a batch of token sequences.

        Returns:
            (logprobs, indices) tensors of shape [batch, max_seq_len, topk].
        """
        topk = top_logprobs or self.top_logprobs
        batch_size = len(input_ids)
        max_seq_len = max(len(ids) for ids in input_ids)
        url = f'{self.base_url}/v1/completions'
        model = self.model_name

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
                resp = requests.post(url, json=payload, timeout=self.timeout)
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
