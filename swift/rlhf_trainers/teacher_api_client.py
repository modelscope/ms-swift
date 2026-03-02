# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client for fetching teacher model logprobs from OpenAI-compatible endpoints.

Supports swift deploy (vLLM backend) and standalone vLLM servers.
Used for knowledge distillation (GKD) training with top-k logprobs.
"""
import logging
from typing import Dict, List, Optional, Tuple

import requests
import torch

logger = logging.getLogger(__name__)


class TeacherAPIClient:
    """Fetch teacher top-k logprobs from an OpenAI-compatible completions API.

    Args:
        base_url: Server URL (e.g., 'http://localhost:8000').
        top_logprobs: Number of top log probabilities per token.
        timeout: Request timeout in seconds.
        api_key: Optional API key for authentication.
        model_name: Model name for API requests. Auto-detected if None.
    """

    def __init__(
        self,
        base_url: str,
        top_logprobs: int = 20,
        timeout: float = 300.0,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip('/')
        self.top_logprobs = top_logprobs
        self.timeout = timeout
        self.api_key = api_key
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        if self._model_name is None:
            self._model_name = self._detect_model_name()
        return self._model_name

    def _headers(self) -> Dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    def _detect_model_name(self) -> str:
        try:
            resp = requests.get(f'{self.base_url}/v1/models', headers=self._headers(), timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('data'):
                    return data['data'][0]['id']
        except Exception as e:
            logger.warning(f'Failed to detect model name: {e}')
        return 'default'

    def check_server_health(self, timeout: float = 5.0) -> bool:
        """Check if the teacher model server is reachable."""
        for endpoint in ['/health', '/v1/models']:
            try:
                resp = requests.get(f'{self.base_url}{endpoint}', timeout=timeout)
                if resp.status_code == 200:
                    return True
            except requests.RequestException:
                continue
        return False

    def get_logprobs_sync(
        self,
        input_ids: List[List[int]],
        top_logprobs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch top-k logprobs for a batch of token sequences.

        Args:
            input_ids: List of token ID sequences.
            top_logprobs: Override default top_logprobs.

        Returns:
            (logprobs, indices) tensors of shape [batch, max_seq_len, topk].
        """
        topk = top_logprobs or self.top_logprobs
        batch_size = len(input_ids)
        max_seq_len = max(len(ids) for ids in input_ids)

        logprobs_tensor = torch.full((batch_size, max_seq_len, topk), float('-inf'), dtype=torch.float32)
        indices_tensor = torch.zeros((batch_size, max_seq_len, topk), dtype=torch.long)

        url = f'{self.base_url}/v1/completions'
        model = self.model_name

        for batch_idx, ids in enumerate(input_ids):
            payload = {
                'model': model,
                'prompt': ids,
                'max_tokens': 0,
                'temperature': 0,
                'logprobs': topk,
                'echo': True,
            }
            try:
                resp = requests.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
                if resp.status_code != 200:
                    logger.error(f'API error for sequence {batch_idx}: {resp.status_code} - {resp.text}')
                    continue
                self._parse_into_tensors(resp.json(), batch_idx, logprobs_tensor, indices_tensor, topk)
            except Exception as e:
                logger.error(f'Failed to get logprobs for sequence {batch_idx}: {e}')

        return logprobs_tensor, indices_tensor

    @staticmethod
    def _parse_into_tensors(
        response: dict,
        batch_idx: int,
        logprobs_out: torch.Tensor,
        indices_out: torch.Tensor,
        topk: int,
    ) -> None:
        """Parse a single completions API response into pre-allocated tensors."""
        choices = response.get('choices', [])
        if not choices:
            return
        logprobs_data = choices[0].get('logprobs') or {}
        top_logprobs_list = logprobs_data.get('top_logprobs', [])

        for pos_idx, pos_logprobs in enumerate(top_logprobs_list):
            if pos_logprobs is None:
                continue
            sorted_items = sorted(
                pos_logprobs.items(),
                key=lambda x: -(x[1] if isinstance(x[1], (int, float)) else
                                (x[1].get('logprob', float('-inf')) if isinstance(x[1], dict) else float('-inf'))),
            )[:topk]
            for k_idx, (token_id_str, logprob_val) in enumerate(sorted_items):
                try:
                    indices_out[batch_idx, pos_idx, k_idx] = int(token_id_str)
                    if isinstance(logprob_val, (int, float)):
                        logprobs_out[batch_idx, pos_idx, k_idx] = logprob_val
                    elif isinstance(logprob_val, dict):
                        logprobs_out[batch_idx, pos_idx, k_idx] = logprob_val.get('logprob', float('-inf'))
                    elif hasattr(logprob_val, 'logprob'):
                        logprobs_out[batch_idx, pos_idx, k_idx] = logprob_val.logprob
                except (ValueError, TypeError):
                    continue
