# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client for fetching teacher model logprobs from swift deploy or vLLM server.

This module provides a client for communicating with OpenAI-compatible endpoints
(e.g., swift deploy with vLLM backend, standalone vLLM server) to obtain teacher
model logprobs for knowledge distillation (GKD) training.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import torch

logger = logging.getLogger(__name__)


class TeacherAPIClient:
    """Client for fetching teacher logprobs from swift deploy or vLLM server.

    This client is designed to work with OpenAI-compatible API endpoints:
    - swift deploy (with vLLM backend)
    - Standalone vLLM server (vllm serve)

    The client fetches top-k log probabilities for each token position,
    which are then used for knowledge distillation (GKD) training.

    Args:
        base_url: The base URL of the teacher model server (e.g., 'http://localhost:8000').
        top_logprobs: Number of top log probabilities to request per token.
        timeout: Request timeout in seconds.
        api_key: Optional API key for authentication.
        model_name: Optional model name for the API request. If None, auto-detects.
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
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.api_key = api_key
        self.model_name = model_name

        if top_logprobs <= 0:
            raise ValueError(f'top_logprobs must be positive, got {top_logprobs}')

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    async def _get_model_name(self, session: aiohttp.ClientSession) -> str:
        """Get model name from server if not provided."""
        if self.model_name:
            return self.model_name

        try:
            async with session.get(
                    f'{self.base_url}/v1/models', headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('data') and len(data['data']) > 0:
                        self.model_name = data['data'][0]['id']
                        return self.model_name
        except Exception as e:
            logger.warning(f'Failed to get model name: {e}')

        self.model_name = 'default'
        return self.model_name

    async def get_logprobs_batch(
        self,
        input_ids: List[List[int]],
        top_logprobs: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch logprobs for a batch of sequences using OpenAI-compatible API.

        Args:
            input_ids: List of token ID sequences.
            top_logprobs: Override the default top_logprobs if provided.

        Returns:
            List of dictionaries, each containing:
            - 'indices': List of token indices per position [seq_len, topk]
            - 'values': List of logprob values per position [seq_len, topk]
        """
        topk = top_logprobs or self.top_logprobs

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            model_name = await self._get_model_name(session)
            url = f'{self.base_url}/v1/completions'

            results = []
            for i, ids in enumerate(input_ids):
                # Use prompt tokens and request logprobs with echo
                payload = {
                    'model': model_name,
                    'prompt': ids,
                    'max_tokens': 0,
                    'temperature': 0,
                    'logprobs': topk,
                    'echo': True,
                }

                try:
                    async with session.post(url, json=payload, headers=self._get_headers()) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            logger.error(f'API error: {resp.status} - {error_text}')
                            results.append(self._empty_result(len(ids), topk))
                            continue

                        data = await resp.json()
                        parsed = self._parse_response(data, len(ids), topk)
                        results.append(parsed)
                except Exception as e:
                    logger.error(f'Failed to get logprobs for sequence {i}: {e}')
                    results.append(self._empty_result(len(ids), topk))

            return results

    def _parse_response(self, response: Dict[str, Any], seq_len: int, topk: int) -> Dict[str, Any]:
        """Parse vLLM completions API response to extract logprobs.

        vLLM returns logprobs in two formats:
        1. `prompt_logprobs`: List of dicts where keys are token IDs (as strings), values have 'logprob' field
        2. `top_logprobs` in logprobs: List of dicts where keys are token text

        We prefer `prompt_logprobs` because it has token IDs directly.
        """
        result = {'indices': [], 'values': []}

        try:
            if 'choices' not in response or len(response['choices']) == 0:
                return self._empty_result(seq_len, topk)

            choice = response['choices'][0]

            # Try prompt_logprobs first (vLLM native format with token IDs as keys)
            prompt_logprobs = choice.get('prompt_logprobs')
            if prompt_logprobs is not None:
                for pos_idx, pos_logprobs in enumerate(prompt_logprobs):
                    pos_indices = []
                    pos_values = []

                    if pos_logprobs is not None:
                        # vLLM format: {token_id_str: {logprob: float, ...}, ...}
                        sorted_items = sorted(pos_logprobs.items(), key=lambda x: -self._get_logprob_value(x[1]))[:topk]

                        for token_id_str, logprob_data in sorted_items:
                            try:
                                token_id = int(token_id_str)
                                pos_indices.append(token_id)
                                pos_values.append(self._get_logprob_value(logprob_data))
                            except (ValueError, TypeError):
                                continue

                    # Pad if needed
                    while len(pos_indices) < topk:
                        pos_indices.append(0)
                        pos_values.append(float('-inf'))

                    result['indices'].append(pos_indices)
                    result['values'].append(pos_values)

                # Pad to seq_len if needed
                while len(result['indices']) < seq_len:
                    result['indices'].append([0] * topk)
                    result['values'].append([float('-inf')] * topk)

                return result

            # Fallback to logprobs.top_logprobs (OpenAI format, keys are token text)
            logprobs_data = choice.get('logprobs', {})
            if logprobs_data is None:
                return self._empty_result(seq_len, topk)

            top_logprobs_list = logprobs_data.get('top_logprobs', [])

            for pos_idx, pos_logprobs in enumerate(top_logprobs_list):
                pos_indices = []
                pos_values = []

                if pos_logprobs is not None:
                    sorted_items = sorted(pos_logprobs.items(), key=lambda x: -self._get_logprob_value(x[1]))[:topk]

                    for token_str, logprob in sorted_items:
                        try:
                            token_id = int(token_str)
                            pos_indices.append(token_id)
                            pos_values.append(self._get_logprob_value(logprob))
                        except (ValueError, TypeError):
                            # Token is text, not ID - skip (can't use without tokenizer)
                            continue

                # Pad if needed
                while len(pos_indices) < topk:
                    pos_indices.append(0)
                    pos_values.append(float('-inf'))

                result['indices'].append(pos_indices)
                result['values'].append(pos_values)

            # Pad to seq_len if needed
            while len(result['indices']) < seq_len:
                result['indices'].append([0] * topk)
                result['values'].append([float('-inf')] * topk)

        except Exception as e:
            logger.error(f'Failed to parse response: {e}')
            return self._empty_result(seq_len, topk)

        return result

    @staticmethod
    def _get_logprob_value(logprob) -> float:
        """Extract logprob value from vLLM response (handles both float and Logprob object)."""
        if isinstance(logprob, (int, float)):
            return float(logprob)
        elif hasattr(logprob, 'logprob'):
            return float(logprob.logprob)
        elif isinstance(logprob, dict) and 'logprob' in logprob:
            return float(logprob['logprob'])
        return float('-inf')

    def _empty_result(self, seq_len: int, topk: int) -> Dict[str, Any]:
        """Return empty result for failed requests."""
        return {
            'indices': [[0] * topk for _ in range(seq_len)],
            'values': [[float('-inf')] * topk for _ in range(seq_len)],
        }

    def check_server_health(self, timeout: float = 5.0) -> bool:
        """Check if the teacher model server is healthy."""
        import requests
        try:
            for endpoint in ['/health', '/v1/models']:
                try:
                    response = requests.get(f'{self.base_url}{endpoint}', timeout=timeout)
                    if response.status_code == 200:
                        return True
                except requests.RequestException:
                    continue
            return False
        except Exception as e:
            logger.warning(f'Health check failed: {e}')
            return False

    def get_logprobs_sync(
        self,
        input_ids: List[List[int]],
        top_logprobs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Synchronous wrapper for get_logprobs_batch.

        Args:
            input_ids: List of token ID sequences
            top_logprobs: Number of top logprobs to fetch

        Returns:
            Tuple of (logprobs_tensor, indices_tensor) with shapes [batch, seq_len, topk]
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.get_logprobs_batch(input_ids, top_logprobs))
                    results = future.result()
            else:
                results = loop.run_until_complete(self.get_logprobs_batch(input_ids, top_logprobs))
        except RuntimeError:
            results = asyncio.run(self.get_logprobs_batch(input_ids, top_logprobs))

        # Convert to tensors
        topk = top_logprobs or self.top_logprobs
        batch_size = len(input_ids)
        max_seq_len = max(len(ids) for ids in input_ids)

        logprobs_tensor = torch.full((batch_size, max_seq_len, topk), float('-inf'), dtype=torch.float32)
        indices_tensor = torch.zeros((batch_size, max_seq_len, topk), dtype=torch.long)

        for batch_idx, result in enumerate(results):
            indices = result.get('indices', [])
            values = result.get('values', [])
            for pos_idx, (pos_indices, pos_values) in enumerate(zip(indices, values)):
                if pos_idx >= max_seq_len:
                    break
                for k_idx in range(min(len(pos_indices), topk)):
                    indices_tensor[batch_idx, pos_idx, k_idx] = pos_indices[k_idx]
                    logprobs_tensor[batch_idx, pos_idx, k_idx] = pos_values[k_idx]

        return logprobs_tensor, indices_tensor
