import hashlib
import inspect
from copy import copy
from typing import Any, Dict, List, Optional

import json
import numpy as np

from swift.llm import InferRequest, RequestConfig
from swift.utils import get_logger

logger = get_logger()


def get_messages_md5(row: Dict[str, Any]):
    row = copy(row)
    row.pop('choices', None)
    serialized = json.dumps(row, sort_keys=True)
    return hashlib.md5(serialized.encode('utf-8')).hexdigest()


def get_reward(model: Any,
               infer_requests: List[InferRequest],
               request_config: RequestConfig = None,
               ground_truths: List[str] = None,
               threshold: Optional[float] = None):
    """Get reward from an RM model.

    Args:
        model: The model instance or an RM evaluator
        infer_requests: Infer requests sent to the model
        request_config: Infer config
        ground_truths: The ground truth list
        threshold: An optional threshold to generate the mask

    Returns:
        Tuple
        Index 0: The min-max normalized scores matched the infer_requests
        Index 1: The mask filtered by the threshold
    """
    from swift.llm import InferEngine
    infer_func = model.infer if isinstance(model, InferEngine) else model.__call__
    parameters = inspect.signature(infer_func).parameters
    gt_param = {}
    if 'ground_truths' in parameters:
        gt_param = {'ground_truths': ground_truths}
    if isinstance(infer_requests[0], dict):
        infer_requests = [InferRequest(messages=req['messages']) for req in infer_requests]
    rewards = infer_func(infer_requests, request_config=request_config, **gt_param)
    from swift.llm.infer.protocol import ChatCompletionResponse
    if isinstance(rewards[0], ChatCompletionResponse):
        print('reward:', rewards[0].choices[0].message.content)
        if isinstance(rewards[0].choices[0].message.content, str):
            rewards = [float(r.choices[0].message.content.strip('[]')) for r in rewards]
        elif isinstance(rewards[0].choices[0].message.content, list):
            rewards = [float(min(r.choices[0].message.content)) for r in rewards]
        else:
            rewards = [float(r.choices[0].message.content) for r in rewards]
    arr = []
    for reward in rewards:
        if isinstance(reward, (list, tuple)):
            arr.append(min(reward))
        else:
            arr.append(float(reward))

    _mask = np.array([True] * len(arr))
    if threshold is not None:
        # > not >=, orm caller passes 0, which will cause error
        _mask = np.array([a > threshold for a in arr])

    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if min_val == max_val:
            if min_val == 0:
                constant_value = 0.0
            else:
                constant_value = min(1.0, min_val)
            return np.full_like(arr, fill_value=constant_value, dtype=np.float64)
        normalized = (arr - min_val) / (max_val - min_val + 1e-5)
        return normalized

    return normalize(arr), _mask
