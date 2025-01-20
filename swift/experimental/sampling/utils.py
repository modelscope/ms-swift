import hashlib
import inspect
from typing import Any, List, Optional

import json
import numpy as np

from swift.llm import InferRequest, Messages, RequestConfig


def get_messages_md5(messages: Messages):
    key = ''.join([m['content'] for m in messages])
    return hashlib.md5(key.encode('utf-8')).hexdigest()


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
    parameters = inspect.signature(model.infer).parameters
    gt_param = {}
    if 'ground_truths' in parameters:
        gt_param = {'ground_truths': ground_truths}
    resp_list = model.infer(infer_requests, request_config=request_config, **gt_param)
    arr = []
    for i in range(len(resp_list)):
        content = resp_list[i].choices[0].message.content
        if isinstance(content, str) and '[' in content:
            try:
                content = json.loads(content)
            except Exception:
                content = eval(content)
            arr.append(min(content))
        else:
            arr.append(float(content))

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
