# Copyright (c) ModelScope Contributors. All rights reserved.
# The infer dataset-output component: generative sampling + PRM/ORM scoring + row writing, plus the
# pooling and generative-reranker forward passes (one value per row) -- all inside InferComponent.
from __future__ import annotations

from .component import InferComponent

__all__ = ['InferComponent']
