# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from typing import List, Literal, Optional, Tuple

import json

from ..utils import ContextType, Messages, get_last_user_round
from .utils import calculate_loss_scale

ALL_BASE_STRATEGY = ["default", "last_round", "all"]


class LossScale:
    """Weigh which rendered parts contribute to the loss.

    ``base_strategy``: ``'default'`` (assistant responses only), ``'last_round'`` (last response only),
    ``'all'`` (everything). ``is_binary=True`` means weights are only 0/1 and live in ``labels``
    (-100 masks); otherwise a per-token ``loss_scale`` field is emitted for a weighted loss.
    """

    is_binary = True

    def __init__(self, base_strategy: Literal["default", "last_round", "all"] = "default"):
        if base_strategy not in ALL_BASE_STRATEGY:
            raise ValueError(f"ALL_BASE_STRATEGY: {ALL_BASE_STRATEGY}, base_strategy: {base_strategy}")
        self.base_strategy = base_strategy

    def get_loss_scale(self, context: str, **kwargs) -> Tuple[List[str], List[float]]:
        """Split one context into (parts, weights); the base keeps it whole at weight 1."""
        return [context], [1.0]

    def __call__(
        self, context_list: List[str], context_types: List[ContextType], messages: Messages, **kwargs
    ) -> Tuple[List[str], List[float]]:
        """Weigh the rendered conversation; returns ``(contexts, weights)`` aligned 1:1.

        A RESPONSE segment maps to ``messages[2 * i + 1]`` (strict query/response pairs), whose
        per-message ``loss`` / ``loss_scale`` override the strategy: ``loss=1`` forces training,
        ``loss=0`` forces masking, ``None`` defers to ``base_strategy``. Segments may split.
        """
        res_context_list = []
        res_loss_scale = []
        i = 0
        last_user_round = get_last_user_round(messages)
        for context, context_type in zip(context_list, context_types):
            is_last_round = 2 * i >= last_user_round
            query, loss, loss_scale = None, None, None
            if context_type == ContextType.RESPONSE:
                query = messages[2 * i]["content"]
                # Currently, we only support applying loss/mask to the response part.
                loss = messages[2 * i + 1].get("loss")
                loss_scale = messages[2 * i + 1].get("loss_scale")
                assert context == messages[2 * i + 1]["content"]
                i += 1
            if not isinstance(context, list) or (len(context) > 0 and isinstance(context[0], int)):
                context = [context]
            for j in range(len(context)):
                new_context, new_loss_scale = self._inner_call(
                    context[j],
                    context_type,
                    query,
                    loss[j] if isinstance(loss, list) else loss,
                    loss_scale[j] if isinstance(loss_scale, list) else loss_scale,
                    is_last_round,
                )
                res_context_list += new_context
                res_loss_scale += new_loss_scale

        # The values in loss_scale_list correspond one-to-one with the values in context_list.
        return res_context_list, res_loss_scale

    def _inner_call(self, context, context_type, query, loss, loss_scale, is_last_round):
        if isinstance(context, dict) and "loss_scale" in context:
            new_context = [[token] for token in context["token_ids"]]
            loss_scale = context["loss_scale"]
        else:
            if isinstance(context, dict) and "token_ids" in context:
                context = context["token_ids"]
            is_assistant = context_type in {ContextType.RESPONSE, ContextType.SUFFIX}
            if (
                loss
                or loss is None
                and (
                    self.base_strategy == "all"
                    or (self.base_strategy == "default" and is_assistant)
                    or (self.base_strategy == "last_round" and is_assistant and is_last_round)
                )
            ):
                if loss_scale is None:
                    new_context, loss_scale = self.get_loss_scale(context, query=query)
                else:
                    new_context, loss_scale = [context], [loss_scale]
            else:
                new_context, loss_scale = [context], [0.0]
        return new_context, loss_scale

    @property
    def is_binary_loss_scale(self):
        """Check if loss scale values are binary (only 0 and 1)."""
        return self.is_binary


class ConfigLossScale(LossScale):
    """Weights come from ``config/<loss_scale_config>`` (a JSON map of delimiters -> weights)."""

    is_binary = None
    loss_scale_config = None  # path

    def __init__(self, base_strategy: Literal["default", "last_round", "all"] = "default"):
        super().__init__(base_strategy)
        self.loss_scale_map = None
        if self.loss_scale_config is not None:
            path = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(path, "config", self.loss_scale_config)
            with open(config_path, "r", encoding="utf-8") as json_file:
                self.loss_scale_map = json.load(json_file)

    @property
    def is_binary_loss_scale(self):
        if self.is_binary is not None:
            return self.is_binary
        if self.loss_scale_map is None:
            return True
        return all(scale in {0.0, 1.0} for lst in self.loss_scale_map.values() for scale in lst)

    def get_loss_scale(self, context: str, *, query: Optional[str] = None, **kwargs):
        """String contexts split by the JSON map (query shortcuts + delimiter weights)."""
        if isinstance(context, str):
            return calculate_loss_scale(query, context, self.loss_scale_map)
        return super().get_loss_scale(context)


class ConcatLossScale(LossScale):
    """Apply multiple loss scales sequentially.

    The output segments of each underlying loss scale are fed into the next one,
    and the corresponding weights are multiplied together. This makes it possible
    to compose strategies such as ``hermes+ignore_empty_think``.
    """

    is_binary = None

    def __init__(
        self, loss_scales: List[LossScale], base_strategy: Literal["default", "last_round", "all"] = "default"
    ):
        super().__init__(base_strategy)
        assert loss_scales, "loss_scales must be a non-empty list"
        self.loss_scales = loss_scales

    def get_loss_scale(self, context, **kwargs):
        contexts = [context]
        weights = [1.0]
        for ls in self.loss_scales:
            new_contexts: List = []
            new_weights: List[float] = []
            for c, w in zip(contexts, weights):
                sub_contexts, sub_weights = ls.get_loss_scale(c, **kwargs)
                for sc, sw in zip(sub_contexts, sub_weights):
                    new_contexts.append(sc)
                    new_weights.append(w * sw)
            contexts = new_contexts
            weights = new_weights
        return contexts, weights

    @property
    def is_binary_loss_scale(self):
        return all(ls.is_binary_loss_scale for ls in self.loss_scales)
