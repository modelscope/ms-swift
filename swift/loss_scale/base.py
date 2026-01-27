# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from typing import List, Literal, Optional, Tuple

import json

from swift.template import ContextType, Messages, get_last_user_round
from .utils import calculate_loss_scale

ALL_BASE_STRATEGY = ['default', 'last_round', 'all']


class LossScale:
    """Base class for loss scaling in training.

        This class provides a flexible framework for controlling loss computation weights
        across different parts of the context (e.g., system prompts, user queries, assistant
        responses) during model training. Different strategies can be applied to selectively
        train on specific portions.

        Attributes:
            is_binary (bool, optional): Indicates whether loss_scale contains only 0 and 1.
                If True, loss_scale will be replaced by labels to stay compatible with
                acceleration techniques such as liger_kernel.
                If False, an additional 'loss_scale' key will be stored and the
                corresponding loss function will be used.
            base_strategy (str): Base strategy for loss computation. One of 'default',
                'last_round', or 'all'.
                - 'default': Only compute loss on assistant responses
                - 'last_round': Only compute loss on the last round's assistant response
                - 'all': Compute loss on all parts
    """
    is_binary = True

    def __init__(self, base_strategy: Literal['default', 'last_round', 'all'] = 'default'):
        """Initialize the loss scale object.

        Args:
            base_strategy: Base strategy for loss computation. One of 'default',
                'last_round', or 'all'. Defaults to 'default'.

        Raises:
            ValueError: If the provided base_strategy is not in the allowed list.
        """
        if base_strategy not in ALL_BASE_STRATEGY:
            raise ValueError(f'ALL_BASE_STRATEGY: {ALL_BASE_STRATEGY}, base_strategy: {base_strategy}')
        self.base_strategy = base_strategy

    def get_loss_scale(self, context: str, **kwargs) -> Tuple[List[str], List[float]]:
        """Calculate loss scale for the given context.

        This is a base implementation that subclasses can override to implement
        custom loss scaling logic.

        Args:
            context: The input context (string).
            **kwargs: Additional keyword arguments, such as query (the query of the
                current round).

        Returns:
            A tuple containing:
                - List[str]: List of contexts, potentially split into multiple parts
                - List[float]: Corresponding loss scale values, one-to-one with contexts
        """
        return [context], [1.]

    def __call__(self, context_list: List[str], context_types: List[ContextType], messages: Messages,
                 **kwargs) -> Tuple[List[str], List[float]]:
        """Process the complete conversation context and return contexts with loss scales.

            This method iterates through all context segments and determines the loss scale
            for each based on the context type and base strategy. It handles special cases
            such as explicitly specified loss values in messages and pre-computed loss scales.

            Args:
                context_list: List of context strings or dicts, each representing a segment
                    of the conversation.
                context_types: List of context types corresponding to each context, indicating
                    whether it's a system prompt, user query, assistant response, etc.
                messages: Complete message list containing the conversation history.

            Returns:
                A tuple containing:
                    - List[str]: Processed context list, potentially expanded if contexts
                        are split into multiple parts
                    - List[float]: Loss scale values corresponding one-to-one with the
                        returned context list
        """
        res_context_list = []
        res_loss_scale = []
        i = 0
        last_user_round = get_last_user_round(messages)
        for context, context_type in zip(context_list, context_types):
            is_last_round = 2 * i >= last_user_round
            query, loss = None, None
            if context_type == ContextType.RESPONSE:
                query = messages[2 * i]['content']
                # Currently, we only support applying loss/mask to the response part.
                loss = messages[2 * i + 1].get('loss')
                assert context == messages[2 * i + 1]['content']
                i += 1
            if isinstance(context, dict) and 'loss_scale' in context:
                new_context = [[token] for token in context['token_ids']]
                loss_scale = context['loss_scale']
            else:
                if isinstance(context, dict) and 'token_ids' in context:
                    context = context['token_ids']
                is_assistant = context_type in {ContextType.RESPONSE, ContextType.SUFFIX}
                if loss or loss is None and (self.base_strategy == 'all' or
                                             (self.base_strategy == 'default' and is_assistant) or
                                             (self.base_strategy == 'last_round' and is_assistant and is_last_round)):
                    new_context, loss_scale = self.get_loss_scale(context, query=query)
                else:
                    new_context, loss_scale = [context], [0.]
            res_context_list += new_context
            res_loss_scale += loss_scale
        # The values in loss_scale_list correspond one-to-one with the values in context_list.
        return res_context_list, res_loss_scale

    @property
    def is_loss_scale_binary(self):
        """Check if loss scale values are binary (only 0 and 1)."""
        return self.is_binary


class ConfigLossScale(LossScale):
    """Loss scale class that loads configuration from a JSON file.

    This class extends LossScale to support loading predefined loss scale mappings
    from a configuration file. The mappings can specify different loss weights for
    different tokens or segments based on the content.

    Attributes:
        loss_scale_config (str, optional): Path to the loss scale configuration file
            relative to the 'config' directory.
        loss_scale_map (dict, optional): Dictionary mapping loaded from the config file,
            containing predefined loss scale values for specific patterns or tokens.
    """
    is_binary = None
    loss_scale_config = None  # path

    def __init__(self, base_strategy: Literal['default', 'last_round', 'all'] = 'default'):
        """Initialize the config-based loss scale object.

        Loads the loss scale configuration from a JSON file if loss_scale_config
        is specified.

        Args:
            base_strategy: Base strategy for loss computation. One of 'default',
                'last_round', or 'all'. Defaults to 'default'.
        """
        super().__init__(base_strategy)
        self.loss_scale_map = None
        if self.loss_scale_config is not None:
            path = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(path, 'config', self.loss_scale_config)
            with open(config_path, 'r', encoding='utf-8') as json_file:
                self.loss_scale_map = json.load(json_file)

    @property
    def is_loss_scale_binary(self):
        if self.is_binary is not None:
            return self.is_binary
        if self.loss_scale_map is None:
            return True
        return all(scale in {0.0, 1.0} for lst in self.loss_scale_map.values() for scale in lst)

    def get_loss_scale(self, context: str, *, query: Optional[str] = None):
        """Calculate loss scale using the loaded configuration.

        If context is a string, uses the configuration map to calculate loss scales
        based on the query and context. Otherwise, falls back to the parent class
        implementation.

        Args:
            context: The input context string.
            query: The user query for the current round, used to determine
                appropriate loss scaling based on the configuration.

        Returns:
            Tuple[List[str], List[float]]: List of context segments and their
                corresponding loss scale values.
        """
        if isinstance(context, str):
            return calculate_loss_scale(query, context, self.loss_scale_map)
        return super().get_loss_scale(context)
