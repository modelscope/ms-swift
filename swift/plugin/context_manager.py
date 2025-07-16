# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swift.llm.utils import Messages


class ContextManager(ABC):
    """Base context manager interface for managing conversation history."""

    def __init__(self, ctx_config):
        self.ctx_config = ctx_config

    @abstractmethod
    def manage_context(self, history: 'Messages', trajectory_id: str) -> 'Messages':
        """Manage conversation context and history.

        Args:
            history: Current conversation history
            trajectory_id: Current agent trajectory_id
        Returns:
            Modified conversation history with context management applied
        """
        pass


class DummyContextManager(ContextManager):

    def __init__(self, ctx_config):
        super().__init__(ctx_config)

    def manage_context(self, history: 'Messages', trajectory_id: str) -> 'Messages':
        return history


# Registry for context managers
context_managers = {'dummyContextManager': DummyContextManager}
