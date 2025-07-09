# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod

from swift.llm.template import RolloutInferRequest


class ContextManager(ABC):
    """Base context manager interface for managing conversation history."""
    
    @abstractmethod
    def manage_context(self, history: RolloutInferRequest,trajectory_id:str) -> RolloutInferRequest:
        """Manage conversation context and history.
        
        Args:
            history: Current conversation history
            
        Returns:
            Modified conversation history with context management applied
        """
        pass


class DummyContextManager(ContextManager):
    
    def manage_context(self, history: RolloutInferRequest,trajectory_id:str) -> RolloutInferRequest:
        return history

# Registry for context managers
context_managers = {
    "dummyContextManager":DummyContextManager
}