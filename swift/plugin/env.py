# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from swift.llm.template import RolloutInferRequest
from swift.llm.infer.protocol import RolloutResponseChoice


class Env(ABC):
    """Base environment interface for GRPO training."""
    
    def __init__(self, tools: Optional[str] = None):
        """Initialize environment with available tools.
        
        Args:
            tools: String description of available tools
        """
        self.tools = tools or ""
    
    @abstractmethod
    async def reset(self, config: RolloutInferRequest) -> Tuple[str, Dict[str, Any], str]:
        """Reset environment to initial state.
        
        Args:
            config: Initial configuration containing dataset information
            
        Returns:
            Tuple of (observation, info, system_message):
            - observation: Initial query string for the agent
            - info: Environment debug information as dict
            - system_message: System prompt for this trajectory
        """
        pass
    
    @abstractmethod
    async def step(self, action: RolloutResponseChoice) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute one step in the environment.
        
        Args:
            action: LLM response choice containing the action to execute
            
        Returns:
            Tuple of (next_observation, reward, done, info):
            - next_observation: Next observation string
            - reward: Reward value for this step
            - done: Whether the episode is finished
            - info: Additional information as dict
        """
        pass
    @abstractmethod
    async def close(self):
        """Clean up environment resources."""
        pass

# Registry for environments
envs = {

}