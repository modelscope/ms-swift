# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from swift.llm.utils import Messages


class TeacherAdapter(ABC):
    """Base class for transforming student context to teacher context in GKD training."""

    @abstractmethod
    def shape_context(self, data_dict: Dict[str, Any]) -> 'Messages':
        """Transform student messages to teacher messages.

        Args:
            data_dict: Complete data dictionary containing:
                - 'messages': Student model's messages (OpenAI format)
                - Other fields like 'dataset', 'images', etc. for flexible usage

        Returns:
            Teacher model's messages
        """
        pass


class DefaultTeacherAdapter(TeacherAdapter):
    """Default: teacher uses the same context as student."""

    def shape_context(self, data_dict: Dict[str, Any]) -> 'Messages':
        return data_dict['messages']


class MathTeacherAdapter(TeacherAdapter):
    """Example: add extra instructions to system prompt for teacher."""

    def shape_context(self, data_dict: Dict[str, Any]) -> 'Messages':
        # Create a copy to avoid modifying original
        history = data_dict['messages']
        teacher_history = history.copy()

        # Example: enhance system prompt for teacher
        if teacher_history and teacher_history[0]['role'] == 'system':
            teacher_history[0] = {
                'role': 'system',
                'content': teacher_history[0]['content'] + '\n\nYou are a math expert, solve problems step by step.'
            }
        else:
            # Insert system prompt at the beginning
            teacher_history.insert(0, {
                'role': 'system',
                'content': 'You are a math expert, solve problems step by step.'
            })

        return teacher_history


# Registry for teacher adapter plugins
teacher_adapters = {
    'default': DefaultTeacherAdapter,
    'math_teacher': MathTeacherAdapter,
}
