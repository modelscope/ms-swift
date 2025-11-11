# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swift.llm.utils import Messages


class TeacherAdapter(ABC):
    """Base class for transforming student context to teacher context in GKD training."""

    @abstractmethod
    def shape_context(self, history: 'Messages') -> 'Messages':
        """Transform student messages to teacher messages.

        Args:
            history: Student model's messages (standard OpenAI messages format)

        Returns:
            Teacher model's messages
        """
        pass


class DefaultTeacherAdapter(TeacherAdapter):
    """Default: teacher uses the same context as student."""

    def shape_context(self, history: 'Messages') -> 'Messages':
        return history


class MathTeacherAdapter(TeacherAdapter):
    """Example: add extra instructions to system prompt for teacher."""

    def shape_context(self, history: 'Messages') -> 'Messages':
        # Create a copy to avoid modifying original
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
