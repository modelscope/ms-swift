# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any, Optional

if TYPE_CHECKING:
    from swift.llm.utils import Messages
    import torch


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

    def get_loss_mask(self, student_logits: 'torch.Tensor', teacher_logits: 'torch.Tensor',
                     mask: 'torch.Tensor', **kwargs) -> Optional['torch.Tensor']:
        """Optionally modify the loss mask to control which tokens participate in distillation.

        Args:
            student_logits: Student model logits, shape (batch_size, seq_len, vocab_size)
            teacher_logits: Teacher model logits, shape (batch_size, seq_len, vocab_size)
            mask: Current mask indicating response tokens, shape (batch_size, seq_len)
                  True means the position is a response token (labels != -100 after shift)
            **kwargs: Additional information like 'inputs', 'labels', etc.

        Returns:
            Modified mask with same shape as input mask, or None to use original mask.
            True means the position participates in loss computation.

        Example:
            # Only train on first 50 tokens + last 5 tokens (to ensure learning stop token)
            new_mask = torch.zeros_like(mask)
            for i in range(mask.shape[0]):
                response_indices = mask[i].nonzero(as_tuple=True)[0]
                if len(response_indices) > 0:
                    new_mask[i, response_indices[:50]] = True  # First 50 tokens
                    new_mask[i, response_indices[-6:-1]] = True  # Last 5 valid predictions
            return new_mask
        """
        return None  # Default: use original mask


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
