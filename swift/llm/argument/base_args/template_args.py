# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from swift.llm import TEMPLATE_MAPPING
from swift.utils import get_logger

logger = get_logger()


@dataclass
class TemplateArguments:
    """TemplateArguments class holds various arguments for template configuration.

    This dataclass manages settings related to how data is formatted and processed using templates, including
    tokenization, truncation, loss calculation, and special handling for multimodal and agent-based models.

    Args:
        template (Optional[str]): The dialogue template type. Defaults to None, which automatically selects the
            template corresponding to the model type. Refer to the list of supported models for mappings.
        system (Optional[str]): Custom system prompt. Can be a string or a path to a .txt file. Defaults to None,
            which uses the default system from the registered template.
            Note: The priority for the system prompt is as follows:
            1. System prompt from the dataset.
            2. The `--system` command-line argument.
            3. The `default_system` set when the template was registered.
        max_length (Optional[int]): The maximum number of tokens for a single sample after tokenization. Samples
            exceeding this length are handled according to `truncation_strategy` to prevent OOM errors. Defaults to
            None, which uses the model's maximum supported length (`max_model_len`). In PPO, GRPO, and inference
            scenarios, this argument specifies the `max_prompt_length`.
        truncation_strategy (Literal['delete', 'left', 'right', 'split']): Strategy for handling samples exceeding
            `max_length`. Options are 'delete', 'left' (truncate from the left), 'right' (truncate from the right),
            and 'split' (split into multiple samples). Defaults to 'delete'.
            Note: The 'split' strategy is only supported during pre-training (e.g., `swift/megatron pt`), requires
            `ms-swift>=3.11`, and is incompatible with `cached_dataset`. It splits long samples to avoid wasting
            tokens.
            Note: For multimodal models, setting this to 'left' or 'right' preserves all image tokens, which may lead
            to OOM errors.
        max_pixels (Optional[int]): The maximum number of pixels (H*W) for an input image in a multimodal model.
            Images exceeding this limit will be scaled down to prevent OOM errors. Defaults to None, meaning no limit.
            Note: This parameter applies to all multimodal models. The model-specific `MAX_PIXELS` parameter for
            Qwen2.5-VL is separate and only applies to that model.
        agent_template (Optional[str]): The Agent template to use. This determines how the 'tools' list is converted
            into a 'system' prompt, how tool calls are extracted from the model's response during inference, and the
            format for tool call messages. Options include "react_en", "hermes", "glm4", "qwen_en", "toolbench", etc.
            Defaults to None, which auto-selects based on the model type. Refer to the Agent documentation for more
            details.
        norm_bbox (Optional[Literal['norm1000', 'none']]): Controls how bounding box coordinates (from the "bbox"
            field in the dataset) are scaled. 'norm1000' scales coordinates to a 1000x1000 grid, while 'none' performs
            no scaling. Defaults to None, which auto-selects based on the model. This handles cases where images are
            resized during training (e.g., due to `max_pixels`).
        use_chat_template (bool): Whether to use the chat template or the generation template. The generation template
            is typically used for pre-training. Defaults to True.
            Note: Defaults to False for `swift pt`, which uses the generation template. This parameter is compatible
            with multimodal models.
        padding_free (bool): If True, flattens the data within a batch to avoid padding, reducing memory usage and
            speeding up training. Sequences within the batch remain causally isolated. Defaults to False. Supported for
            CPT/SFT/DPO/GRPO/KTO/GKD.
            Note: This requires `--attn_impl flash_attn` and `transformers>=4.44`. Compared to packing, padding_free
            has no preprocessing overhead, but packing offers faster training speeds and more stable memory usage.
        padding_side (Literal['left', 'right', None]): The side to pad on when `batch_size >= 2` during training.
            Options are 'left' or 'right'. Defaults to 'right'. For inference with `batch_size >= 2`, padding is always
            on the left.
            Note: Defaults to 'left' for PPO and GKD.
        loss_scale (str): Specifies the weight for loss calculation on tokens. Defaults to 'default'.
            - 'default': Calculates loss on all assistant responses (including history), but excludes system prompts,
            user inputs, multimodal tokens, and `tool_response` content.
            - 'last_round': Only calculates loss on the final round's response. (Default for RLHF).
            - 'all': Calculates loss on all tokens. (Default for `swift pt`).
            - 'ignore_empty_think': Same as 'default', but ignores loss for empty `<think>` blocks.
            - 'last_round_with_ignore_empty_think': Same as 'last_round', but ignores empty `<think>` blocks.
            - Agent-specific scales (e.g., 'react', 'hermes', 'qwen'): Based on 'default', but with a loss weight of 2
            for `tool_call` parts.
            - For more options, see
            [`loss_scale.py`](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/loss_scale/loss_scale.py).
        sequence_parallel_size (int): The size of sequence parallelism. Defaults to 1. Currently supported for CPT,
            SFT, DPO, and GRPO.
        response_prefix (Optional[str]): A prefix string for the response, e.g., '<think>\\n' for Qwen-32B. This
            parameter only affects inference. Defaults to None, which is auto-set based on the model.
        template_backend (Literal['swift', 'jinja']): The backend to use for templating. Options are 'swift' or
            'jinja'. Defaults to 'swift'. If 'jinja' is used, it will leverage `transformers.apply_chat_template`.
            Note: The 'jinja' backend is only supported for inference, not for training, as it cannot determine the
            token range for loss calculation.
    """
    template: Optional[str] = field(
        default=None, metadata={'help': f'template choices: {list(TEMPLATE_MAPPING.keys())}'})
    system: Optional[str] = None  # Override the default_system in the template.
    max_length: Optional[int] = None

    truncation_strategy: Literal['delete', 'left', 'right', 'split', None] = None
    max_pixels: Optional[int] = None
    agent_template: Optional[str] = None
    norm_bbox: Literal['norm1000', 'none', None] = None
    use_chat_template: Optional[bool] = None
    # train
    padding_free: bool = False
    padding_side: Literal['left', 'right', None] = None
    loss_scale: str = 'default'
    sequence_parallel_size: int = 1
    # infer/deploy
    response_prefix: Optional[str] = None
    template_backend: Literal['swift', 'jinja'] = 'swift'

    def __post_init__(self):
        if self.template is None and getattr(self, 'model_meta', None):
            self.template = self.model_meta.template
        if self.use_chat_template is None:
            self.use_chat_template = True
        if self.system is not None:
            if self.system.endswith('.txt'):
                assert os.path.isfile(self.system), f'self.system: {self.system}'
                with open(self.system, 'r') as f:
                    self.system = f.read()
            else:
                self.system = self.system.replace('\\n', '\n')
        if self.response_prefix is not None:
            self.response_prefix = self.response_prefix.replace('\\n', '\n')
        if self.truncation_strategy is None:
            self.truncation_strategy = 'delete'
        if self.padding_side is None:
            if getattr(self, 'task_type', None) in ('reranker', 'generative_reranker'):
                self.padding_side = 'left'
                logger.info(f'Setting args.padding_side to {self.padding_side} for task_type={self.task_type}')
            else:
                self.padding_side = 'right'

    def get_template_kwargs(self):
        from ..train_args import TrainArguments
        truncation_strategy = self.truncation_strategy
        if truncation_strategy == 'delete':
            truncation_strategy = 'raise'
        remove_unused_columns = self.remove_unused_columns  # from DataArguments
        if not isinstance(self, TrainArguments) or hasattr(self, 'rlhf_type') and self.rlhf_type == 'grpo':
            remove_unused_columns = True
        return {
            'default_system': self.system,
            'max_length': self.max_length,
            'truncation_strategy': truncation_strategy,
            'max_pixels': self.max_pixels,
            'agent_template': self.agent_template,
            'norm_bbox': self.norm_bbox,
            'use_chat_template': self.use_chat_template,
            'remove_unused_columns': remove_unused_columns,
            # train
            'padding_free': self.padding_free,
            'padding_side': self.padding_side,
            'loss_scale': self.loss_scale,
            'sequence_parallel_size': self.sequence_parallel_size,
            # infer/deploy
            'response_prefix': self.response_prefix,
            'template_backend': self.template_backend,
        }
