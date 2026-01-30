# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from swift.template import TEMPLATE_MAPPING, get_template_meta
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
        padding_side (Literal['left', 'right']): The side to pad on when `batch_size >= 2` during training.
            Options are 'left' or 'right'. Defaults to 'right'. For inference with `batch_size >= 2`, padding is always
            on the left.
            Note: Defaults to 'left' for PPO and GKD.
        padding_free (bool): If True, flattens the data within a batch to avoid padding, reducing memory usage and
            speeding up training. Sequences within the batch remain causally isolated. Defaults to False. Supported for
            CPT/SFT/DPO/GRPO/KTO/GKD.
            Note: This requires `--attn_impl flash_attn` and `transformers>=4.44`. Compared to packing, padding_free
            has no preprocessing overhead, but packing offers faster training speeds and more stable memory usage.
        loss_scale (str): Loss weight configuration for training tokens. Default is `'default'`.
            loss_scale includes 3 basic strategies: 'default', 'last_round', 'all', and other strategies:
            'ignore_empty_think' and agent-specific ones: 'react', 'hermes', 'qwen', 'agentflan', 'alpha_umi', etc.
            For available options, refer to
            [loss_scale module](https://github.com/modelscope/ms-swift/blob/main/swift/loss_scale/mapping.py).
            ms-swift>=3.12 supports mixing basic strategies with other strategies,
            for example: `'default+ignore_empty_think'`, `'last_round+ignore_empty_think'`.
            If no basic strategy is specified, it defaults to 'default',
            for example: 'hermes' is equivalent to 'default+hermes'.
            - 'default': All responses (including history) are calculated with weight 1 for cross-entropy loss
            (**system/user/multimodal tokens in messages and `tool_response` parts in Agent training are
            not included in loss calculation**). (**Default value for SFT**)
            - 'last_round': Only calculate loss for the last round response. In "ms-swift>=3.12", the last round
            means all content after the last "user", previously it only included the last "assistant".
            (**Default value for RLHF**)
            - 'all': Calculate loss for all tokens. (**Default value for `swift pt`**)
            - 'ignore_empty_think': Ignore loss computation for empty `'<think>\n\n</think>\n\n'`
            (as long as it matches the regex `'<think>\\s*</think>\\s*'`).
            - 'react', 'hermes', 'qwen': Adjust the loss weight of the `tool_call` part to 2.
        sequence_parallel_size (int): The size of sequence parallelism. Defaults to 1. Currently supported for CPT,
            SFT, DPO, and GRPO.
        template_backend (Literal['swift', 'jinja']): The backend to use for templating. Options are 'swift' or
            'jinja'. Defaults to 'swift'. If 'jinja' is used, it will leverage `transformers.apply_chat_template`.
            Note: The 'jinja' backend is only supported for inference, not for training, as it cannot determine the
            token range for loss calculation.
        response_prefix (Optional[str]): A prefix string for the response, e.g., '<think>\\n' for Qwen-32B. This
            parameter only affects inference. Defaults to None, which is auto-set based on the model.
        enable_thinking (Optional[bool]): (ms-swift>=3.12) This parameter takes effect during inference,
            indicating whether to enable thinking mode. Default is None, the default value is determined by the
            template (model) type (True for thinking/hybrid thinking templates, False for non-thinking templates).
            If enable_thinking is False, a non-thinking prefix is added, for example the Qwen3-8B hybrid thinking
            model adds the prefix `'<think>\n\n</think>\n\n'`, while Qwen3-8B-Thinking does not add a prefix.
            If enable_thinking is True, a thinking prefix is added, for example `'<think>\n'`.
            Note: The priority of this parameter is lower than the response_prefix parameter.
            - Note: For thinking models (thinking/hybrid thinking) or when enable_thinking is explicitly enabled,
            we will remove historical thinking content during both inference and training (the thinking content
            of the last round is retained, i.e., the content after the last user message).
            If the basic strategy of loss_scale during training is not last_round, for example 'default',
            then historical thinking content will not be removed.
        add_non_thinking_prefix (bool): This parameter only takes effect during training, indicating whether to
            add a non-thinking prefix to data samples whose assistant part does not start with the thinking
            marker `'<think>'` (typically hybrid thinking models contain a non-thinking prefix).
            This feature allows swift's built-in datasets to train hybrid thinking models. Default value is True.
            For example: the non-thinking prefix for the Qwen3-8B hybrid thinking model is
            `'<think>\n\n</think>\n\n'`, while the non-thinking prefix for Qwen3-8B-Thinking/Instruct is `''`.
            Note: During training, if the basic strategy of loss_scale is last_round, this modification is only
            applied to the last round; otherwise, for example 'default' or 'all', this modification is applied to
            every round of data. If set to False, no non-thinking prefix is added to data samples.


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
    padding_side: Literal['left', 'right'] = 'right'
    # train
    padding_free: bool = False
    loss_scale: str = 'default'
    sequence_parallel_size: int = 1
    # infer/deploy
    template_backend: Literal['swift', 'jinja'] = 'swift'
    # thinking
    response_prefix: Optional[str] = None
    enable_thinking: Optional[bool] = None
    add_non_thinking_prefix: bool = True

    def __post_init__(self):
        if getattr(self, 'model_meta', None) is not None:
            self.template_meta = get_template_meta(self.model_info, self.model_meta, template_type=self.template)
            self.template = self.template_meta.template_type
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

    def get_template_kwargs(self):
        from ..sft_args import SftArguments
        truncation_strategy = self.truncation_strategy
        if truncation_strategy == 'delete':
            truncation_strategy = 'raise'
        remove_unused_columns = self.remove_unused_columns  # from DataArguments
        if not isinstance(self, SftArguments) or hasattr(self, 'rlhf_type') and self.rlhf_type == 'grpo':
            remove_unused_columns = True
        return {
            'template_type': self.template,
            'default_system': self.system,
            'max_length': self.max_length,
            'truncation_strategy': truncation_strategy,
            'max_pixels': self.max_pixels,
            'agent_template': self.agent_template,
            'norm_bbox': self.norm_bbox,
            'use_chat_template': self.use_chat_template,
            'remove_unused_columns': remove_unused_columns,
            'padding_side': self.padding_side,
            # train
            'padding_free': self.padding_free,
            'loss_scale': self.loss_scale,
            'sequence_parallel_size': self.sequence_parallel_size,
            # infer/deploy
            'template_backend': self.template_backend,
            # thinking
            'response_prefix': self.response_prefix,
            'enable_thinking': self.enable_thinking,
            'add_non_thinking_prefix': self.add_non_thinking_prefix,
        }
