# Copyright (c) ModelScope Contributors. All rights reserved.

import os
from typing import TYPE_CHECKING, Dict, Literal, Optional

from swift.utils import Processor
from .base import Template
from .template_meta import TemplateMeta

if TYPE_CHECKING:
    from swift.model import ModelInfo, ModelMeta

TEMPLATE_MAPPING: Dict[str, TemplateMeta] = {}


def register_template(template_meta: TemplateMeta, *, exist_ok: bool = False) -> None:
    template_type = template_meta.template_type
    if not exist_ok and template_type in TEMPLATE_MAPPING:
        raise ValueError(f'The `{template_type}` has already been registered in the TEMPLATE_MAPPING.')
    TEMPLATE_MAPPING[template_type] = template_meta


def _read_args_json_template_type(model_dir):
    if not os.path.exists(os.path.join(model_dir, 'args.json')):
        return
    from swift.arguments import BaseArguments
    args = BaseArguments.from_pretrained(model_dir)
    return args.template


def get_template_meta(model_info: 'ModelInfo',
                      model_meta: 'ModelMeta',
                      template_type: Optional[str] = None) -> TemplateMeta:
    if template_type is None and model_info is not None:
        template_type = _read_args_json_template_type(model_info.model_dir)
    template_type = template_type or model_meta.template
    if template_type is None:
        candidates = model_meta.candidate_templates
        if len(candidates) > 1 or len(candidates) == 0:
            candidates_str = ''
            if len(candidates) > 1:
                candidates_str = f'Multiple possible types found: {candidates}. '
            raise ValueError(
                f'Failed to automatically match `template_type` for `{model_info.model_dir}`. {candidates_str}'
                'Please specify `template_type` manually. See documentation: '
                'https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html')
        elif len(candidates) == 1:
            template_type = candidates[0]
    elif template_type not in TEMPLATE_MAPPING:
        raise ValueError(f"template_type: '{template_type}' not in {list(TEMPLATE_MAPPING.keys())}")
    template_meta = TEMPLATE_MAPPING[template_type]
    return template_meta


def get_template(
    processor: Processor,
    default_system: Optional[str] = None,
    max_length: Optional[int] = None,
    *,
    template_type: Optional[str] = None,
    truncation_strategy: Literal['raise', 'left', 'right', 'split'] = 'raise',
    max_pixels: Optional[int] = None,  # h * w
    agent_template: Optional[str] = None,
    norm_bbox: Literal['norm1000', 'none', None] = None,
    use_chat_template: bool = True,
    remove_unused_columns: bool = True,
    padding_side: Literal['left', 'right'] = 'right',
    # train
    padding_free: bool = False,
    loss_scale: str = 'default',
    sequence_parallel_size: int = 1,
    # infer/deploy
    template_backend: Literal['swift', 'jinja'] = 'swift',
    # thinking
    response_prefix: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
    add_non_thinking_prefix: bool = True,
) -> 'Template':
    """Get or create a template instance for model input/output formatting.

    This function retrieves the appropriate template class based on the model type and initializes
    it with the specified configuration. It handles automatic template type detection from model
    metadata, validates configuration, and supports various modes including training, inference,
    RLHF, and agent-based interactions.

    The template system provides a unified interface for:
    - Converting conversations to token sequences and back
    - Handling multimodal inputs (images, videos, audio, bounding boxes)
    - Managing different chat formats and special tokens
    - Supporting various training strategies (standard, RLHF, KTO, embedding, etc.)
    - Integrating with multiple inference engines (Transformers, vLLM, LMDeploy, SGLang)

    Args:
        processor (Processor): Processor object containing model information, metadata,
            tokenizer, and preprocessing capabilities. Required for template initialization.
        default_system (Optional[str], optional): Default system prompt to prepend to conversations.
            If None, uses the template's default system prompt. Can be used to override the
            model's built-in system message. Defaults to None.
        max_length (Optional[int], optional): Maximum sequence length for tokenized inputs.
            Sequences exceeding this length are handled according to truncation_strategy.
            If None, set to the maximum length supported by the model. Defaults to None.
        template_type (Optional[str], optional): Explicit template type identifier
            (e.g., 'chatml', 'qwen', 'llama3'). If None, automatically detected from model
            metadata or args.json in the model directory. Defaults to None.
            Template auto-detection priority: explicit template_type > args.json > model metadata
        truncation_strategy (Literal['raise', 'left', 'right', 'split'], optional):
            Strategy for handling sequences that exceed max_length:
            - 'raise': Raise MaxLengthError
            - 'left': Truncate from the left, preserving recent context
            - 'right': Truncate from the right, preserving initial context
            - 'split': Split into multiple sequences of max_length
            Defaults to 'raise'.
        max_pixels (Optional[int], optional): Maximum number of pixels (height × width) for
            image inputs in vision-language models. Images exceeding this limit are rescaled
            proportionally. None means no limit. Defaults to None.
        agent_template (Optional[str], optional): Template type for agent-based interactions
            such as ReAct, function calling, or tool use. Examples: 'react', 'hermes'.
            If None, uses the model's default agent template if available. Defaults to None.
        norm_bbox (Literal['norm1000', 'none', None], optional): Bounding box normalization
            strategy for grounding and detection tasks:
            - 'norm1000': Normalize coordinates to [0, 1000] range
            - 'none': Keep original pixel coordinates
            - None: Use the default normalization of the corresponding model's template
            Defaults to None.
        use_chat_template (bool, optional): Whether to use the model's native chat template
            format. If False, uses a simpler generation-only template without chat structure.
            Defaults to True.
        remove_unused_columns (bool, optional): Whether to remove dataset columns not used
            by the model during data processing. Helps reduce memory usage. Defaults to True.
        padding_side (Literal['left', 'right'], optional): Side to add padding tokens:
            - 'left': Pad on the left (useful for batched inference)
            - 'right': Pad on the right (standard for training)
            Defaults to 'right'.
        padding_free (bool, optional): Enable padding-free (packing) training where multiple
            sequences are concatenated without padding tokens. Improves training efficiency.
            Defaults to False.
        loss_scale (str, optional): Loss scaling strategy identifier for different parts
            of sequences. Controls the contribution value of tokens to the loss.
            Defaults to 'default'.
        sequence_parallel_size (int, optional): Number of devices for sequence parallelism
            in distributed training. Splits long sequences across devices.
            Defaults to 1 (no parallelism).
        template_backend (Literal['swift', 'jinja'], optional): Template rendering engine:
            - 'swift': Swift's native template engine with advanced features
            - 'jinja': Jinja2 template engine
            Defaults to 'swift'.
        response_prefix (Optional[str], optional): Prefix string to add before model responses.
            Useful for structured output, thinking tokens, or format indicators. If None,
            uses template's default prefix based on thinking mode. Defaults to None.
        enable_thinking (Optional[bool], optional): Controls whether thinking mode is enabled
            during inference.
        add_non_thinking_prefix (bool, optional): This parameter only takes effect during
            training and indicates whether to add a non-thinking prefix to data samples
            whose assistant part does not start with the thinking tag '<think>'
            (typically used in hybrid thinking models that contain non-thinking prefixes).

    Returns:
        Template: Initialized template instance configured with the specified parameters.
            The template is ready to encode conversations, handle multimodal inputs, and
            integrate with training or inference pipelines.

    Raises:
        ValueError: If template_type cannot be automatically determined and multiple or no
            candidate templates are found. The error message will list candidates if multiple
            are available and provide a link to supported models documentation.
        KeyError: If the specified or detected template_type is not found in TEMPLATE_MAPPING.

    Examples:
        >>> from swift import get_processor, get_template
        >>>
        >>> # Basic usage with auto-detection
        >>> processor = get_processor('Qwen/Qwen2.5-VL-7B-Instruct')
        >>> template = get_template(processor)
        >>>
        >>> # Specify template type explicitly
        >>> tokenizer = get_processor('Qwen/Qwen2.5-7B-Instruct-123')
        >>> template = get_template(tokenizer, template_type='qwen2_5')
    """
    model_info = processor.model_info
    model_meta = processor.model_meta
    template_meta = get_template_meta(model_info, model_meta, template_type=template_type)
    template_cls = template_meta.template_cls
    return template_cls(
        processor,
        template_meta,
        default_system,
        max_length,
        truncation_strategy=truncation_strategy,
        max_pixels=max_pixels,
        agent_template=agent_template,
        norm_bbox=norm_bbox,
        use_chat_template=use_chat_template,
        remove_unused_columns=remove_unused_columns,
        padding_side=padding_side,
        # train
        padding_free=padding_free,
        loss_scale=loss_scale,
        sequence_parallel_size=sequence_parallel_size,
        # infer/deploy
        template_backend=template_backend,
        # thinking
        response_prefix=response_prefix,
        enable_thinking=enable_thinking,
        add_non_thinking_prefix=add_non_thinking_prefix,
    )
