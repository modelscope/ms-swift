# Copyright (c) Alibaba, Inc. and its affiliates.
from .argument import (AppUIArguments, DeployArguments, EvalArguments, ExportArguments, InferArguments, RLHFArguments,
                       RomeArguments, SftArguments, is_adapter, swift_to_peft_format)
from .client_utils import (convert_to_base64, decode_base64, get_model_list_client, inference_client,
                           inference_client_async)
from .dataset import (DATASET_MAPPING, DatasetName, HfDataset, get_dataset, get_dataset_from_repo,
                      load_dataset_from_local, load_ms_dataset, register_dataset, register_dataset_info,
                      register_local_dataset, sample_dataset)
from .media import MediaCache, MediaTag
from .model import (MODEL_MAPPING, GetModelTokenizerFunction, LoRATM, ModelType, get_additional_saved_files,
                    get_default_lora_target_modules, get_default_template_type, get_model_tokenizer,
                    get_model_tokenizer_from_repo, get_model_tokenizer_with_flash_attn, register_model)
from .preprocess import (AlpacaPreprocessor, ClsPreprocessor, ComposePreprocessor, ConversationsPreprocessor,
                         PreprocessFunc, RenameColumnsPreprocessor, SmartPreprocessor, SwiftPreprocessor,
                         TextGenerationPreprocessor, preprocess_sharegpt)
from .protocol import (ChatCompletionMessageToolCall, ChatCompletionRequest, ChatCompletionResponse,
                       ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse,
                       ChatMessage, CompletionRequest, CompletionResponse, CompletionResponseChoice,
                       CompletionResponseStreamChoice, CompletionStreamResponse, DeltaMessage, Function, Model,
                       ModelList, UsageInfo, XRequestConfig, random_uuid)
from .template import (DEFAULT_SYSTEM, TEMPLATE_MAPPING, History, Prompt, StopWords, Template, TemplateType,
                       get_template, register_template)
from .utils import (LazyLLMDataset, LLMDataset, dataset_map, download_dataset, find_all_linears, find_embedding,
                    find_ln, get_max_model_len, get_time_info, history_to_messages, inference, inference_stream,
                    is_quant_model, is_vllm_available, limit_history_length, messages_join_observation,
                    messages_to_history, print_example, safe_tokenizer_decode, set_generation_config,
                    sort_by_max_length, stat_dataset, to_device)

try:
    if is_vllm_available():
        from .vllm_utils import (VllmGenerationConfig, get_vllm_engine, inference_stream_vllm, inference_vllm,
                                 prepare_vllm_engine_template, vllm_context)
        try:
            from .vllm_utils import LoRARequest
        except ImportError:
            pass
except Exception as e:
    from swift.utils import get_logger
    logger = get_logger()
    logger.warning(f'import vllm_utils error: {e}')
