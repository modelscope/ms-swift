# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.utils import get_logger
from .argument import (AppUIArguments, DeployArguments, EvalArguments, ExportArguments, InferArguments, PtArguments,
                       RLHFArguments, RomeArguments, SftArguments, WebuiArguments, is_adapter, swift_to_peft_format)
from .client_utils import (compat_openai, convert_to_base64, decode_base64, get_model_list_client,
                           get_model_list_client_async, inference_client, inference_client_async)
from .dataset import (DATASET_MAPPING, DatasetName, HfDataset, get_dataset, get_dataset_from_repo,
                      load_dataset_from_local, load_ms_dataset, register_dataset, register_dataset_info,
                      register_local_dataset, sample_dataset, standard_keys)
from .media import MediaCache, MediaTag
from .model import (MODEL_MAPPING, GetModelTokenizerFunction, LoRATM, ModelType, get_additional_saved_files,
                    get_default_lora_target_modules, get_default_template_type, get_model_tokenizer,
                    get_model_tokenizer_from_repo, get_model_tokenizer_with_flash_attn, get_model_with_value_head,
                    git_clone_github, register_model)
from .preprocess import (AlpacaPreprocessor, ClsPreprocessor, ComposePreprocessor, ConversationsPreprocessor,
                         PreprocessFunc, RenameColumnsPreprocessor, SmartPreprocessor, SwiftPreprocessor,
                         TextGenerationPreprocessor, preprocess_sharegpt)
from .protocol import (ChatCompletionMessageToolCall, ChatCompletionRequest, ChatCompletionResponse,
                       ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse,
                       ChatMessage, CompletionRequest, CompletionResponse, CompletionResponseChoice,
                       CompletionResponseStreamChoice, CompletionStreamResponse, DeltaMessage, Function, Model,
                       ModelList, UsageInfo, XRequestConfig, random_uuid)
from .template import (DEFAULT_SYSTEM, TEMPLATE_MAPPING, History, KTOTemplateMixin, PPOTemplateMixin, Prompt,
                       RLHFTemplateMixin, StopWords, Template, TemplateType, get_env_args, get_template,
                       register_template)
from .utils import (LazyLLMDataset, LLMDataset, dataset_map, deep_getattr, download_dataset,
                    dynamic_vit_gradient_checkpointing, find_all_linears, find_embedding, find_ln, get_max_model_len,
                    get_mllm_arch, get_time_info, history_to_messages, inference, inference_stream,
                    is_lmdeploy_available, is_megatron_available, is_quant_model, is_vllm_available,
                    limit_history_length, messages_join_observation, messages_to_history, print_example,
                    safe_tokenizer_decode, set_generation_config, sort_by_max_length, stat_dataset, to_device)

logger = get_logger()

try:
    if is_vllm_available():
        from .vllm_utils import (VllmGenerationConfig, get_vllm_engine, inference_stream_vllm, inference_vllm,
                                 prepare_vllm_engine_template, add_vllm_request)
        try:
            from .vllm_utils import LoRARequest
        except ImportError:
            # Earlier vLLM version has no `LoRARequest`
            logger.info('LoRARequest cannot be imported due to a early vLLM version, '
                        'if you are using vLLM+LoRA, please install a latest version.')
            pass
    else:
        logger.info('No vLLM installed, if you are using vLLM, '
                    'you will get `ImportError: cannot import name \'get_vllm_engine\' from \'swift.llm\'`')
except Exception as e:
    logger.error(f'import vllm_utils error: {e}')

try:
    if is_lmdeploy_available():
        from .lmdeploy_utils import (
            prepare_lmdeploy_engine_template,
            LmdeployGenerationConfig,
            get_lmdeploy_engine,
            inference_stream_lmdeploy,
            inference_lmdeploy,
        )
    else:
        logger.info('No LMDeploy installed, if you are using LMDeploy, '
                    'you will get `ImportError: cannot import name '
                    '\'prepare_lmdeploy_engine_template\' from \'swift.llm\'`')
except Exception as e:
    from swift.utils import get_logger
    logger = get_logger()
    logger.error(f'import lmdeploy_utils error: {e}')
