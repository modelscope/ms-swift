# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import math
import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import accelerate
import json
import numpy as np
import torch
import torch.distributed as dist
import transformers
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset
from datasets import concatenate_datasets
from packaging import version
from transformers.utils import is_torch_bf16_gpu_available, is_torch_cuda_available, is_torch_npu_available, strtobool
from transformers.utils.versions import require_version

from swift.hub import HubApi, ModelScopeConfig
from swift.trainers import LOSS_MAPPING, TrainerFactory
from swift.tuners import Swift
from swift.utils import (add_version_to_work_dir, get_dist_setting, get_logger, get_pai_tensorboard_dir, is_dist,
                         is_local_master, is_mp, is_pai_training_job, use_torchacc)
from .client_utils import get_model_list_client
from .dataset import (DATASET_MAPPING, _dataset_name_exists, get_dataset, parse_dataset_name,
                      register_dataset_info_file, sample_dataset)
from .media import MediaTag
from .model import (MODEL_MAPPING, dtype_mapping, get_additional_saved_files, get_default_lora_target_modules,
                    get_default_template_type)
from .template import TEMPLATE_MAPPING
from .utils import get_mllm_arch, is_liger_available, is_lmdeploy_available, is_quant_model, is_vllm_available

logger = get_logger()
DATASET_TYPE = Union[HfDataset, HfIterableDataset]


def is_adapter(sft_type: str) -> bool:
    return sft_type in {
        'lora', 'longlora', 'adalora', 'ia3', 'llamapro', 'adapter', 'vera', 'boft', 'fourierft', 'reft'
    }


class ArgumentsBase:

    def _load_json_or_path(self, key) -> None:
        value = getattr(self, key)
        if isinstance(value, str):
            if os.path.exists(value):  # local path
                with open(value, 'r') as f:
                    value = json.load(f)
            else:  # json str
                value = json.loads(value)
        setattr(self, key, value)

    def __post_init__(self) -> None:
        if self.max_length == -1:
            self.max_length = None
        if self.model_kwargs is None:
            self.model_kwargs = {}
        self._load_json_or_path('model_kwargs')
        for k, v in self.model_kwargs.items():
            k = k.upper()
            os.environ[k] = str(v)

        self._load_json_or_path('device_map_config')
        _, local_rank, _, local_world_size = get_dist_setting()
        # compat mp&ddp
        if local_world_size > 1 and isinstance(self.device_map_config, dict) and local_rank > 0:
            for k, v in self.device_map_config.items():
                if isinstance(v, int):
                    self.device_map_config[k] += local_rank

    @classmethod
    def _check_path(cls,
                    value: Union[str, List[str]],
                    k: Optional[str] = None,
                    check_exist_path_set: Optional[Set[str]] = None) -> Union[str, List[str]]:
        if check_exist_path_set is None:
            check_exist_path_set = set()
        if isinstance(value, str):
            value = os.path.expanduser(value)
            value = os.path.abspath(value)
            if k in check_exist_path_set and not os.path.exists(value):
                if k is not None:
                    raise FileNotFoundError(f"`{k}`: '{value}'")
                else:
                    raise FileNotFoundError(f"path: '{value}'")
        elif isinstance(value, list):
            res = []
            for v in value:
                res.append(cls._check_path(v, k, check_exist_path_set))
            value = res
        return value

    def _is_multimodal(self, model_type: Optional[str] = None) -> bool:
        if model_type is None:
            return False
        model_info = MODEL_MAPPING[model_type]
        tags = model_info.get('tags') or []
        return 'multi-modal' in tags

    def _is_vision(self, model_type: Optional[str] = None) -> bool:
        if model_type is None:
            return False
        model_info = MODEL_MAPPING[model_type]
        tags = model_info.get('tags') or []
        return 'vision' in tags

    def handle_path(self: Union['SftArguments', 'InferArguments']) -> None:
        check_exist_path = ['ckpt_dir', 'resume_from_checkpoint', 'custom_register_path']
        maybe_check_exist_path = ['model_id_or_path', 'custom_dataset_info']
        if isinstance(self, SftArguments):
            check_exist_path.append('deepspeed_config_path')
            maybe_check_exist_path.append('deepspeed')

        for k in maybe_check_exist_path:
            v = getattr(self, k)
            if isinstance(v, str) and v is not None and (v.startswith('~') or v.startswith('/') or os.path.exists(v)):
                check_exist_path.append(k)
        check_exist_path_set = set(check_exist_path)
        other_path = ['output_dir', 'logging_dir']
        for k in check_exist_path + other_path:
            value = getattr(self, k, None)
            if value is None:
                continue
            value = self._check_path(value, k, check_exist_path_set)
            setattr(self, k, value)

    def check_flash_attn(self: Union['SftArguments', 'InferArguments']) -> None:
        model_info = MODEL_MAPPING[self.model_type]
        support_flash_attn = model_info.get('support_flash_attn', False)
        if self.use_flash_attn and not support_flash_attn:
            logger.warning(f'use_flash_attn: {self.use_flash_attn}, ' f'but support_flash_attn: {support_flash_attn}')

    def handle_generation_config(self: Union['SftArguments', 'InferArguments']) -> None:
        if self.temperature == 0:
            self.do_sample = False
        if self.do_sample is False and (isinstance(self, InferArguments) and self.infer_backend == 'pt'
                                        or isinstance(self, SftArguments)):
            # fix warning
            self.temperature = 1.
            self.top_p = 1.
            self.top_k = 50
            logger.info('Due to do_sample=False, the following settings are applied: args.temperature: '
                        f'{self.temperature}, args.top_p: {self.top_p}, args.top_k: {self.top_k}.')

    def select_dtype(self: Union['SftArguments', 'InferArguments']) -> Tuple[Optional[torch.dtype], bool, bool]:
        if not is_torch_cuda_available() and not is_torch_npu_available():
            # cpu
            if self.dtype == 'AUTO':
                self.dtype = 'fp32'
                logger.info(f'Setting args.dtype: {self.dtype}')
            assert self.dtype != 'fp16', 'The CPU does not support matrix multiplication with FP16.'
            if self.dtype == 'fp32':
                return torch.float32, False, False
            elif self.dtype == 'bf16':
                return torch.bfloat16, False, True
            else:
                raise ValueError(f'args.dtype: {self.dtype}')
        # cuda, npu
        if self.dtype == 'AUTO':
            if not is_torch_bf16_gpu_available():
                self.dtype = 'fp16'
            else:
                model_torch_dtype = MODEL_MAPPING[self.model_type].get('torch_dtype')
                if model_torch_dtype is not None:
                    self.dtype = dtype_mapping[model_torch_dtype]
                elif isinstance(self, SftArguments):
                    self.dtype = 'bf16'
                else:
                    return None, False, False

        torch_dtype = dtype_mapping_reversed[self.dtype]

        assert torch_dtype in {torch.float16, torch.bfloat16, torch.float32}
        if torch_dtype == torch.float16:
            if isinstance(self, SftArguments) and self.sft_type == 'full':
                self.dtype = 'fp32'
                torch_dtype = torch.float32
                logger.warning(
                    'Fine-tuning with full parameters does not support fp16, and is prone to NaN. '
                    'We will use the fp32 & AMP approach, which consumes approximately twice the memory of bf16.')
                logger.info(f'Setting torch_dtype: {torch_dtype}')
            fp16, bf16 = True, False
        elif torch_dtype == torch.bfloat16:
            support_bf16 = is_torch_bf16_gpu_available()
            if not support_bf16:
                logger.warning(f'support_bf16: {support_bf16}')
            fp16, bf16 = False, True
        else:
            fp16, bf16 = False, False
        return torch_dtype, fp16, bf16

    def select_bnb(self: Union['SftArguments', 'InferArguments']) -> Tuple[Optional[torch.dtype], bool, bool]:
        if self.bnb_4bit_comp_dtype == 'AUTO':
            self.bnb_4bit_comp_dtype = self.dtype

        if self.bnb_4bit_comp_dtype != 'AUTO':
            bnb_4bit_compute_dtype = dtype_mapping_reversed[self.bnb_4bit_comp_dtype]
            assert bnb_4bit_compute_dtype in {torch.float16, torch.bfloat16, torch.float32}
        else:
            bnb_4bit_compute_dtype = None
        quantization_bit = self.quantization_bit
        if self.quant_method == 'bnb':
            if quantization_bit == 4:
                require_version('bitsandbytes')
                load_in_4bit, load_in_8bit = True, False
            elif quantization_bit == 8:
                require_version('bitsandbytes')
                load_in_4bit, load_in_8bit = False, True
            else:
                logger.warning('bnb only support 4/8 bits quantization, you should assign --quantization_bit 4 or 8,\
                    Or specify another quantization method; No quantization will be performed here.')
                load_in_4bit, load_in_8bit = False, False
        else:
            load_in_4bit, load_in_8bit = False, False

        return bnb_4bit_compute_dtype, load_in_4bit, load_in_8bit

    def handle_custom_register(self: Union['SftArguments', 'InferArguments']) -> None:
        if self.custom_register_path is None:
            return
        folder, fname = os.path.split(self.custom_register_path)
        sys.path.append(folder)
        __import__(fname.rstrip('.py'))

    def handle_compatibility(self: Union['SftArguments', 'InferArguments']) -> None:
        template_type_mapping = {'chatglm2-generation': 'chatglm-generation', 'chatml': 'qwen'}
        model_type_mapping = {
            'openbmb-minicpm-2b-sft-chat': 'minicpm-2b-sft-chat',
            'openbmb-minicpm-2b-chat': 'minicpm-2b-chat',
            'cogvlm-17b-instruct': 'cogvlm-17b-chat',
            'minicpm-v-v2': 'minicpm-v-v2-chat',
            'mplug-owl2d1-chat': 'mplug-owl2_1-chat',
            'llava1d6-mistral-7b-instruct': 'llava1_6-mistral-7b-instruct',
            'llava1d6-yi-34b-instruct': 'llava1_6-yi-34b-instruct',
        }
        dataset_name_mapping = {
            'ms-bench-mini': 'ms-bench#20000',
            'multi-alpaca-all': 'multi-alpaca',
            'instinwild-en': 'instinwild:subset',
            'instinwild-zh': 'instinwild:default',
            'firefly-all-zh': 'firefly-zh',
            'sharegpt-en': 'sharegpt:common-en/computer-en',
            'sharegpt-zh': 'sharegpt:common-zh/computer-zh/unknow-zh',
            'open-orca-gpt4': 'open-orca:default',
            'sharegpt-gpt4-mini': 'sharegpt-gpt4:default',
            'deepctrl-sft-zh': 'deepctrl-sft:default',
            'deepctrl-sft-en': 'deepctrl-sft:en',
            'ms-agent-for-agentfabric-default': 'ms-agent-for-agentfabric:default',
            'ms-agent-for-agentfabric-addition': 'ms-agent-for-agentfabric:addition',
            **{
                f'toolbench-for-alpha-umi-{sn}': f'toolbench-for-alpha-umi:{sn}'
                for sn in DATASET_MAPPING['toolbench-for-alpha-umi']['subsets']
            },
            'medical-mini-zh': 'medical-zh#50000',
            'cmnli-mini-zh': 'cmnli-zh#20000',
            'coco-mini-en': 'coco-en-mini',
            'coco-mini-en-2': 'coco-en-2-mini',
            'aishell1-mini-zh': 'aishell1-zh-mini',
            **{f'hh-rlhf-{sn}': f'hh-rlhf:{sn}'
               for sn in DATASET_MAPPING['hh-rlhf']['subsets']},
            **{
                f"hh-rlhf-cn-{sn.replace('_', '-')}": f'hh-rlhf-cn:{sn}'
                for sn in DATASET_MAPPING['hh-rlhf-cn']['subsets']
            },
            **{
                f"coig-cqia-{sn.replace('_', '-')}": f'coig-cqia:{sn}'
                for sn in DATASET_MAPPING['coig-cqia']['subsets']
            },
            **{f'ruozhiba-{sn}': f'ruozhiba:{sn}'
               for sn in DATASET_MAPPING['ruozhiba']['subsets']},
        }
        for _name, _mapping in [['template_type', template_type_mapping], ['model_type', model_type_mapping]]:
            k = getattr(self, _name)
            if k in _mapping:
                v = _mapping[k]
                setattr(self, _name, v)
                break
        for key in ['dataset', 'val_dataset']:
            _dataset = getattr(self, key)
            if isinstance(_dataset, str):
                _dataset = [_dataset]
            elif _dataset is None:
                _dataset = []
            if len(_dataset) == 1 and ',' in _dataset[0]:
                _dataset = _dataset[0].split(',')
            for i, d in enumerate(_dataset):
                if d in dataset_name_mapping:
                    _dataset[i] = dataset_name_mapping[d]
            for d in _dataset:
                assert ',' not in d, f'dataset: {d}, please use `/`'
            setattr(self, key, _dataset)
        if self.truncation_strategy == 'ignore':
            self.truncation_strategy = 'delete'
        if self.safe_serialization is not None:
            self.save_safetensors = self.safe_serialization
        if len(self.custom_train_dataset_path) > 0:
            self.dataset += self.custom_train_dataset_path
        if len(self.custom_val_dataset_path) > 0:
            self.val_dataset += self.custom_val_dataset_path
        if self.device_map_config_path is not None:
            self.device_map_config = self.device_map_config_path

        if isinstance(self, InferArguments):
            if self.merge_lora_and_save is not None:
                self.merge_lora = self.merge_lora_and_save
            if self.vllm_lora_modules is not None:
                self.lora_modules = self.vllm_lora_modules
        if isinstance(self, AppUIArguments):
            if self.server_name is not None:
                self.host = self.server_name
            if self.server_port is not None:
                self.port = self.server_port
        if isinstance(self, SftArguments):
            log_freeze_warning = False
            try:
                if isinstance(self.freeze_parameters, (int, float)):
                    log_freeze_warning = True
                elif isinstance(self.freeze_parameters, list) and len(self.freeze_parameters) == 1:
                    self.freeze_parameters = float(self.freeze_parameters[0])
                    log_freeze_warning = True
            except Exception:
                pass
            if log_freeze_warning:
                logger.warning(f'please use `--freeze_parameters_ratio {self.freeze_parameters}`')
                self.freeze_parameters_ratio = self.freeze_parameters
                self.freeze_parameters = []

            if isinstance(self.train_dataset_mix_ds, str):
                self.train_dataset_mix_ds = [self.train_dataset_mix_ds]
            if self.only_save_model is not None:
                self.save_only_model = self.only_save_model
            if self.neftune_alpha is not None:
                self.neftune_noise_alpha = self.neftune_alpha
            if self.per_device_train_batch_size is not None:
                self.batch_size = self.per_device_train_batch_size
            if self.per_device_eval_batch_size is not None:
                self.eval_batch_size = self.per_device_eval_batch_size
            if self.deepspeed_config_path is not None:
                self.deepspeed = self.deepspeed_config_path
            if self.eval_strategy is not None:
                self.evaluation_strategy = self.eval_strategy
            if self.lora_dropout_p is not None:
                self.lora_dropout = self.lora_dropout_p

            if self.boft_target_modules:
                self.target_modules = self.boft_target_modules
            if self.boft_modules_to_save:
                self.modules_to_save = self.boft_modules_to_save

            if self.ia3_target_modules:
                self.target_modules = self.ia3_target_modules
            if self.ia3_modules_to_save:
                self.modules_to_save = self.ia3_modules_to_save

            if self.vera_target_modules:
                self.target_modules = self.vera_target_modules
            if self.vera_modules_to_save:
                self.modules_to_save = self.vera_modules_to_save

            if self.lora_target_modules:
                self.target_modules = self.lora_target_modules
            if self.lora_modules_to_save:
                self.modules_to_save = self.lora_modules_to_save
            if self.lora_target_regex:
                self.target_regex = self.lora_target_regex

        if getattr(self, 'push_hub_strategy', None):
            self.hub_strategy = self.push_hub_strategy
            if self.hub_strategy in ('push_last', 'push_best'):
                self.hub_strategy = 'every_save'

    def handle_custom_dataset_info(self: Union['SftArguments', 'InferArguments']):
        if self.custom_dataset_info is None:
            return
        register_dataset_info_file(self.custom_dataset_info)

    def _handle_dataset_sample(self: Union['SftArguments', 'InferArguments']):
        # compatibility. (Deprecated)
        # Avoid post-processing
        if len(self.dataset) != 1 or self.train_dataset_sample == -1:
            return
        _dataset = self.dataset[0]
        train_sample = parse_dataset_name(_dataset)[3]
        if train_sample == -1:
            train_sample = self.train_dataset_sample
        else:
            _dataset = _dataset[:_dataset.find('#')]
            if self.train_dataset_sample < train_sample:
                train_sample = self.train_dataset_sample
        _dataset = f'{_dataset}#{train_sample}'
        self.dataset[0] = _dataset
        self.train_dataset_sample = -1

    def _register_self_cognition(self: Union['SftArguments', 'InferArguments']) -> None:

        # compatibility. (Deprecated)
        idx_list = _dataset_name_exists(self.dataset, 'self-cognition')
        assert len(idx_list) <= 1
        self.use_self_cognition = len(idx_list) == 1
        if self.self_cognition_sample > 0:
            d = f'self-cognition#{self.self_cognition_sample}'
            if len(idx_list) == 1:
                self.dataset[idx_list[0]] = d
            else:
                self.dataset.append(d)
            self.use_self_cognition = True
        # check
        if self.use_self_cognition:
            for k in ['model_name', 'model_author']:
                v = getattr(self, k)
                if isinstance(v, str):
                    v = [v]
                elif v is None:
                    v = [None, None]
                if len(v) == 1:
                    v = v * 2
                if v[0] is None and v[1] is None:
                    raise ValueError('Please set self.model_name self.model_author. '
                                     'For example: `--model_name 小黄 "Xiao Huang" --model_author 魔搭 ModelScope`. '
                                     'Representing the model name and model author in Chinese and English.')
                setattr(self, k, v)

    def _handle_dataset_compat(
            self: Union['SftArguments', 'InferArguments'], train_dataset: Optional[DATASET_TYPE],
            val_dataset: Optional[DATASET_TYPE]) -> Tuple[Optional[DATASET_TYPE], Optional[DATASET_TYPE]]:
        # compatibility. (Deprecated)
        streaming = getattr(self, 'streaming', False)
        random_state = np.random.RandomState(self.dataset_seed)
        val_dataset_sample = self.val_dataset_sample

        if train_dataset is not None and self.train_dataset_sample >= 0:
            train_dataset_sample = min(self.train_dataset_sample, train_dataset.shape[0])
            if train_dataset.shape[0] > train_dataset_sample:
                logger.info(f'train_dataset_sample: {train_dataset_sample}')
                train_idxs = random_state.permutation(train_dataset_sample)
                train_dataset = train_dataset.select(train_idxs)
            if val_dataset_sample is None:
                val_dataset_sample = max(int(train_dataset_sample * self.dataset_test_ratio), 1)
        if val_dataset is not None and val_dataset_sample is not None and val_dataset_sample >= 0:
            if not streaming and val_dataset.shape[0] > val_dataset_sample:
                logger.info(f'val_dataset_sample: {val_dataset_sample}')
                val_idxs = random_state.permutation(val_dataset_sample)
                val_dataset = val_dataset.select(val_idxs)
            elif streaming:
                val_dataset = val_dataset.shuffle(
                    seed=self.dataset_seed, buffer_size=self.streaming_buffer_size).take(val_dataset_sample)

        if (train_dataset is None or not hasattr(self, 'train_dataset_mix_ratio') or self.train_dataset_mix_ratio <= 0
                or len(self.train_dataset_mix_ds) == 0):
            return train_dataset, val_dataset

        mix_dataset_sample = int(len(train_dataset) * self.train_dataset_mix_ratio)
        logger.info(f'train_dataset_mix_ds: {self.train_dataset_mix_ds}')
        logger.info(f'len(train_dataset): {len(train_dataset)}, mix_dataset_sample: {mix_dataset_sample}')
        mixed_dataset = get_dataset(
            self.train_dataset_mix_ds,
            0.0,
            random_state,
            check_dataset_strategy=self.check_dataset_strategy,
            streaming=streaming)[0]
        if len(mixed_dataset) < mix_dataset_sample:
            logger.warn(f'The length of dataset used for mixin: {self.train_dataset_mix_ds} are '
                        'lesser than the ratio required by the `train_dataset_mix_ratio` '
                        f'argument: {self.train_dataset_mix_ratio}. '
                        f'the actual ratio is: {len(mixed_dataset) / len(train_dataset):.6}.')
        else:
            mixed_dataset = sample_dataset(mixed_dataset, mix_dataset_sample, random_state)
        train_dataset = concatenate_datasets([train_dataset, mixed_dataset])
        return train_dataset, val_dataset

    def prepare_template(self: Union['SftArguments', 'InferArguments']):
        if self.template_type == 'AUTO':
            self.template_type = get_default_template_type(self.model_type)
            logger.info(f'Setting template_type: {self.template_type}')

    def set_model_type(self: Union['SftArguments', 'InferArguments']) -> None:
        # compat with swift<1.7
        if self.model_cache_dir is not None and self.model_id_or_path is None:
            self.model_id_or_path = self.model_cache_dir
            self.model_cache_dir = None

        if self.model_id_or_path is not None:
            use_hf = strtobool(os.environ.get('USE_HF', 'False'))
            model_mapping_reversed = {}
            for k, v in MODEL_MAPPING.items():
                if use_hf:
                    model_id = v.get('hf_model_id')
                else:
                    model_id = v.get('model_id_or_path')
                if model_id is None:
                    continue
                model_id = model_id.lower()
                model_mapping_reversed[model_id] = k
            model_id_or_path = self.model_id_or_path
            model_id_or_path_lower = model_id_or_path.lower()

            if self.model_type is None and model_id_or_path_lower in model_mapping_reversed:
                model_type = model_mapping_reversed[model_id_or_path_lower]
                assert self.model_type is None or self.model_type == model_type
                self.model_type = model_type
                logger.info(f'Setting args.model_type: {model_type}')
                if self.model_cache_dir is not None:
                    self.model_id_or_path = self.model_cache_dir
            else:
                if (isinstance(self, InferArguments) and 'checkpoint-' in model_id_or_path
                        and 'merged' not in model_id_or_path and self.ckpt_dir is None):
                    raise ValueError('Please use `--ckpt_dir vx-xxx/checkpoint-xxx` to use the checkpoint.')
                if self.model_type is None:
                    raise ValueError(f"model_id_or_path: '{model_id_or_path}' is not registered. "
                                     'Please set `--model_type <model_type> --model_id_or_path <model_id_or_path>`.')
                assert self.model_cache_dir is None

        error_msg = f'The model_type you can choose: {list(MODEL_MAPPING.keys())}'
        if self.model_type is None:
            raise ValueError('please setting `--model_type <model_type>`. ' + error_msg)
        elif self.model_type not in MODEL_MAPPING:
            raise ValueError(f"model_type: '{self.model_type}' is not registered. " + error_msg)
        model_info = MODEL_MAPPING[self.model_type]
        use_hf = strtobool(os.environ.get('USE_HF', 'False'))
        if self.model_revision is not None:
            model_info['revision'] = self.model_revision
            logger.info(f"Setting model_info['revision']: {self.model_revision}")
        elif use_hf:
            model_info['revision'] = 'main'
        self.model_revision = model_info['revision']
        if self.model_id_or_path is None:
            self.model_id_or_path = model_info['hf_model_id'] if use_hf else model_info['model_id_or_path']
        requires = model_info['requires']
        for require in requires:
            require_version(require)

    def prepare_ms_hub(self: Union['SftArguments', 'InferArguments']) -> None:
        hub_token = self.hub_token
        if hub_token is None:
            hub_token = os.environ.get('MODELSCOPE_API_TOKEN')
        if hub_token:
            api = HubApi()
            api.login(hub_token)
        if not hasattr(self, 'push_to_hub') or not self.push_to_hub:
            return
        self.hub_token = hub_token
        assert ModelScopeConfig.get_token() is not None, 'Please enter hub_token'
        if self.hub_model_id is None:
            self.hub_model_id = f'{self.model_type}-{self.sft_type}'
            logger.info(f'Setting hub_model_id: {self.hub_model_id}')
        logger.info('hub login successful!')

    def load_from_ckpt_dir(self, is_sft: bool = False) -> None:
        if is_sft:
            ckpt_dir = self.resume_from_checkpoint
        else:
            ckpt_dir = self.ckpt_dir
        sft_args_path = os.path.join(ckpt_dir, 'sft_args.json')
        export_args_path = os.path.join(ckpt_dir, 'export_args.json')
        from_sft_args = os.path.exists(sft_args_path)
        if not os.path.exists(sft_args_path) and not os.path.exists(export_args_path):
            logger.warning(f'{sft_args_path} not found')
            return
        args_path = sft_args_path if from_sft_args else export_args_path
        with open(args_path, 'r', encoding='utf-8') as f:
            old_args = json.load(f)

        imported_keys = [
            'model_type', 'model_revision', 'template_type', 'dtype', 'quant_method', 'quantization_bit',
            'bnb_4bit_comp_dtype', 'bnb_4bit_quant_type', 'bnb_4bit_use_double_quant', 'model_id_or_path',
            'custom_register_path', 'custom_dataset_info'
        ]
        if (isinstance(self, SftArguments) and self.train_backend == 'megatron'
                or isinstance(self, ExportArguments) and self.to_hf is True):
            imported_keys += ['tp', 'pp']
        if not is_sft:
            imported_keys += ['sft_type', 'rope_scaling', 'system']
            if getattr(self, 'load_dataset_config', False) and from_sft_args:
                imported_keys += [
                    'dataset', 'val_dataset', 'dataset_seed', 'dataset_test_ratio', 'check_dataset_strategy',
                    'self_cognition_sample', 'model_name', 'model_author', 'train_dataset_sample', 'val_dataset_sample'
                ]
        for key in imported_keys:
            if not hasattr(self, key):
                continue
            value = getattr(self, key)
            old_value = old_args.get(key)
            if old_value is None:
                continue
            if key in {'dataset', 'val_dataset'} and len(value) > 0:
                continue
            if key in {
                    'system', 'quant_method', 'model_id_or_path', 'custom_register_path', 'custom_dataset_info',
                    'dataset_seed'
            } and value is not None:
                continue
            if key in {'template_type', 'dtype'} and value != 'AUTO':
                continue
            setattr(self, key, old_value)

        # compat
        if self.val_dataset is None:
            self.val_dataset = []


@dataclass
class SftArguments(ArgumentsBase):
    # You can specify the model by either using the model_type or model_id_or_path.
    model_type: Optional[str] = field(
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    model_id_or_path: Optional[str] = None
    model_revision: Optional[str] = None

    full_determinism: bool = False

    sft_type: Literal['lora', 'full', 'longlora', 'adalora', 'ia3', 'llamapro', 'adapter', 'vera', 'boft', 'fourierft',
                      'reft'] = 'lora'
    freeze_parameters: List[str] = field(default_factory=list)
    freeze_vit: bool = False
    freeze_parameters_ratio: float = 0.  # 0 ~ 1
    additional_trainable_parameters: List[str] = field(default_factory=list)
    tuner_backend: Literal['swift', 'peft', 'unsloth'] = 'peft'
    template_type: str = field(
        default='AUTO', metadata={'help': f"template_type choices: {list(TEMPLATE_MAPPING.keys()) + ['AUTO']}"})
    output_dir: str = 'output'
    add_output_dir_suffix: Optional[bool] = None
    ddp_backend: Optional[Literal['nccl', 'gloo', 'mpi', 'ccl', 'hccl']] = None
    ddp_find_unused_parameters: Optional[bool] = None
    ddp_broadcast_buffers: Optional[bool] = None
    ddp_timeout: int = 1800

    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    resume_only_model: bool = False
    ignore_data_skip: bool = False
    dtype: Literal['bf16', 'fp16', 'fp32', 'AUTO'] = 'AUTO'
    packing: bool = False
    # megatron
    train_backend: Literal['transformers', 'megatron'] = 'transformers'
    tp: int = 1
    pp: int = 1
    min_lr: Optional[float] = None
    sequence_parallel: bool = False

    # multimodal
    model_kwargs: Optional[str] = None
    loss_name: Optional[str] = field(default=None, metadata={'help': f'loss_func choices: {list(LOSS_MAPPING.keys())}'})

    # dataset_id or dataset_name or dataset_path or ...
    dataset: List[str] = field(
        default_factory=list, metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    val_dataset: List[str] = field(
        default_factory=list, metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    dataset_seed: Optional[int] = None
    dataset_test_ratio: float = 0.01
    use_loss_scale: bool = False  # for agent
    loss_scale_config_path: str = 'DEFAULT'
    system: Optional[str] = None
    tools_prompt: Literal['react_en', 'react_zh', 'toolbench'] = 'react_en'
    max_length: int = 2048  # -1: no limit
    truncation_strategy: Literal['delete', 'truncation_left'] = 'delete'
    check_dataset_strategy: Literal['none', 'discard', 'error', 'warning'] = 'none'
    # streaming dataset
    streaming: bool = False
    streaming_val_size: int = 0
    streaming_buffer_size: int = 16384
    # Chinese name and English name
    model_name: List[str] = field(default_factory=lambda: [None, None], metadata={'help': "e.g. ['小黄', 'Xiao Huang']"})
    model_author: List[str] = field(
        default_factory=lambda: [None, None], metadata={'help': "e.g. ['魔搭', 'ModelScope']"})

    # note: bf16 and quantization have requirements for gpu architecture
    # awq, gptq, and aqlm need to be pre-quantized models,
    # while bnb, hqq, and eetq can be quantized during SFT using the original models.
    quant_method: Literal['bnb', 'hqq', 'eetq', 'awq', 'gptq', 'aqlm'] = None
    quantization_bit: Literal[0, 1, 2, 3, 4, 8] = 0  # hqq: 1,2,3,4,8. bnb: 4,8
    hqq_axis: Literal[0, 1] = 0
    hqq_dynamic_config_path: Optional[str] = None
    bnb_4bit_comp_dtype: Literal['fp16', 'bf16', 'fp32', 'AUTO'] = 'AUTO'
    bnb_4bit_quant_type: Literal['fp4', 'nf4'] = 'nf4'
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_storage: Optional[str] = None

    # multi-modal
    rescale_image: int = -1

    # tuners
    target_modules: List[str] = field(default_factory=lambda: ['DEFAULT'])
    target_regex: Optional[str] = None
    # e.g. ['wte', 'ln_1', 'ln_2', 'ln_f', 'lm_head']
    modules_to_save: List[str] = field(default_factory=list)

    # lora
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias_trainable: Literal['none', 'all'] = 'none'
    lora_dtype: Literal['fp16', 'bf16', 'fp32', 'AUTO'] = 'AUTO'
    lora_lr_ratio: float = None
    use_rslora: bool = False
    use_dora: bool = False
    # Literal['gaussian', 'pissa', 'pissa_niter_[number of iters]', 'olora', 'loftq', 'true', 'false']
    init_lora_weights: str = 'true'

    # fourierft
    fourier_n_frequency: int = 2000
    fourier_scaling: float = 300.0

    # rope-scaling
    rope_scaling: Literal['linear', 'dynamic'] = None

    # BOFT
    boft_block_size: int = 4
    boft_block_num: int = 0
    boft_n_butterfly_factor: int = 1
    boft_dropout: float = 0.0

    # Vera
    vera_rank: int = 256
    vera_projection_prng_key: int = 0
    vera_dropout: float = 0.0
    vera_d_initial: float = 0.1

    # adapter
    adapter_act: str = 'gelu'
    adapter_length: int = 128

    # galore
    use_galore: bool = False
    galore_target_modules: Optional[List[str]] = None
    galore_rank: int = 128
    galore_update_proj_gap: int = 50
    galore_scale: float = 1.0
    galore_proj_type: str = 'std'
    galore_optim_per_parameter: bool = False
    galore_with_embedding: bool = False
    galore_quantization: bool = False
    galore_proj_quant: bool = False
    galore_proj_bits: int = 4
    galore_proj_group_size: int = 256
    galore_cos_threshold: float = 0.4
    galore_gamma_proj: int = 2
    galore_queue_size: int = 5

    # adalora
    adalora_target_r: int = 8
    adalora_init_r: int = 12
    adalora_tinit: int = 0
    adalora_tfinal: int = 0
    adalora_deltaT: int = 1
    adalora_beta1: float = 0.85
    adalora_beta2: float = 0.85
    adalora_orth_reg_weight: float = 0.5

    # ia3
    ia3_feedforward_modules: List[str] = field(default_factory=list)

    # llamapro
    llamapro_num_new_blocks: int = 4
    llamapro_num_groups: Optional[int] = None

    # neftune
    neftune_noise_alpha: Optional[float] = None  # e.g. 5, 10, 15
    neftune_backend: Literal['swift', 'transformers'] = None

    # lisa
    lisa_activated_layers: int = 0
    lisa_step_interval: int = 20

    # reft
    reft_layer_key: Optional[str] = None
    reft_layers: Optional[List[int]] = None
    reft_rank: int = 4
    reft_intervention_type: Literal['NoreftIntervention', 'LoreftIntervention', 'ConsreftIntervention',
                                    'LobireftIntervention', 'DireftIntervention',
                                    'NodireftIntervention'] = 'LoreftIntervention'
    reft_args: Optional[str] = None

    # use_liger
    use_liger: bool = False

    gradient_checkpointing: Optional[bool] = None
    vit_use_gc: bool = True  # vit use gradient_checkpointing
    # e.g. 'default-zero3', 'default-zero2', 'ds_config/zero2.json', 'zero2-offload', 'zero3-offload'
    deepspeed: Optional[str] = None
    batch_size: int = 1
    eval_batch_size: Optional[int] = None
    auto_find_batch_size: bool = False
    num_train_epochs: int = 1
    # if max_steps >= 0, override num_train_epochs
    max_steps: int = -1
    optim: str = 'adamw_torch'
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    learning_rate: Optional[float] = None
    weight_decay: float = 0.1
    gradient_accumulation_steps: Optional[int] = None
    max_grad_norm: float = 1
    predict_with_generate: bool = False
    lr_scheduler_type: str = 'cosine'
    lr_scheduler_kwargs: Optional[str] = None  # json
    warmup_ratio: float = 0.05
    warmup_steps: int = 0  # Overrides any effect of `warmup_ratio` if warmup_steps > 0

    eval_steps: Optional[int] = None  # full: 200, other: 50
    save_steps: Optional[int] = None
    save_only_model: bool = False
    save_total_limit: int = 2  # save last and best. -1: all checkpoints
    logging_steps: int = 5
    acc_steps: int = 1
    dataloader_num_workers: Optional[int] = None
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = False

    # push to ms hub
    push_to_hub: bool = False
    # 'user_name/repo_name' or 'repo_name'
    hub_model_id: Optional[str] = None
    # None: use env var `MODELSCOPE_API_TOKEN`
    hub_token: Optional[str] = field(
        default=None, metadata={'help': 'SDK token can be found in https://modelscope.cn/my/myaccesstoken'})
    hub_private_repo: bool = False
    hub_strategy: Literal['end', 'every_save', 'checkpoint', 'all_checkpoints'] = 'every_save'

    # other
    test_oom_error: bool = field(
        default=False,
        metadata={
            'help':
            'If set to True, the train_dataset will be sorted in descending order based on max_length, '
            'enabling faster detection of OOM (Out of Memory) errors.'
        })
    disable_tqdm: bool = False
    lazy_tokenize: Optional[bool] = None
    preprocess_num_proc: int = 1
    use_flash_attn: Optional[bool] = None
    ignore_args_error: bool = False  # True: notebook compatibility
    check_model_is_latest: bool = True

    logging_dir: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ['tensorboard'])
    acc_strategy: Literal['token', 'sentence'] = 'token'
    save_on_each_node: bool = False
    evaluation_strategy: Literal['steps', 'epoch', 'no'] = 'steps'
    save_strategy: Literal['steps', 'epoch', 'no'] = 'steps'
    save_safetensors: bool = True
    gpu_memory_fraction: Optional[float] = None
    include_num_input_tokens_seen: Optional[bool] = False
    local_repo_path: Optional[str] = None
    custom_register_path: Optional[str] = None  # .py
    custom_dataset_info: Optional[str] = None  # .json

    device_map_config: Optional[str] = None
    device_max_memory: List[str] = field(default_factory=list)

    # generation config
    max_new_tokens: int = 2048
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    num_beams: int = 1

    # fsdp option
    fsdp: Optional[str] = ''
    # fsdp config file
    fsdp_config: Optional[str] = None

    sequence_parallel_size: int = 1
    # for torchacc
    model_layer_cls_name: Optional[str] = field(
        default=None,
        metadata={'help': "Decoder Class name of model, e.g. 'QWenBlock' for QWen, 'LlamaDecoderLayer' for LLama"})
    metric_warmup_step: Optional[float] = 0
    fsdp_num: int = 1

    # compatibility hf
    per_device_train_batch_size: Optional[int] = None
    per_device_eval_batch_size: Optional[int] = None
    eval_strategy: Literal['steps', 'epoch', 'no', None] = None
    # compatibility. (Deprecated)
    self_cognition_sample: int = 0
    train_dataset_mix_ratio: float = 0.
    train_dataset_mix_ds: List[str] = field(default_factory=lambda: ['ms-bench'])
    train_dataset_sample: int = -1  # -1: all dataset
    val_dataset_sample: Optional[int] = None  # -1: all dataset
    safe_serialization: Optional[bool] = None
    only_save_model: Optional[bool] = None
    neftune_alpha: Optional[float] = None
    deepspeed_config_path: Optional[str] = None
    model_cache_dir: Optional[str] = None
    lora_dropout_p: Optional[float] = None
    lora_target_modules: List[str] = field(default_factory=list)
    lora_target_regex: Optional[str] = None
    lora_modules_to_save: List[str] = field(default_factory=list)
    boft_target_modules: List[str] = field(default_factory=list)
    boft_modules_to_save: List[str] = field(default_factory=list)
    vera_target_modules: List[str] = field(default_factory=list)
    vera_modules_to_save: List[str] = field(default_factory=list)
    ia3_target_modules: List[str] = field(default_factory=list)
    ia3_modules_to_save: List[str] = field(default_factory=list)

    custom_train_dataset_path: List[str] = field(default_factory=list)
    custom_val_dataset_path: List[str] = field(default_factory=list)
    device_map_config_path: Optional[str] = None
    push_hub_strategy: Optional[Literal['end', 'push_best', 'push_last', 'checkpoint', 'all_checkpoints']] = None

    def _prepare_target_modules(self, target_modules) -> Union[List[str], str]:
        if isinstance(target_modules, str):
            target_modules = [target_modules]
        if len(target_modules) == 0:
            return target_modules
        elif len(target_modules) == 1:
            if ',' in target_modules[0]:
                target_modules = target_modules[0].split(',')
        if 'AUTO' in target_modules:
            target_modules.remove('AUTO')
            target_modules.append('DEFAULT')
        if 'DEFAULT' in target_modules:
            target_modules.remove('DEFAULT')
            default_lora_tm = get_default_lora_target_modules(self.model_type) or []
            if isinstance(default_lora_tm, str):
                return default_lora_tm
            target_modules += default_lora_tm
        if 'EMBEDDING' in target_modules:
            self.lora_use_embedding = True
        if 'ALL' in target_modules:
            self.lora_use_all = True
        return target_modules

    def handle_lr_scheduler_kwargs(self):
        if self.lr_scheduler_kwargs is None:
            self.lr_scheduler_kwargs = {}
        elif isinstance(self.lr_scheduler_kwargs, str):
            self.lr_scheduler_kwargs = json.loads(self.lr_scheduler_kwargs)

    def _prepare_modules_to_save(self, modules_to_save) -> List[str]:
        if isinstance(modules_to_save, str):
            modules_to_save = [modules_to_save]
        if len(modules_to_save) == 0:
            return modules_to_save
        if 'EMBEDDING' in modules_to_save:
            modules_to_save.remove('EMBEDDING')
            self.lora_m2s_use_embedding = True
        if 'LN' in modules_to_save:
            modules_to_save.remove('LN')
            self.lora_m2s_use_ln = True
        return modules_to_save

    def __post_init__(self) -> None:
        super().__post_init__()
        self.handle_compatibility()
        if self.preprocess_num_proc and self.preprocess_num_proc > 1:
            os.environ['DATASET_MAP_NPROC'] = str(self.preprocess_num_proc)
        if len(self.val_dataset) > 0:
            self.dataset_test_ratio = 0.0
            logger.info('Using val_dataset, ignoring dataset_test_ratio')
        if is_pai_training_job():
            self._handle_pai_compat()
        ds_config_folder = os.path.abspath(os.path.join(__file__, '..', '..', 'ds_config'))
        deepspeed_mapping = {
            'default-zero2': 'zero2.json',
            'default-zero3': 'zero3.json',
            'zero2-offload': 'zero2_offload.json',
            'zero3-offload': 'zero3_offload.json',
        }
        for ds_name, ds_config in deepspeed_mapping.items():
            if self.deepspeed == ds_name:
                self.deepspeed = os.path.join(ds_config_folder, ds_config)
                break
        if self.loss_scale_config_path:
            if self.loss_scale_config_path == 'DEFAULT':
                self.loss_scale_config_path = os.path.abspath(
                    os.path.join(__file__, '..', '..', 'agent', 'default_loss_scale_config.json'))
            elif self.loss_scale_config_path == 'alpha-umi':  # https://arxiv.org/pdf/2401.07324
                self.loss_scale_config_path = os.path.abspath(
                    os.path.join(__file__, '..', '..', 'agent', 'alpha_umi_loss_scale_config.json'))
            elif self.loss_scale_config_path == 'agent-flan':  # https://arxiv.org/abs/2403.12881
                self.loss_scale_config_path = os.path.abspath(
                    os.path.join(__file__, '..', '..', 'agent', 'agentflan.json'))
        if self.train_backend == 'megatron' and self.resume_from_checkpoint is None:
            self.resume_from_checkpoint = f'{self.model_type}-tp{self.tp}-pp{self.pp}'
        self.handle_path()
        self._handle_dataset_sample()
        self._register_self_cognition()
        self.handle_custom_register()
        self.handle_custom_dataset_info()
        if self.resume_from_checkpoint:
            self.load_from_ckpt_dir(True)
            if self.sft_type == 'full' or self.train_backend == 'megatron':
                self.model_id_or_path = self.resume_from_checkpoint

        if self.rope_scaling:
            logger.info(f'rope_scaling is set to {self.rope_scaling}, please remember to set max_length')

        if self.dataset_seed is None:
            self.dataset_seed = self.seed
        self.set_model_type()
        self.check_flash_attn()
        self.handle_lr_scheduler_kwargs()
        self.is_multimodal = self._is_multimodal(self.model_type)
        self.is_vision = self._is_vision(self.model_type)

        self.lora_use_embedding = False
        self.lora_use_all = False
        self.lora_m2s_use_embedding = False
        self.lora_m2s_use_ln = False
        self.target_modules = self._prepare_target_modules(self.target_modules)
        self.modules_to_save = self._prepare_modules_to_save(self.modules_to_save)
        if self.use_self_cognition and self.sft_type == 'lora' and not self.lora_use_all:
            logger.warning('Due to knowledge editing involved, it is recommended to add LoRA on MLP. '
                           'For example: `--lora_target_modules ALL`. '
                           'If you have already added LoRA on MLP, please ignore this warning.')

        if self.sft_type in {'adalora', 'ia3'} and self.lora_use_embedding:
            raise ValueError('`adalora` and `ia3` do not support setting embedding as target_modules.')

        self.torch_dtype, self.fp16, self.bf16 = self.select_dtype()
        self.rank, self.local_rank, self.world_size, self.local_world_size = get_dist_setting()
        if is_dist():
            if is_torch_npu_available():
                torch.npu.set_device(self.local_rank)
            else:
                torch.cuda.set_device(self.local_rank)
            self.seed += self.rank  # Avoid the same dropout
            if self.ddp_backend is None:
                self.ddp_backend = 'nccl'
            if self.ddp_backend == 'gloo' and self.quantization_bit != 0:
                raise ValueError('not supported, please use `nccl`')

        if self.train_backend == 'megatron' and self.sft_type == 'lora':
            logger.warning('Currently, only full parameter is supported. Setting args.sft_type: "full"')
            self.sft_type = 'full'

        model_info = MODEL_MAPPING[self.model_type]
        if is_adapter(self.sft_type):
            assert self.freeze_parameters_ratio == 0., (
                'lora does not support `freeze_parameters_ratio`, please set `--sft_type full`')
            assert len(self.additional_trainable_parameters) == 0, (
                'lora does not support `additional_trainable_parameters`, please set `--sft_type full`')
            if is_quant_model(self.model_type):
                assert self.quantization_bit == 0, (
                    f'{self.model_type} is already a quantized model and does not need to be quantized again.')
            if self.learning_rate is None:
                self.learning_rate = 1e-4
            if self.eval_steps is None:
                self.eval_steps = 50
        elif self.sft_type == 'full':
            mllm_arch = get_mllm_arch(self.model_type)
            if mllm_arch:
                if self.freeze_vit and mllm_arch.vision_tower:
                    self.freeze_parameters += mllm_arch.vision_tower
                if mllm_arch.generator:
                    self.freeze_parameters += mllm_arch.generator
            assert 0 <= self.freeze_parameters_ratio <= 1
            assert self.quantization_bit == 0, 'Full parameter fine-tuning does not support quantization.'
            assert self.dtype != 'fp16', ("Fine-tuning with dtype=='fp16' can lead to NaN issues. "
                                          'Please use fp32+AMP or bf16 to perform full parameter fine-tuning.')
            if isinstance(self.additional_trainable_parameters, str):
                self.additional_trainable_parameters = [self.additional_trainable_parameters]
            if self.learning_rate is None:
                self.learning_rate = 1e-5
            if self.eval_steps is None:
                self.eval_steps = 200
        else:
            raise ValueError(f'sft_type: {self.sft_type}')

        self.prepare_template()
        if len(self.dataset) == 0:
            raise ValueError(f'self.dataset: {self.dataset}, Please input the training dataset.')

        if self.save_steps is None:
            self.save_steps = self.eval_steps

        if self.use_liger:
            assert is_liger_available(), 'use_liger requires liger_kernels, try `pip install liger-kernel`'
            if self.use_loss_scale:
                logger.warn('use_liger is not compatible with `use_loss_scale`, setting to False...')
                self.use_loss_scale = False

        # compatibility
        if self.quantization_bit > 0 and self.quant_method is None:
            if self.quantization_bit == 4 or self.quantization_bit == 8:
                logger.info('Since you have specified quantization_bit as greater than 0 '
                            "and have not designated a quant_method, quant_method will be set to 'bnb'.")
                self.quant_method = 'bnb'
            else:
                self.quant_method = 'hqq'
                logger.info('Since you have specified quantization_bit as greater than 0 '
                            "and have not designated a quant_method, quant_method will be set to 'hqq'.")

        self.bnb_4bit_compute_dtype, self.load_in_4bit, self.load_in_8bit = self.select_bnb()

        if self.neftune_backend is None:
            self.neftune_backend = 'swift' if version.parse(transformers.__version__) < version.parse('4.35') \
                else 'transformers'

        self.prepare_ms_hub()
        self.train_sampler_random = not self.test_oom_error
        if self.eval_batch_size is None:
            if self.predict_with_generate:
                self.eval_batch_size = 1
            else:
                self.eval_batch_size = self.batch_size
        if self.save_total_limit == -1:
            self.save_total_limit = None

        if self.deepspeed is not None:
            if is_mp():
                raise ValueError('DeepSpeed is not compatible with MP. '
                                 f'n_gpu: {torch.cuda.device_count()}, '
                                 f'local_world_size: {self.local_world_size}.')
            require_version('deepspeed')
            if self.deepspeed.endswith('.json') or os.path.isfile(self.deepspeed):
                with open(self.deepspeed, 'r', encoding='utf-8') as f:
                    self.deepspeed = json.load(f)
            logger.info(f'Using deepspeed: {self.deepspeed}')

        if self.gradient_accumulation_steps is None:
            self.gradient_accumulation_steps = math.ceil(16 / self.batch_size / self.world_size)
        template_info = TEMPLATE_MAPPING[self.template_type]
        self._handle_streaming_args()
        if self.lazy_tokenize is None and not self.streaming:
            self.lazy_tokenize = template_info.get('lazy_tokenize', False)
            logger.info(f'Setting args.lazy_tokenize: {self.lazy_tokenize}')
        if self.dataloader_num_workers is None:
            if 'dataloader_num_workers' in template_info:
                self.dataloader_num_workers = template_info['dataloader_num_workers']
            elif platform.system() == 'Windows':
                self.dataloader_num_workers = 0
            else:
                self.dataloader_num_workers = 1
            logger.info(f'Setting args.dataloader_num_workers: {self.dataloader_num_workers}')
        if 'dataloader_pin_memory' in template_info:
            self.dataloader_pin_memory = template_info['dataloader_pin_memory']
            logger.info(f'Setting args.dataloader_pin_memory: {self.dataloader_pin_memory}')
        if 'qwen-audio' in self.model_type:
            assert self.preprocess_num_proc == 1 or self.lazy_tokenize, 'not support'
        support_gradient_checkpointing = model_info.get('support_gradient_checkpointing', True)
        if self.gradient_checkpointing is None:
            self.gradient_checkpointing = support_gradient_checkpointing
        elif not support_gradient_checkpointing and self.gradient_checkpointing:
            logger.warning(f'{self.model_type} not support gradient_checkpointing.')

        if use_torchacc():
            self.dataloader_drop_last = True

        if self.train_backend == 'transformers':
            self._init_training_args()
        else:
            assert is_dist(), 'Please start in distributed mode.'
            dist.init_process_group(backend=self.ddp_backend)
            if self.min_lr is None:
                self.min_lr = self.learning_rate * 0.1
        if self.add_output_dir_suffix is None:
            self.add_output_dir_suffix = True
        if self.add_output_dir_suffix:
            if self.train_backend == 'megatron':
                self.output_dir = os.path.join(self.output_dir, f'{self.model_type}-tp{self.tp}-pp{self.pp}')
            else:
                self.output_dir = os.path.join(self.output_dir, self.model_type)
            self.output_dir = add_version_to_work_dir(self.output_dir)
            logger.info(f'output_dir: {self.output_dir}')
            if self.train_backend == 'transformers':
                self.training_args.output_dir = self.output_dir
                self.training_args.run_name = self.output_dir
        if is_local_master():
            os.makedirs(self.output_dir, exist_ok=True)
        if self.logging_dir is None:
            self.logging_dir = f'{self.output_dir}/runs'
            if self.train_backend == 'transformers':
                self.training_args.logging_dir = self.logging_dir
        self.handle_generation_config()

    def _init_training_args(self) -> None:
        self.train_type = self.rlhf_type if hasattr(self, 'rlhf_type') else 'sft'
        training_args_cls, kwargs = TrainerFactory.get_training_args_info(self)
        additional_saved_files = []
        if self.sft_type == 'full':
            additional_saved_files = get_additional_saved_files(self.model_type)

        if self.neftune_backend != 'swift':
            kwargs['neftune_noise_alpha'] = self.neftune_noise_alpha

        parameters = inspect.signature(training_args_cls.__init__).parameters
        for k in ['lr_scheduler_kwargs', 'include_num_input_tokens_seen', 'auto_find_batch_size']:
            if k in parameters:
                kwargs[k] = getattr(self, k)
        if 'eval_strategy' in parameters:
            kwargs['eval_strategy'] = self.evaluation_strategy
        else:
            kwargs['evaluation_strategy'] = self.evaluation_strategy

        if 'accelerator_config' in parameters:
            kwargs['accelerator_config'] = {'dispatch_batches': False}

        metric_for_best_model = 'rouge-l' if self.predict_with_generate else 'loss'
        if hasattr(self, 'rlhf_type') and self.rlhf_type == 'ppo':
            metric_for_best_model = None

        if version.parse('1.0') <= version.parse(accelerate.__version__) < version.parse('1.1'):
            self.dataset_seed = None

        training_args = training_args_cls(
            output_dir=self.output_dir,
            logging_dir=self.logging_dir,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio,
            warmup_steps=self.warmup_steps,
            logging_steps=self.logging_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            remove_unused_columns=False,
            bf16=self.bf16,
            fp16=self.fp16,
            eval_steps=self.eval_steps,
            dataloader_num_workers=self.dataloader_num_workers,
            dataloader_pin_memory=self.dataloader_pin_memory,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=self.predict_with_generate,
            full_determinism=self.full_determinism,
            optim=self.optim,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            adam_epsilon=self.adam_epsilon,
            hub_model_id=self.hub_model_id,
            hub_private_repo=self.hub_private_repo,
            hub_strategy=self.hub_strategy,
            hub_token=self.hub_token,
            push_to_hub=self.push_to_hub,
            resume_from_checkpoint=self.resume_from_checkpoint,
            ignore_data_skip=self.ignore_data_skip,
            ddp_backend=self.ddp_backend,
            gradient_checkpointing=self.gradient_checkpointing,
            local_rank=self.local_rank,
            save_only_model=self.save_only_model,
            train_sampler_random=self.train_sampler_random,
            report_to=self.report_to,
            deepspeed=self.deepspeed,
            additional_saved_files=additional_saved_files,
            disable_tqdm=self.disable_tqdm,
            save_on_each_node=self.save_on_each_node,
            acc_strategy=self.acc_strategy,
            save_safetensors=self.save_safetensors,
            logging_first_step=True,
            metric_warmup_step=self.metric_warmup_step,
            fsdp=self.fsdp,
            fsdp_config=self.fsdp_config,
            dataloader_drop_last=self.dataloader_drop_last,
            seed=self.seed,
            data_seed=self.dataset_seed,
            loss_name=self.loss_name,
            **kwargs)

        training_args.ddp_find_unused_parameters = self.ddp_find_unused_parameters
        training_args.ddp_broadcast_buffers = self.ddp_broadcast_buffers
        training_args.ddp_timeout = self.ddp_timeout
        if is_dist() and training_args.ddp_find_unused_parameters is None:
            if self.gradient_checkpointing:
                training_args.ddp_find_unused_parameters = False
            else:
                training_args.ddp_find_unused_parameters = True

        if is_dist() and training_args.ddp_broadcast_buffers is None:
            if self.gradient_checkpointing:
                training_args.ddp_broadcast_buffers = False
            else:
                training_args.ddp_broadcast_buffers = True

        self.training_args = training_args

    def _handle_pai_compat(self) -> None:
        assert is_pai_training_job()
        logger.info('Handle pai compat...')
        pai_tensorboard_dir = get_pai_tensorboard_dir()
        if self.logging_dir is None and pai_tensorboard_dir is not None:
            self.logging_dir = pai_tensorboard_dir
            logger.info(f'Setting args.logging_dir: {self.logging_dir}')
        if self.add_output_dir_suffix is None:
            self.add_output_dir_suffix = False
            logger.info(f'Setting args.add_output_dir_suffix: {self.add_output_dir_suffix}')

    def _handle_streaming_args(self) -> None:
        if not self.streaming:
            return
        if self.max_steps == -1:
            raise ValueError('Please specify `max_steps` in streaming mode.')

        if self.packing:
            self.packing = False
            logger.warning('Packing is not supported for streaming dataset, set to False')

        if self.test_oom_error:
            self.test_oom_error = False
            logger.warning('test_oom_error is not supported for streaming dataset, set to False')

        if self.lazy_tokenize:
            self.lazy_tokenize = False
            logger.info('lazy_tokenize set to False in streaming dataset')

        if self.train_dataset_mix_ratio > 0:
            logger.warning('train_dataset_mix_ratio is not supported for streaming dataset, set to 0')
            self.train_dataset_mix_ratio = 0

        if self.dataset_test_ratio > 0:
            logger.info('Set dataset_test_ratio to 0 in streaming mode.'
                        'You can manually set val_dataset and val_dataset_sample.'
                        'or set streaming_val_size instead to split from train dataset')
            self.dataset_test_ratio = 0

        if self.train_dataset_sample > 0:
            logger.warning('train_dataset_sample is not supported for streaming dataset, set to -1')
            self.train_dataset_sample = -1

        if self.dataloader_num_workers is None or self.dataloader_num_workers > 0:
            logger.info('Set dataloader_num_workers to 0 in streaming mode')
            self.dataloader_num_workers = 0


@dataclass
class InferArguments(ArgumentsBase):
    # You can specify the model by either using the model_type or model_id_or_path.
    model_type: Optional[str] = field(
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    model_id_or_path: Optional[str] = None
    model_revision: Optional[str] = None

    sft_type: Literal['lora', 'full', 'longlora', 'adalora', 'ia3', 'llamapro', 'vera', 'boft'] = 'lora'
    template_type: str = field(
        default='AUTO', metadata={'help': f"template_type choices: {list(TEMPLATE_MAPPING.keys()) + ['AUTO']}"})
    infer_backend: Literal['AUTO', 'vllm', 'pt', 'lmdeploy'] = 'AUTO'
    ckpt_dir: Optional[str] = field(default=None, metadata={'help': '/path/to/your/vx-xxx/checkpoint-xxx'})
    result_dir: Optional[str] = field(default=None, metadata={'help': '/path/to/your/infer_result'})
    load_args_from_ckpt_dir: bool = True
    load_dataset_config: bool = False
    eval_human: Optional[bool] = None

    seed: int = 42
    dtype: Literal['bf16', 'fp16', 'fp32', 'AUTO'] = 'AUTO'

    # multimodal
    model_kwargs: Optional[str] = None

    # dataset_id or dataset_name or dataset_path or ...
    dataset: List[str] = field(
        default_factory=list, metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    val_dataset: List[str] = field(
        default_factory=list, metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    dataset_seed: Optional[int] = None
    dataset_test_ratio: float = 0.01
    show_dataset_sample: int = -1
    save_result: bool = True
    system: Optional[str] = None
    tools_prompt: Literal['react_en', 'react_zh', 'toolbench'] = 'react_en'
    max_length: int = -1  # -1: no limit
    truncation_strategy: Literal['delete', 'truncation_left'] = 'delete'
    check_dataset_strategy: Literal['none', 'discard', 'error', 'warning'] = 'none'
    # Chinese name and English name
    model_name: List[str] = field(default_factory=lambda: [None, None], metadata={'help': "e.g. ['小黄', 'Xiao Huang']"})
    model_author: List[str] = field(
        default_factory=lambda: [None, None], metadata={'help': "e.g. ['魔搭', 'ModelScope']"})
    # 'awq', 'gptq', 'aqlm' are used for inference on pre-quantized models.
    quant_method: Literal['bnb', 'hqq', 'eetq', 'awq', 'gptq', 'aqlm'] = None
    quantization_bit: Literal[0, 1, 2, 3, 4, 8] = 0  # hqq: 1,2,3,4,8. bnb: 4,8
    hqq_axis: Literal[0, 1] = 0
    hqq_dynamic_config_path: Optional[str] = None
    bnb_4bit_comp_dtype: Literal['fp16', 'bf16', 'fp32', 'AUTO'] = 'AUTO'
    bnb_4bit_quant_type: Literal['fp4', 'nf4'] = 'nf4'
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_storage: Optional[str] = None

    max_new_tokens: int = 2048
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    num_beams: int = 1
    stop_words: List[str] = field(default_factory=list)

    # rope-scaling
    rope_scaling: Literal['linear', 'dynamic'] = None

    # other
    use_flash_attn: Optional[bool] = None
    ignore_args_error: bool = False  # True: notebook compatibility
    stream: bool = True
    merge_lora: bool = False
    merge_device_map: Optional[str] = None
    save_safetensors: bool = True
    overwrite_generation_config: bool = False
    verbose: Optional[bool] = None
    local_repo_path: Optional[str] = None
    custom_register_path: Optional[str] = None  # .py
    custom_dataset_info: Optional[str] = None  # .json
    device_map_config: Optional[str] = None
    device_max_memory: List[str] = field(default_factory=list)
    # None: use env var `MODELSCOPE_API_TOKEN`
    hub_token: Optional[str] = field(
        default=None, metadata={'help': 'SDK token can be found in https://modelscope.cn/my/myaccesstoken'})

    # vllm
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    max_num_seqs: int = 256
    max_model_len: Optional[int] = None
    disable_custom_all_reduce: bool = True  # Default values different from vllm
    enforce_eager: bool = False
    limit_mm_per_prompt: Optional[str] = None  # '{"image": 10, "video": 5}'
    vllm_enable_lora: bool = False
    vllm_max_lora_rank: int = 16
    lora_modules: List[str] = field(default_factory=list)
    max_logprobs: int = 20

    # lmdeploy
    tp: int = 1
    cache_max_entry_count: float = 0.8
    quant_policy: int = 0  # e.g. 4, 8
    vision_batch_size: int = 1  # max_batch_size in VisionConfig

    # compatibility. (Deprecated)
    self_cognition_sample: int = 0
    train_dataset_sample: int = -1  # Used for splitting the validation set.
    val_dataset_sample: Optional[int] = None  # -1: all dataset
    safe_serialization: Optional[bool] = None
    model_cache_dir: Optional[str] = None
    merge_lora_and_save: Optional[bool] = None
    custom_train_dataset_path: List[str] = field(default_factory=list)
    custom_val_dataset_path: List[str] = field(default_factory=list)
    vllm_lora_modules: List[str] = None
    device_map_config_path: Optional[str] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.ckpt_dir is not None and not self.check_ckpt_dir_correct(self.ckpt_dir):
            logger.warning(f'The checkpoint dir {self.ckpt_dir} passed in is invalid, please make sure'
                           'the dir contains a `configuration.json` file.')
        self.handle_compatibility()
        if len(self.val_dataset) > 0:
            self.dataset_test_ratio = 0.0
            logger.info('Using val_dataset, ignoring dataset_test_ratio')
        self.handle_path()
        logger.info(f'ckpt_dir: {self.ckpt_dir}')
        if self.ckpt_dir is None and self.load_args_from_ckpt_dir:
            self.load_args_from_ckpt_dir = False
            logger.info('Due to `ckpt_dir` being `None`, `load_args_from_ckpt_dir` is set to `False`.')
        if self.load_args_from_ckpt_dir:
            self.load_from_ckpt_dir()
        else:
            assert self.load_dataset_config is False, 'You need to first set `--load_args_from_ckpt_dir true`.'

        if self.rope_scaling:
            logger.info(f'rope_scaling is set to {self.rope_scaling}, '
                        f'please remember to set max_length, which is supposed to be the same as training')
        if self.dataset_seed is None:
            self.dataset_seed = self.seed
        self._handle_dataset_sample()
        self._register_self_cognition()
        self.handle_custom_register()
        self.handle_custom_dataset_info()
        self.set_model_type()
        self.check_flash_attn()
        self.is_multimodal = self._is_multimodal(self.model_type)
        self.prepare_ms_hub()

        self.torch_dtype, _, _ = self.select_dtype()
        self.prepare_template()
        if self.eval_human is None:
            if len(self.dataset) == 0 and len(self.val_dataset) == 0:
                self.eval_human = True
            else:
                self.eval_human = False
            logger.info(f'Setting args.eval_human: {self.eval_human}')
        elif self.eval_human is False and len(self.dataset) == 0 and len(self.val_dataset) == 0:
            raise ValueError('Please provide the dataset or set `--load_dataset_config true`.')

        # compatibility
        if self.quantization_bit > 0 and self.quant_method is None:
            if self.quantization_bit == 4 or self.quantization_bit == 8:
                logger.info('Since you have specified quantization_bit as greater than 0 '
                            "and have not designated a quant_method, quant_method will be set to 'bnb'.")
                self.quant_method = 'bnb'
            else:
                self.quant_method = 'hqq'
                logger.info('Since you have specified quantization_bit as greater than 0 '
                            "and have not designated a quant_method, quant_method will be set to 'hqq'.")

        self.bnb_4bit_compute_dtype, self.load_in_4bit, self.load_in_8bit = self.select_bnb()

        if self.ckpt_dir is None:
            self.sft_type = 'full'

        self.handle_infer_backend()
        self.handle_generation_config()
        self._load_json_or_path('limit_mm_per_prompt')

    def handle_infer_backend(self):
        model_info = MODEL_MAPPING[self.model_type]
        support_vllm = model_info.get('support_vllm', False)
        support_lmdeploy = model_info.get('support_lmdeploy', False)
        self.lora_request_list = None
        if self.infer_backend == 'AUTO':
            self.infer_backend = 'pt'
            if is_vllm_available() and support_vllm and not self.is_multimodal:
                if ((self.sft_type == 'full' or self.sft_type == 'lora' and self.merge_lora)
                        and self.quantization_bit == 0):
                    self.infer_backend = 'vllm'
                if self.vllm_enable_lora:
                    self.infer_backend = 'vllm'
            if is_lmdeploy_available() and support_lmdeploy and self.is_multimodal:
                if ((self.sft_type == 'full' or self.sft_type == 'lora' and self.merge_lora)
                        and self.quantization_bit == 0):
                    self.infer_backend = 'lmdeploy'
        if self.infer_backend == 'vllm':
            require_version('vllm')
            if not support_vllm:
                logger.warning(f'vllm not support `{self.model_type}`')
            if self.sft_type == 'lora' and not self.vllm_enable_lora:
                assert self.merge_lora, ('To use vLLM, you need to provide the complete weight parameters. '
                                         'Please set `--merge_lora true`.')
        if self.infer_backend == 'lmdeploy':
            require_version('lmdeploy')
            assert self.quantization_bit == 0, 'lmdeploy does not support bnb.'
            if not support_lmdeploy:
                logger.warning(f'lmdeploy not support `{self.model_type}`')
            if self.sft_type == 'lora':
                assert self.merge_lora, ('To use LMDeploy, you need to provide the complete weight parameters. '
                                         'Please set `--merge_lora true`.')

        if (self.infer_backend == 'vllm' and self.vllm_enable_lora
                or self.infer_backend == 'pt' and isinstance(self, DeployArguments) and self.sft_type == 'lora'):
            assert self.ckpt_dir is not None
            self.lora_modules.append(f'default-lora={self.ckpt_dir}')
            self.lora_request_list, self.use_dora = _parse_lora_modules(self.lora_modules, self.infer_backend == 'vllm')

        template_info = TEMPLATE_MAPPING[self.template_type]
        if self.num_beams != 1 or not template_info.get('stream', True):
            self.stream = False
            logger.info('Setting args.stream: False')
        self.infer_media_type = template_info.get('infer_media_type', 'none')
        if self.infer_media_type == 'none' and self.is_multimodal:
            self.infer_media_type = 'interleave'
        self.media_type = template_info.get('media_type', 'image')
        self.media_key = MediaTag.media_keys.get(self.media_type, 'images')
        if self.merge_device_map is None and not isinstance(self, ExportArguments):
            self.merge_device_map = 'cpu'

    @staticmethod
    def check_ckpt_dir_correct(ckpt_dir) -> bool:
        """Check the checkpoint dir is correct, which means it must contain a `configuration.json` file.
        Args:
            ckpt_dir: The checkpoint dir
        Returns:
            A bool value represents the dir is valid or not.
        """
        if not os.path.exists(ckpt_dir):
            return False
        return os.path.isfile(os.path.join(ckpt_dir, 'configuration.json'))


@dataclass
class AppUIArguments(InferArguments):
    host: str = '127.0.0.1'
    port: int = 7860
    share: bool = False
    # compatibility. (Deprecated)
    server_name: Optional[str] = None
    server_port: Optional[int] = None


@dataclass
class DeployArguments(InferArguments):
    host: str = '0.0.0.0'
    port: int = 8000
    api_key: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None

    owned_by: str = 'swift'
    served_model_name: Optional[str] = None
    verbose: bool = True  # Whether to log request_info
    log_interval: int = 10  # Interval for printing global statistics


@dataclass
class EvalArguments(InferArguments):

    eval_dataset: List[str] = field(default_factory=list)
    eval_few_shot: Optional[int] = None
    eval_limit: Optional[str] = None

    name: str = ''
    eval_url: Optional[str] = None
    eval_token: str = 'EMPTY'
    eval_is_chat_model: Optional[bool] = None
    custom_eval_config: Optional[str] = None  # path
    eval_use_cache: bool = False
    eval_output_dir: str = 'eval_outputs'
    eval_backend: Literal['Native', 'OpenCompass'] = 'OpenCompass'
    eval_batch_size: int = 8
    deploy_timeout: int = 1800

    do_sample: bool = False  # Note: for evaluation default is False
    temperature: float = 0.
    eval_nproc: int = 16

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.eval_dataset, str):
            self.eval_dataset = [self.eval_dataset]
        if len(self.eval_dataset) == 1 and self.eval_dataset[0] == 'no':
            self.eval_dataset = []
        if self.eval_url is not None and (self.eval_is_chat_model is None or self.model_type is None):
            model = get_model_list_client(url=self.eval_url).data[0]
            if self.eval_is_chat_model is None:
                self.eval_is_chat_model = model.is_chat
            if self.model_type is None:
                self.model_type = model.id

    def select_dtype(self):
        if self.eval_url is None:
            return super().select_dtype()
        return None, None, None

    def set_model_type(self) -> None:
        if self.eval_url is None:
            super().set_model_type()

    def check_flash_attn(self) -> None:
        if self.eval_url is None:
            super().check_flash_attn()

    def prepare_template(self) -> None:
        if self.eval_url is None:
            super().prepare_template()

    def handle_infer_backend(self) -> None:
        if self.eval_url is None:
            super().handle_infer_backend()

    def _is_multimodal(self, model_type: Optional[str] = None) -> bool:
        return False if self.eval_url is not None else super()._is_multimodal(model_type)

    def _is_vision(self, model_type: Optional[str] = None) -> bool:
        return False if self.eval_url is not None else super()._is_vision(model_type)


@dataclass
class ExportArguments(InferArguments):
    to_peft_format: bool = False
    to_ollama: bool = False
    ollama_output_dir: Optional[str] = None
    gguf_file: Optional[str] = None

    # awq: 4; gptq: 2, 3, 4, 8
    quant_bits: int = 0  # e.g. 4
    quant_method: Literal['awq', 'gptq', 'bnb'] = 'awq'
    quant_n_samples: int = 256
    quant_seqlen: int = 2048
    quant_device_map: Optional[str] = None  # e.g. 'cpu', 'auto'
    quant_output_dir: Optional[str] = None
    quant_batch_size: int = 1

    # push to ms hub
    push_to_hub: bool = False
    # 'user_name/repo_name' or 'repo_name'
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = False
    commit_message: str = 'update files'

    # megatron
    to_megatron: bool = False
    to_hf: bool = False
    megatron_output_dir: Optional[str] = None
    hf_output_dir: Optional[str] = None
    tp: int = 1
    pp: int = 1

    # The parameter has been defined in InferArguments.
    # merge_lora, hub_token

    def __post_init__(self):
        if self.merge_device_map is None and self.quant_bits > 0:
            self.merge_device_map = 'cpu'
        if self.quant_bits > 0 and self.dtype == 'AUTO':
            self.dtype = 'fp16'
            logger.info(f'Setting args.dtype: {self.dtype}')
        super().__post_init__()
        if self.quant_bits > 0:
            if len(self.dataset) == 0:
                self.dataset = ['alpaca-zh#10000', 'alpaca-en#10000']
                logger.info(f'Setting args.dataset: {self.dataset}')
            if self.quant_output_dir is None:
                if self.ckpt_dir is None:
                    self.quant_output_dir = f'{self.model_type}-{self.quant_method}-int{self.quant_bits}'
                else:
                    ckpt_dir, ckpt_name = os.path.split(self.ckpt_dir)
                    self.quant_output_dir = os.path.join(ckpt_dir,
                                                         f'{ckpt_name}-{self.quant_method}-int{self.quant_bits}')
                self.quant_output_dir = self._check_path(self.quant_output_dir)
                logger.info(f'Setting args.quant_output_dir: {self.quant_output_dir}')
            assert not os.path.exists(self.quant_output_dir), f'args.quant_output_dir: {self.quant_output_dir}'
        elif self.to_ollama:
            assert self.sft_type in ('full', 'lora', 'longlora', 'llamapro')
            if self.sft_type in ('lora', 'longlora', 'llamapro'):
                self.merge_lora = True
            if not self.ollama_output_dir:
                self.ollama_output_dir = f'{self.model_type}-ollama'
            self.ollama_output_dir = self._check_path(self.ollama_output_dir)
            assert not os.path.exists(
                self.ollama_output_dir), f'Please make sure your output dir does not exists: {self.ollama_output_dir}'
        elif self.to_megatron or self.to_hf:
            self.quant_method = None
            os.environ['RANK'] = '0'
            os.environ['LOCAL_RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            os.environ['LOCAL_WORLD_SIZE'] = '1'
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
            assert is_dist(), 'Please start in distributed mode.'
            dist.init_process_group(backend='nccl')
        if self.to_megatron:
            if self.megatron_output_dir is None:
                self.megatron_output_dir = f'{self.model_type}-tp{self.tp}-pp{self.pp}'
            self.megatron_output_dir = self._check_path(self.megatron_output_dir)
            logger.info(f'Setting args.megatron_output_dir: {self.megatron_output_dir}')
        if self.to_hf:
            if self.hf_output_dir is None:
                self.hf_output_dir = os.path.join(self.ckpt_dir, f'{self.model_type}-hf')
            self.hf_output_dir = self._check_path(self.hf_output_dir)
            logger.info(f'Setting args.hf_output_dir: {self.hf_output_dir}')


@dataclass
class PtArguments(SftArguments):
    sft_type: Literal['lora', 'full', 'longlora', 'adalora', 'ia3', 'llamapro', 'vera', 'boft'] = 'full'
    target_modules: List[str] = field(default_factory=lambda: ['ALL'])
    lazy_tokenize: Optional[bool] = True
    eval_steps: int = 500


@dataclass
class RLHFArguments(SftArguments):
    rlhf_type: Literal['dpo', 'orpo', 'simpo', 'kto', 'cpo', 'rm', 'ppo'] = 'dpo'
    ref_model_type: Optional[str] = field(
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    ref_model_id_or_path: Optional[str] = None
    ref_model_revision: Optional[str] = None

    beta: Optional[float] = None
    label_smoothing: float = 0.0
    # dpo: 'sigmoid', 'hinge', 'ipo', 'exo_pair', 'nca_pair', 'robust', 'bco_pair',
    #      'sppo_hard', 'aot', 'aot_pair', 'apo_zero', 'apo_down'
    # cpo: 'sigmoid', 'hinge', 'ipo', 'simpo'
    loss_type: Optional[str] = None
    # DPO
    # The alpha parameter from the [RPO](https://huggingface.co/papers/2404.19733) paper V3.
    # The paper recommends `rpo_alpha=1.0`.
    rpo_alpha: float = 1.
    # CPO
    cpo_alpha: float = 1.
    # SimPO
    simpo_gamma: float = 1
    # KTO
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0
    # PPO
    reward_model_id_or_path: Optional[str] = None
    reward_model_type: Optional[str] = field(
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    reward_model_revision: Optional[str] = None
    local_rollout_forward_batch_size: int = 64
    whiten_rewards: bool = False
    kl_coef: float = 0.05
    cliprange: float = 0.2
    vf_coef: float = 0.1
    cliprange_value: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    num_sample_generations: int = 10

    def __post_init__(self):
        self._check_simpo()
        self._set_default()
        self.ref_model_free = self.rlhf_type in ['cpo', 'orpo', 'rm']
        from contextlib import nullcontext, contextmanager

        @contextmanager
        def ppocontext():
            from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig
            from transformers.utils import is_sagemaker_mp_enabled
            if is_sagemaker_mp_enabled():
                import smdistributed.modelparallel.torch as smp
                smp.init()
            old_trainer_config_process = HfTrainerDeepSpeedConfig.trainer_config_process

            def trainer_config_process(self, args, auto_find_batch_size=False):
                if args.world_size is None:
                    if args.distributed_state is not None:
                        return args.distributed_state.num_processes
                    elif is_sagemaker_mp_enabled():
                        return smp.dp_size() if not smp.state.cfg.prescaled_batch else smp.rdp_size()
                    return 1
                return old_trainer_config_process(self, args, auto_find_batch_size)

            HfTrainerDeepSpeedConfig.trainer_config_process = trainer_config_process
            yield
            HfTrainerDeepSpeedConfig.trainer_config_process = old_trainer_config_process

        context = nullcontext if self.rlhf_type != 'ppo' else ppocontext
        with context():
            super().__post_init__()
        self._check_ppo()

    def _check_simpo(self):
        if self.rlhf_type != 'simpo':
            return

        self.rlhf_type = 'cpo'
        if self.loss_type is None:
            self.loss_type = 'simpo'
        if self.beta is None:
            self.beta = 2.

    def _set_default(self):
        if self.beta is None:
            self.beta = 0.1
        if self.loss_type is None:
            if self.rlhf_type in ['dpo', 'cpo']:
                self.loss_type = 'sigmoid'  # else None
            elif self.rlhf_type in ['kto']:
                self.loss_type = 'kto'
        if self.rlhf_type == 'ppo':
            self.response_length = self.max_new_tokens
            logger.info(f'set max_new_tokens {self.max_new_tokens} in generation config during ppo,'
                        'you can set by --max_new_tokens')
            self.num_ppo_epochs = self.num_train_epochs
            logger.info(
                f'set num_ppo_epochs {self.num_train_epochs} in ppo training config, you can set by --num_train_epochs')

    def _check_ppo(self):
        if self.rlhf_type != 'ppo':
            return
        if self.streaming:
            raise ValueError('Streaming is currently not supported by PPO')
        if self.is_multimodal:
            raise ValueError('MLLM is currently not supported by PPO')


@dataclass
class WebuiArguments:
    share: bool = False
    lang: str = 'zh'
    host: str = '127.0.0.1'
    port: int = 7860


@dataclass
class RomeArguments(InferArguments):
    rome_request_file: str = field(
        default=None, metadata={'help': 'The rome request file, please check the documentation '
                                'to get the format'})

    def __post_init__(self) -> None:
        self.handle_compatibility()
        self.handle_path()
        self.set_model_type()
        self.check_flash_attn()

        self.torch_dtype, _, _ = self.select_dtype()
        if self.template_type == 'AUTO':
            self.template_type = get_default_template_type(self.model_type)
            logger.info(f'Setting template_type: {self.template_type}')

        if self.max_length == -1:
            self.max_length = None


dtype_mapping_reversed = {v: k for k, v in dtype_mapping.items()}


def swift_to_peft_format(lora_checkpoint_path: str) -> str:
    if 'default' in os.listdir(lora_checkpoint_path):  # swift_backend
        new_lora_checkpoint_path = f'{lora_checkpoint_path}-peft'
        Swift.save_to_peft_format(lora_checkpoint_path, new_lora_checkpoint_path)
        lora_checkpoint_path = new_lora_checkpoint_path
        logger.info('Converting the swift format checkpoint to peft format, '
                    f"and saving it to: '{new_lora_checkpoint_path}'")
    else:
        logger.info('The format of the checkpoint is already in peft format.')
    return lora_checkpoint_path


def _parse_lora_modules(lora_modules: List[str], use_vllm: bool) -> Tuple[List[Any], bool]:
    VllmLoRARequest = None
    if use_vllm:
        try:
            from .vllm_utils import LoRARequest as VllmLoRARequest
        except ImportError:
            logger.warning('The current version of VLLM does not support `enable_lora`. Please upgrade VLLM.')
            raise

    @dataclass
    class PtLoRARequest:
        lora_name: str
        lora_int_id: int
        lora_local_path: str

    LoRARequest = VllmLoRARequest if use_vllm else PtLoRARequest
    lora_request_list = []
    use_dora_list = []
    for i, lora_module in enumerate(lora_modules):
        lora_name, lora_local_path = lora_module.split('=')
        lora_local_path = swift_to_peft_format(lora_local_path)
        with open(os.path.join(lora_local_path, 'adapter_config.json'), 'r') as f:
            _json = json.load(f)
            use_dora_list.append(_json.get('use_dora', False))
        lora_request_list.append(LoRARequest(lora_name, i + 1, lora_local_path))
    if any(use_dora_list) and len(lora_modules) > 1:
        raise ValueError('Dora does not support inference with other loras')
    elif not any(use_dora_list):
        use_dora = False
    else:
        use_dora = True
    return lora_request_list, use_dora
