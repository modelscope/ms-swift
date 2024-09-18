# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import field
from typing import List, Literal, Optional, Union

from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset

from swift.llm.model.loader import MODEL_MAPPING
from swift.utils import (get_logger)

logger = get_logger()
DATASET_TYPE = Union[HfDataset, HfIterableDataset]


class ArgumentsBase:

    sft_type: Literal['lora', 'full', 'longlora', 'adalora', 'ia3', 'llamapro', 'adapter', 'vera', 'boft', 'fourierft',
                      'reft'] = 'lora'

    def __post_init__(self) -> None:
        if self.max_length == -1:
            self.max_length = None
        model_kwargs = self.model_kwargs
        if model_kwargs is None:
            model_kwargs = {}
        if isinstance(model_kwargs, str):
            model_kwargs = json.loads(model_kwargs)
        for k, v in model_kwargs.items():
            k = k.upper()
            os.environ[k] = str(v)

        if isinstance(self.device_map_config, str):
            if os.path.exists(self.device_map_config):  # local path
                with open(self.device_map_config, 'r') as f:
                    self.device_map_config = json.load(f)
            else:  # json str
                self.device_map_config = json.loads(self.device_map_config)
        _, local_rank, _, local_world_size = get_dist_setting()
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
                                        and isinstance(self, SftArguments)):
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


class GenerationArguments:

    # generation config
    max_new_tokens: int = 2048
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    num_beams: int = 1
    stop_words: List[str] = field(default_factory=list)


class QuantizeArguments:
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


class ModelArguments:

    # You can specify the model by either using the model_type or model_id_or_path.
    model_type: Optional[str] = field(
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    model_id_or_path: Optional[str] = None
    model_revision: Optional[str] = None

    model_kwargs: Optional[str] = None

    # rope-scaling
    rope_scaling: Literal['linear', 'dynamic'] = None

    dtype: Literal['bf16', 'fp16', 'fp32', 'AUTO'] = 'AUTO'

    local_repo_path: Optional[str] = None

    device_map_config: Optional[str] = None
    device_max_memory: List[str] = field(default_factory=list)
