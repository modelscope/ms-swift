# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import json
import torch
import torch.distributed as dist
from torch import dtype as Dtype

from swift import HubStrategy, get_logger
from swift.hub import HubApi, ModelScopeConfig
from swift.utils import get_dist_setting, is_dist
from .dataset import DATASET_MAPPING, DatasetName
from .model import MODEL_MAPPING, ModelType
from .preprocess import TEMPLATE_MAPPING, TemplateType

logger = get_logger()


@dataclass
class SftArguments:
    model_type: str = field(
        default=ModelType.qwen_7b_chat,
        metadata={'choices': list(MODEL_MAPPING.keys())})
    sft_type: str = field(
        default='lora', metadata={'choices': ['longlora', 'lora', 'full']})
    tuner_bankend: str = field(
        default='swift', metadata={'choices': ['swift', 'peft']})
    template_type: Optional[str] = field(
        default=None, metadata={'choices': list(TEMPLATE_MAPPING.keys())})
    output_dir: str = 'output'
    ddp_backend: Optional[str] = field(
        default=None, metadata={'choices': ['nccl', 'gloo', 'mpi', 'ccl']})

    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    dtype: str = field(
        default='bf16', metadata={'choices': ['bf16', 'fp16', 'fp32']})
    ignore_args_error: bool = False  # True: notebook compatibility

    dataset: Optional[List[str]] = field(
        default=None,
        metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    dataset_split_seed: int = 42
    dataset_test_ratio: float = 0.01
    train_dataset_sample: int = 20000  # -1: all dataset
    system: str = 'you are a helpful assistant!'
    max_length: int = 2048

    # If you want to use qlora, set the quantization_bit to 8 or 4.
    # And you need to install bitsandbytes: `pip install bitsandbytes -U`
    # note: bf16 and quantization have requirements for gpu architecture
    quantization_bit: int = field(default=0, metadata={'choices': [0, 4, 8]})
    bnb_4bit_comp_dtype: str = field(
        default=None, metadata={'choices': ['fp16', 'bf16', 'fp32']})
    bnb_4bit_quant_type: str = field(
        default='nf4', metadata={'choices': ['fp4', 'nf4']})
    bnb_4bit_use_double_quant: bool = True

    lora_target_modules: Optional[List[str]] = None
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout_p: float = 0.

    gradient_checkpointing: bool = False
    deepspeed_config_path: Optional[str] = None  # e.g. 'ds_config/zero2.json'
    batch_size: int = 1
    eval_batch_size: Optional[int] = None
    num_train_epochs: int = 1
    # if max_steps >= 0, override num_train_epochs
    max_steps: int = -1
    optim: str = 'adamw_torch'
    learning_rate: Optional[float] = None
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 16
    max_grad_norm: float = 1.
    predict_with_generate: bool = False
    lr_scheduler_type: str = 'cosine'
    warmup_ratio: float = 0.05

    eval_steps: int = 50
    save_steps: Optional[int] = None
    only_save_model: Optional[bool] = None
    save_total_limit: int = 2  # save last and best. -1: all checkpoints
    logging_steps: int = 5
    dataloader_num_workers: int = 1

    push_to_hub: bool = False
    # 'user_name/repo_name' or 'repo_name'
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = True
    push_hub_strategy: str = field(
        default='push_best',
        metadata={
            'choices':
            ['end', 'push_best', 'push_last', 'checkpoint', 'all_checkpoints']
        })
    # None: use env var `MODELSCOPE_API_TOKEN`
    hub_token: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'SDK token can be found in https://modelscope.cn/my/myaccesstoken'
        })

    # other
    test_oom_error: bool = field(
        default=False,
        metadata={
            'help':
            'If set to True, the train_dataset will be sorted in descending order based on max_length, '
            'enabling faster detection of OOM (Out of Memory) errors.'
        })
    use_flash_attn: Optional[bool] = field(
        default=None,
        metadata={
            'help':
            "This parameter is used only when model_type.startswith('qwen')"
        })

    # generation config
    max_new_tokens: int = 1024
    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 20
    top_p: float = 0.9
    repetition_penalty: float = 1.

    def init_argument(self):
        # Can be manually initialized, unlike __post_init__
        handle_compatibility(self)
        if self.dtype == 'bf16' and not torch.cuda.is_bf16_supported():
            logger.info(
                'Your machine does not support bf16, automatically using fp16.'
            )
            self.dtype = 'fp16'
        if is_dist():
            rank, local_rank, _, _ = get_dist_setting()
            torch.cuda.set_device(local_rank)
            self.seed += rank  # Avoid the same dropout
            if self.ddp_backend is None:
                self.ddp_backend = 'nccl'
            if self.ddp_backend == 'gloo' and self.quantization_bit != 0:
                raise ValueError('not supported, please use `nccl`')

            # Initialize in advance
            dist.init_process_group(backend=self.ddp_backend)

        if self.sft_type == 'lora' or self.sft_type == 'longlora':
            if self.learning_rate is None:
                self.learning_rate = 1e-4
            if self.only_save_model is None:
                if self.deepspeed_config_path is None:
                    self.only_save_model = False
                else:
                    self.only_save_model = True
        elif self.sft_type == 'full':
            assert self.quantization_bit == 0, 'not supported'
            assert self.dtype != 'fp16', 'please use bf16 or fp32'
            if self.learning_rate is None:
                self.learning_rate = 2e-5
            if self.only_save_model is None:
                self.only_save_model = True
        else:
            raise ValueError(f'sft_type: {self.sft_type}')

        if self.template_type is None:
            self.template_type = MODEL_MAPPING[self.model_type].get(
                'template', TemplateType.default)
            logger.info(f'Setting template_type: {self.template_type}')
        if self.dataset is None:
            self.dataset = [DatasetName.blossom_math_zh]

        if self.save_steps is None:
            self.save_steps = self.eval_steps
        self.output_dir = os.path.join(self.output_dir, self.model_type)

        if self.lora_target_modules is None:
            self.lora_target_modules = MODEL_MAPPING[
                self.model_type]['lora_TM']
        self.torch_dtype, self.fp16, self.bf16 = select_dtype(self)
        if self.bnb_4bit_comp_dtype is None:
            self.bnb_4bit_comp_dtype = self.dtype
        self.bnb_4bit_compute_dtype, self.load_in_4bit, self.load_in_8bit = select_bnb(
            self)

        if self.hub_model_id is None:
            self.hub_model_id = f'{self.model_type}-{self.sft_type}'
            logger.info(f'Setting hub_model_id: {self.hub_model_id}')
        if self.push_to_hub:
            api = HubApi()
            if self.hub_token is None:
                self.hub_token = os.environ.get('MODELSCOPE_API_TOKEN')
            if self.hub_token is not None:
                api.login(self.hub_token)
            else:
                assert ModelScopeConfig.get_token(
                ) is not None, 'Please enter hub_token'
            logger.info('hub login successful!')

        if self.use_flash_attn is None:
            self.use_flash_attn = 'auto'
        self.train_sampler_random = not self.test_oom_error
        if self.eval_batch_size is None:
            if self.predict_with_generate:
                self.eval_batch_size = 1
            else:
                self.eval_batch_size = self.batch_size
        if self.save_total_limit == -1:
            self.save_total_limit = None

        self.deepspeed = None
        if self.deepspeed_config_path is not None:
            with open(self.deepspeed_config_path, 'r') as f:
                self.deepspeed = json.load(f)
            logger.info(f'Using deepspeed: {self.deepspeed}')


@dataclass
class InferArguments:
    model_type: str = field(
        default=ModelType.qwen_7b_chat,
        metadata={'choices': list(MODEL_MAPPING.keys())})
    sft_type: str = field(
        default='lora', metadata={'choices': ['longlora', 'lora', 'full']})
    template_type: Optional[str] = field(
        default=None, metadata={'choices': list(TEMPLATE_MAPPING.keys())})
    ckpt_dir: str = '/path/to/your/vx_xxx/checkpoint-xxx'
    eval_human: bool = False  # False: eval val_dataset

    seed: int = 42
    dtype: str = field(
        default='bf16', metadata={'choices': ['bf16', 'fp16', 'fp32']})
    ignore_args_error: bool = False  # True: notebook compatibility

    dataset: Optional[List[str]] = field(
        default=None,
        metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    dataset_split_seed: int = 42
    dataset_test_ratio: float = 0.01
    show_dataset_sample: int = 20
    system: str = 'you are a helpful assistant!'
    max_length: int = 2048

    quantization_bit: int = field(default=0, metadata={'choices': [0, 4, 8]})
    bnb_4bit_comp_dtype: str = field(
        default=None, metadata={'choices': ['fp16', 'bf16', 'fp32']})
    bnb_4bit_quant_type: str = field(
        default='nf4', metadata={'choices': ['fp4', 'nf4']})
    bnb_4bit_use_double_quant: bool = True

    max_new_tokens: int = 1024
    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 20
    top_p: float = 0.9
    repetition_penalty: float = 1.

    # other
    use_flash_attn: Optional[bool] = field(
        default=None,
        metadata={
            'help':
            "This parameter is used only when model_type.startswith('qwen')"
        })
    use_streamer: bool = True
    merge_lora_and_save: bool = False
    save_generation_config: bool = True

    def init_argument(self):
        # Can be manually initialized, unlike __post_init__
        handle_compatibility(self)
        if self.dtype == 'bf16' and not torch.cuda.is_bf16_supported():
            logger.info(
                'Your machine does not support bf16, automatically using fp16.'
            )
            self.dtype = 'fp16'
        if self.template_type is None:
            self.template_type = MODEL_MAPPING[self.model_type].get(
                'template', TemplateType.default)
            logger.info(f'Setting template_type: {self.template_type}')
        if self.dataset is None:
            self.dataset = [DatasetName.blossom_math_zh]

        self.torch_dtype, _, _ = select_dtype(self)
        if self.bnb_4bit_comp_dtype is None:
            self.bnb_4bit_comp_dtype = self.dtype
        self.bnb_4bit_compute_dtype, self.load_in_4bit, self.load_in_8bit = select_bnb(
            self)

        if self.use_flash_attn is None:
            self.use_flash_attn = 'auto'


DTYPE_MAPPING = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'fp32': torch.float32
}


def select_dtype(
        args: Union[SftArguments, InferArguments]) -> Tuple[Dtype, bool, bool]:
    dtype = args.dtype
    torch_dtype = DTYPE_MAPPING[dtype]

    assert torch_dtype in {torch.float16, torch.bfloat16, torch.float32}
    if torch_dtype == torch.float16:
        fp16, bf16 = True, False
    elif torch_dtype == torch.bfloat16:
        support_bf16 = torch.cuda.is_bf16_supported()
        if not support_bf16:
            logger.warning(f'support_bf16: {support_bf16}')
        fp16, bf16 = False, True
    else:
        fp16, bf16 = False, False
    return torch_dtype, fp16, bf16


def select_bnb(
        args: Union[SftArguments, InferArguments]) -> Tuple[Dtype, bool, bool]:
    quantization_bit = args.quantization_bit
    bnb_4bit_compute_dtype = DTYPE_MAPPING[args.bnb_4bit_comp_dtype]
    assert bnb_4bit_compute_dtype in {
        torch.float16, torch.bfloat16, torch.float32
    }
    if quantization_bit == 4:
        load_in_4bit, load_in_8bit = True, False
    elif quantization_bit == 8:
        load_in_4bit, load_in_8bit = False, True
    else:
        load_in_4bit, load_in_8bit = False, False

    return bnb_4bit_compute_dtype, load_in_4bit, load_in_8bit


def handle_compatibility(args: Union[SftArguments, InferArguments]):
    if args.dataset is not None and len(
            args.dataset) == 1 and ',' in args.dataset[0]:
        args.dataset = args.dataset[0].split(',')
