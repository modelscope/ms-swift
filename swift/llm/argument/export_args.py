

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
    quant_device_map: str = 'cpu'  # e.g. 'cpu', 'auto'
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
        if self.merge_device_map is None:
            self.merge_device_map = 'cpu' if self.quant_bits > 0 else 'auto'
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