from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Model:
    ms_model_id: Optional[str] = None
    hf_model_id: Optional[str] = None
    model_path: Optional[str] = None

    ms_revision: Optional[str] = None
    hf_revision: Optional[str] = None


@dataclass
class ModelGroup:
    models: List[Model]
    template: str
    # File patterns to ignore when downloading the model.
    ignore_file_pattern: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # Higher priority. If set to None, the attributes of the DatasetMeta will be used.
    requires: Optional[List[str]] = None
    support_flash_attn: Optional[bool] = None
    support_vllm: Optional[bool] = None
    support_lmdeploy: Optional[bool] = None
    support_megatron: Optional[bool] = None


@dataclass
class ModelMeta:
    model_type: str
    # Used to automatically infer the model_type from config.json.
    architectures: List[str]
    # Used to list the model_ids from huggingface/modelscope,
    # which participate in the automatic inference of the model_type.
    model_groups: List[ModelGroup]
    get_function: GetModelTokenizerFunction

    template: Optional[str] = None
    is_moe: bool = False
    is_multimodal: bool = False
    # Additional files that need to be saved for full parameter training/merge-lora.
    additional_saved_files: List[str] = field(default_factory=list)
    support_gradient_checkpointing: bool = True

    # Usually specifies the version limits of transformers.
    requires: List[str] = field(default_factory=list)
    support_flash_attn: bool = False
    support_vllm: bool = False
    support_lmdeploy: bool = False
    support_megatron: bool = False

    def get_matched_model_groups(self, model_dir: str) -> List[ModelGroup]:
        model_name = HfConfigFactory._get_model_name(model_dir).lower()
        res = []
        seen = set()
        for model_group in self.model_groups:
            id_ = id(model_group)
            for model, key in itertools.product(model_group.models, ['ms_model_id', 'hf_model_id', 'model_path']):
                value = getattr(model, key)
                if value is None:
                    continue
                m_name = value.rsplit('/', 1)[-1].lower()
                if m_name == model_name and id_ not in seen:
                    seen.add(id_)
                    res.append(model_group)
                    break
        if len(res) == 0:
            return self.model_groups
        return res

    def get_model_names(self) -> List[str]:
        res = set()
        for model_group in self.model_groups:
            for model in model_group.models:
                for key in ['ms_model_id', 'hf_model_id', 'model_path']:
                    value = getattr(model, key)

                    if isinstance(value, str):
                        model_name = value.rsplit('/', 1)[-1]
                        res.add(model_name)
        return list(res)

    def check_requires(self):
        # TODO: error to warning
        for require in self.requires:
            require_version(require)

    def check_flash_attn(self, attn_impl: Optional[str]) -> None:
        if attn_impl is None:
            return
        if attn_impl == AttnImpl.flash_attn and not self.support_flash_attn:
            logger.warning(f'attn_impl: {attn_impl}, but support_flash_attn: {self.support_flash_attn}')

    def check_infer_backend(self, infer_backend: str) -> None:
        if infer_backend == 'vllm' and not self.support_vllm:
            logger.warning(f'infer_backend: {infer_backend}, but support_vllm: {self.support_vllm}')
        elif infer_backend == 'lmdeploy' and not self.support_lmdeploy:
            logger.warning(f'infer_backend: {infer_backend}, but support_lmdeploy: {self.support_lmdeploy}')

    def check_gradient_checkpointing(self, gradient_checkpoint: bool) -> None:
        if gradient_checkpoint and not self.support_gradient_checkpointing:
            logger.warning(f'gradient_checkpoint: {gradient_checkpoint}, but support_gradient_checkpointing: '
                           f'{self.support_gradient_checkpointing}')
