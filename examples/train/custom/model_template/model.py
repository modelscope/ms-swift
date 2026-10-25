# Custom model family + custom chat template, registered as a plugin and loaded with --external_plugins.
#
# Importing this file (via --external_plugins, before anything is built) runs registration side effects.
# A custom `--model_type` is looked up in TWO independent registries during one SFT run, so a custom
# family registers in both -- they are not derived from each other:
#
#   1. dev's model-loader registry (swift.dev.model.loader). build_model resolves `--model_type`
#      through it to pick the transformers family loader; an UNregistered model_type there is treated as
#      an external plugin path, so the family must be a @register_model ModelLoader subclass. It declares
#      the checkpoints it covers (`models`), the template it speaks (`template`) and the transformers
#      class to build (`model_cls`, which has no safe default).
#   2. the legacy model registry (swift.model). build_template -> load_model_processor delegates to
#      swift.model.get_model_processor, which validates `--model_type` against the legacy MODEL_MAPPING;
#      a custom family is a register_model(ModelMeta(...)) entry listing the same ids and template.
#
# The chat TEMPLATE follows the same two-registry rule: the v5 CLI resolves `--template` through
# swift.dev.template's registry (get_template_meta consults it directly, with NO legacy fallback),
# while template-consuming legacy paths still look in swift.template's registry -- so a custom
# template registers in BOTH. TemplateMeta requires template_type / prefix / prompt / chat_sep;
# {{SYSTEM}} / {{QUERY}} are the placeholders dev fills. Select it with --template custom_im.
#
# Here the family just re-points at a stock Qwen2.5 checkpoint with our own template, which is enough to
# exercise both registries end to end; a real custom family lists its own `models`, and overrides the
# loader's build_config / build_processor / build_model hooks when the plain AutoModel path is not enough.
from swift.dev.model.loader import ModelLoader, register_model
from swift.dev.template import TemplateMeta as DevTemplateMeta
from swift.dev.template import register_template as register_dev_template
from swift.model import Model, ModelGroup, ModelMeta
from swift.model import register_model as register_legacy_model
from swift.template import TemplateMeta, register_template


@register_model(exist_ok=True)
class CustomIMLoader(ModelLoader):
    model_type = 'custom_im'
    model_cls = 'transformers:AutoModelForCausalLM'
    template = 'custom_im'
    models = ['Qwen/Qwen2.5-0.5B-Instruct']


register_legacy_model(
    ModelMeta(
        model_type='custom_im',
        model_groups=[ModelGroup([
            Model('Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct'),
        ])],
        template='custom_im',
        architectures=['Qwen2ForCausalLM'],
    ),
    exist_ok=True)

register_template(
    TemplateMeta(
        template_type='custom_im',
        prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
        prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        chat_sep=['<|im_end|>\n'],
        default_system='You are a helpful assistant.'))

register_dev_template(
    DevTemplateMeta(
        template_type='custom_im',
        prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
        prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        chat_sep=['<|im_end|>\n'],
        default_system='You are a helpful assistant.'))
