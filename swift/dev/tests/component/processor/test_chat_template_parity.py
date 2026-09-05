# Copyright (c) ModelScope Contributors. All rights reserved.
"""Every registered template, checked against the tokenizer's own ``apply_chat_template``.

``apply_chat_template`` is the reference because it is the model author's jinja, shipped in
``tokenizer_config.json``: if dev's template renders a different prompt string then inference in any
other runtime sees different text than training did, and no loss curve will show it.

One conversation per template rather than a suite per template -- rendering is cheap next to
encoding, and the failure being guarded (a family's markup drifting from its own jinja) shows up on
the first multi-turn sample. What makes this affordable across ~150 templates is that only tokenizer
files are fetched, never weights.

Templates whose representative model ships no ``chat_template`` (base models, embedding and reranker
heads) are skipped with that reason rather than silently passing.
"""
import pytest

from swift.dev.tests.tiny import TinyModel

#: system + two exchanges: enough to expose a wrong turn separator, a dropped system prompt, or a
#: generation prompt appended in the wrong place.
CONVERSATION = [
    {
        'role': 'system',
        'content': 'You are helpful.'
    },
    {
        'role': 'user',
        'content': 'hi'
    },
    {
        'role': 'assistant',
        'content': 'hello'
    },
    {
        'role': 'user',
        'content': 'bye'
    },
]


class TemplateSurvey:
    """Pair each template with one model that can supply a tokenizer."""

    @staticmethod
    def representatives() -> dict:
        """template name -> a model id registered against it.

        The registry is keyed by model_type, so walk it once and keep the first model id per
        template. Templates with no registered model cannot be checked this way and are left out.
        """
        from swift.model.register import MODEL_MAPPING

        found = {}
        for meta in MODEL_MAPPING.values():
            template = getattr(meta, 'template', None)
            if not template or template in found:
                continue
            for group in getattr(meta, 'model_groups', None) or []:
                for model in getattr(group, 'models', None) or []:
                    model_id = getattr(model, 'ms_model_id', None) or getattr(model, 'hf_model_id', None)
                    if model_id:
                        found[template] = model_id
                        break
                if template in found:
                    break
        return found


REPRESENTATIVES = TemplateSurvey.representatives()


@pytest.mark.slow
@pytest.mark.parametrize('template', sorted(REPRESENTATIVES), ids=sorted(REPRESENTATIVES))
def test_renders_the_same_prompt_as_apply_chat_template(template):
    """dev's token ids must equal ``apply_chat_template(..., add_generation_prompt=True)``.

    Token ids rather than decoded text: ``decode`` is not the inverse of ``encode`` for sentencepiece
    tokenizers -- it reinserts word-piece boundaries as spaces (``assistant`` came back as
    ``ass istant``), which reads like a template defect and is not one.
    """
    from swift.dev.builders import build_template
    from swift.dev.config import TemplateConfig
    from swift.model import get_model_processor

    model_id = REPRESENTATIVES[template]
    try:
        model_dir = TinyModel.tokenizer_dir(model_id)
    except Exception as exc:  # noqa: BLE001 -- a missing or gated repo is not a template defect
        pytest.skip(f'{model_id}: tokenizer not fetchable ({type(exc).__name__})')

    _, processor = get_model_processor(model_dir, load_model=False)
    tokenizer = processor if hasattr(processor, 'apply_chat_template') else processor.tokenizer
    if getattr(tokenizer, 'chat_template', None) is None:
        pytest.skip(f'{model_id} ships no chat_template, so there is nothing to compare against')

    reference = list(tokenizer.apply_chat_template(CONVERSATION, tokenize=True,
                                                   add_generation_prompt=True)['input_ids'])

    dev_template = build_template(TemplateConfig(template=template, max_length=1024), processor)
    dev_template.set_mode('pt')
    rendered = list(dev_template.encode({'messages': CONVERSATION})['input_ids'])

    assert rendered == reference, (f'{template}: dev renders\n  {tokenizer.decode(rendered)!r}\n'
                                  f'but the model jinja renders\n  {tokenizer.decode(reference)!r}')
