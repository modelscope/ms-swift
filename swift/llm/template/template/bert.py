from ..register import TemplateMeta, register_template
from ..constant import LLMTemplateType

register_template(
    TemplateMeta(
        LLMTemplateType.bert,
        prefix=[],
        prompt=['{{QUERY}}[SEP]'],
        chat_sep=['[SEP]'],
        auto_add_bos=True))
