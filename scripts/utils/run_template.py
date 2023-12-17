from typing import Dict, List, Tuple

from swift.llm import TemplateType


def get_template_name_list() -> List[str]:
    res = []
    for k in TemplateType.__dict__.keys():
        if k.startswith('__'):
            continue
        res.append(TemplateType.__dict__[k])
    return res


if __name__ == '__main__':
    template_name_list = get_template_name_list()
    tn_gen = ', '.join([tn for tn in template_name_list if 'generation' in tn])
    tn_chat = ', '.join(
        [tn for tn in template_name_list if 'generation' not in tn])
    print(f'Text Generation: {tn_gen}')
    print(f'Chat: {tn_chat}')
