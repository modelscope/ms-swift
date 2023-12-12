import os

import json
from gradio.components.base import IOComponent
from swift.llm import SftArguments

current_dir = os.path.dirname(__file__)
lang = os.environ.get('SWIFT_UI_LANG', 'zh')
components = {}
extras = {}
elements = {}


with open(os.path.join(current_dir, 'config', 'i18n.json'), 'r') as f:
    i18n = json.load(f)
    component_json = i18n.get('components', {})
    extra_json = i18n.get('extras', {})
    for key, value in component_json.items():
        components[key] = {
            'label': value['label'][lang],
        }
        if 'info' in value:
            components[key]['info'] = value['info'][lang]

    for key, value in extra_json.items():
        extras[key] = value[lang]


def __init__(self, *args, **kwargs):
    elem_id = kwargs.pop('elem_id', None)
    if elem_id in components:
        values = components[elem_id]
        if 'info' in values:
            kwargs['info'] = values['info']
        kwargs['label'] = values['label']
    if hasattr(SftArguments, elem_id):
        kwargs['value'] = getattr(SftArguments, elem_id)
    self.__old_init__(*args, **kwargs)
    elements[elem_id] = self


IOComponent.__old_init__ = IOComponent.__init__
IOComponent.__init__ = __init__
