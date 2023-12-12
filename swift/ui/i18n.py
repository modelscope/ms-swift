import json
import os

from gradio import Accordion, TabItem
from gradio.components.base import IOComponent


current_dir = os.path.dirname(__file__)
lang = os.environ.get('SWIFT_UI_LANG', 'zh')
components = {}
extras = {}

with open(os.path.join(current_dir, 'config', 'i18n.json'), 'r') as f:
    i18n = json.load(f)
    component_json = i18n.get('components', {})
    extra_json = i18n.get('extras', {})
    for key, value in component_json.items():
        components[key] = {
            "label": value["label"][lang],
        }
        if "info" in value:
            components[key]["info"] = value["info"][lang]

    for key, value in extra_json.items():
        extras[key] = value[lang]


def __init__(self, *args, **kwargs):
    self.component_name = kwargs.pop('elem_id', None)
    if self.component_name in components:
        values = components[self.component_name]
        if 'info' in values:
            kwargs['info'] = values['info']
        kwargs['label'] = values['label']
    self.constructor_args['label'] = kwargs['label']
    self.__old_init__(*args, **kwargs)


IOComponent.__old_init__ = IOComponent.__init__
IOComponent.__init__ = __init__


Accordion.__old_init__ = Accordion.__init__
Accordion.__init__ = __init__

TabItem.__old_init__ = TabItem.__init__
TabItem.__init__ = __init__