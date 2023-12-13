import os
from functools import wraps

from swift.llm import SftArguments
from swift.ui.llm.utils import get_choices
from swift.ui.i18n import components
from gradio import Textbox, Dropdown, Slider, Checkbox, Accordion, TabItem

lang = os.environ.get('SWIFT_UI_LANG', 'zh')
elements = {}


def update_data(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        elem_id = kwargs.pop('elem_id', None)
        self = args[0]
        if elem_id in components:
            values = components[elem_id]
            if 'info' in values:
                kwargs['info'] = values['info']
            kwargs['label'] = values['label']
        if hasattr(SftArguments, elem_id):
            kwargs['value'] = getattr(SftArguments, elem_id)
            choices = get_choices(elem_id)
            if choices:
                kwargs['choices'] = choices
        elements[elem_id] = self
        return fn(self, **kwargs)

    return wrapper


Textbox.__init__ = update_data(Textbox.__init__)
Dropdown.__init__ = update_data(Dropdown.__init__)
Checkbox.__init__ = update_data(Checkbox.__init__)
Slider.__init__ = update_data(Slider.__init__)
TabItem.__init__ = update_data(TabItem.__init__)
Accordion.__init__ = update_data(Accordion.__init__)
