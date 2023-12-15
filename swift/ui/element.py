import os
from functools import wraps

from gradio import (Accordion, Button, Checkbox, Dropdown, Slider, Tab,
                    TabItem, Textbox)
from gradio.events import Changeable

from swift.llm import SftArguments
from swift.ui.i18n import components
from swift.ui.llm.utils import get_choices

lang = os.environ.get('SWIFT_UI_LANG', 'zh')
elements = {}


def update_data(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        elem_id = kwargs.get('elem_id', None)
        self = args[0]
        if elem_id in components:
            values = components[elem_id]
            if 'info' in values:
                kwargs['info'] = values['info']
            if 'value' in values:
                kwargs['value'] = values['value']
            if 'label' in values:
                kwargs['label'] = values['label']
        if hasattr(SftArguments, elem_id) and getattr(SftArguments, elem_id):
            kwargs['value'] = getattr(SftArguments, elem_id)
            choices = get_choices(elem_id)
            if choices:
                kwargs['choices'] = choices
        if not isinstance(
                self,
            (Tab, TabItem, Accordion)) and 'interactive' not in kwargs:  # noqa
            kwargs['interactive'] = True
        if 'is_list' in kwargs:
            self.is_list = kwargs.pop('is_list')
        elements[elem_id] = self
        ret = fn(self, **kwargs)
        if isinstance(self, Changeable):

            def change(value):
                self.changed = True
                if isinstance(value, list):
                    value = ' '.join(value)
                    self.is_list = True
                self.last_value = value

            self.change(change, [self], [])
        return ret

    return wrapper


Textbox.__init__ = update_data(Textbox.__init__)
Dropdown.__init__ = update_data(Dropdown.__init__)
Checkbox.__init__ = update_data(Checkbox.__init__)
Slider.__init__ = update_data(Slider.__init__)
TabItem.__init__ = update_data(TabItem.__init__)
Accordion.__init__ = update_data(Accordion.__init__)
Button.__init__ = update_data(Button.__init__)
