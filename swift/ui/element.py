import os
from functools import wraps

from gradio import (Accordion, Button, Checkbox, Dropdown, Slider, Tab,
                    TabItem, Textbox)
from gradio.events import Changeable

from swift.llm import SftArguments
from swift.ui.i18n import components
from swift.ui.llm_train.utils import get_choices

lang = os.environ.get('SWIFT_UI_LANG', 'zh')
elements = {}


def update_data(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        elem_id = kwargs.get('elem_id', None)
        self = args[0]
        group = kwargs.pop('group', None)
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
        ret = fn(self, **kwargs)
        if group is not None:
            self.group = group
        if isinstance(self, Changeable):

            def change(value):
                self.changed = True
                if isinstance(value, list):
                    value = ' '.join(value)
                    self.is_list = True
                self.last_value = value

            self.change(change, [self], [])

        parent_group = None
        parent = self.parent
        while parent_group is None:
            if parent is not None and hasattr(parent, 'group'):
                parent_group = parent.group
            parent = parent.parent
        assert(parent_group is not None)
        if parent_group not in elements:
            elements[parent_group] = {}
        elements[parent_group][elem_id] = self
        return ret

    return wrapper


def get_elements_by_group(group):
    return elements[group]


Textbox.__init__ = update_data(Textbox.__init__)
Dropdown.__init__ = update_data(Dropdown.__init__)
Checkbox.__init__ = update_data(Checkbox.__init__)
Slider.__init__ = update_data(Slider.__init__)
TabItem.__init__ = update_data(TabItem.__init__)
Accordion.__init__ = update_data(Accordion.__init__)
Button.__init__ = update_data(Button.__init__)
