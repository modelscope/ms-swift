import os
from functools import wraps
from typing import Dict, List, OrderedDict, Type

from gradio import (Accordion, Button, Checkbox, Dropdown, Slider, Tab,
                    TabItem, Textbox)

all_langs = ['zh', 'en']
builder: Type['BaseUI'] = None
lang = os.environ.get('SWIFT_UI_LANG', all_langs[0])


def update_data(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        elem_id = kwargs.get('elem_id', None)
        self = args[0]

        if builder is not None:
            choices = builder.choice(elem_id)
            if choices:
                kwargs['choices'] = choices

        if not isinstance(
                self,
            (Tab, TabItem, Accordion)) and 'interactive' not in kwargs:  # noqa
            kwargs['interactive'] = True

        if 'is_list' in kwargs:
            self.is_list = kwargs.pop('is_list')

        if builder is not None:
            if elem_id in builder.locales(lang):
                values = builder.locale(elem_id, lang)
                if 'info' in values:
                    kwargs['info'] = values['info']
                if 'value' in values:
                    kwargs['value'] = values['value']
                if 'label' in values:
                    kwargs['label'] = values['label']

        ret = fn(self, **kwargs)

        if hasattr(self, 'change'):

            def change(value):
                self.changed = True
                arg_value = value
                if isinstance(value, list):
                    arg_value = ' '.join(value)
                    self.is_list = True
                self.arg_value = arg_value
                return value

            self.change(change, [self], [self], queue=False)

            value = getattr(self, 'value', None)
            self.arg_value = ' '.join(value) if isinstance(value,
                                                           list) else value

        if builder is not None:
            builder.element_dict[elem_id] = self
        return ret

    return wrapper


Textbox.__init__ = update_data(Textbox.__init__)
Dropdown.__init__ = update_data(Dropdown.__init__)
Checkbox.__init__ = update_data(Checkbox.__init__)
Slider.__init__ = update_data(Slider.__init__)
TabItem.__init__ = update_data(TabItem.__init__)
Accordion.__init__ = update_data(Accordion.__init__)
Button.__init__ = update_data(Button.__init__)


class BaseUI:

    choice_dict: Dict[str, List] = {}
    locale_dict: Dict[str, Dict] = {}
    element_dict: Dict[str, Dict] = {}
    sub_ui: List[Type['BaseUI']] = []
    group: str = None
    lang: str = all_langs[0]

    @classmethod
    def build_ui(cls, base_tab: Type['BaseUI']):
        """Build UI"""
        global builder
        cls.element_dict = {}
        old_builder = builder
        builder = cls
        cls.do_build_ui(base_tab)
        builder = old_builder

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        """Build UI"""
        pass

    @classmethod
    def choice(cls, elem_id):
        """Get choice by elem_id"""
        for sub_ui in BaseUI.sub_ui:
            _choice = sub_ui.choice(elem_id)
            if _choice:
                return _choice
        return cls.choice_dict.get(elem_id, [])

    @classmethod
    def locale(cls, elem_id, lang):
        """Get locale by elem_id"""
        return cls.locales(lang)[elem_id]

    @classmethod
    def locales(cls, lang):
        """Get locale by lang"""
        locales = OrderedDict()
        for sub_ui in cls.sub_ui:
            _locales = sub_ui.locales(lang)
            locales.update(_locales)
        for key, value in cls.locale_dict.items():
            locales[key] = {k: v[lang] for k, v in value.items()}
        return locales

    @classmethod
    def elements(cls):
        """Get all elements"""
        elements = OrderedDict()
        elements.update(cls.element_dict)
        for sub_ui in cls.sub_ui:
            _elements = sub_ui.elements()
            elements.update(_elements)
        return elements

    @classmethod
    def element(cls, elem_id):
        """Get element by elem_id"""
        elements = cls.elements()
        return elements[elem_id]

    @classmethod
    def set_lang(cls, lang):
        cls.lang = lang
        for sub_ui in cls.sub_ui:
            sub_ui.lang = lang
