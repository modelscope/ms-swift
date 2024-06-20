import os
import typing
from dataclasses import fields
from functools import partial, wraps
from typing import Any, Dict, List, OrderedDict, Type

from gradio import Accordion, Button, Checkbox, Dropdown, Slider, Tab, TabItem, Textbox

from swift.llm.utils.model import MODEL_MAPPING, ModelType

all_langs = ['zh', 'en']
builder: Type['BaseUI'] = None
base_builder: Type['BaseUI'] = None
lang = os.environ.get('SWIFT_UI_LANG', all_langs[0])


def update_data(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        elem_id = kwargs.get('elem_id', None)
        self = args[0]

        if builder is not None:
            choices = base_builder.choice(elem_id)
            if choices:
                kwargs['choices'] = choices

        if not isinstance(self, (Tab, TabItem, Accordion)) and 'interactive' not in kwargs:  # noqa
            kwargs['interactive'] = True

        if 'is_list' in kwargs:
            self.is_list = kwargs.pop('is_list')

        if base_builder and base_builder.default(elem_id) is not None:
            if os.environ.get('MODELSCOPE_ENVIRONMENT') == 'studio' and kwargs.get('value') is not None:
                pass
            else:
                kwargs['value'] = base_builder.default(elem_id)

        if builder is not None:
            if elem_id in builder.locales(lang):
                values = builder.locale(elem_id, lang)
                if 'info' in values:
                    kwargs['info'] = values['info']
                if 'value' in values:
                    kwargs['value'] = values['value']
                if 'label' in values:
                    kwargs['label'] = values['label']
                argument = base_builder.argument(elem_id)
                if argument and 'label' in kwargs:
                    kwargs['label'] = kwargs['label'] + f'({argument})'

        kwargs['elem_classes'] = 'align'
        ret = fn(self, **kwargs)
        self.constructor_args.update(kwargs)

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
    default_dict: Dict[str, Any] = {}
    locale_dict: Dict[str, Dict] = {}
    element_dict: Dict[str, Dict] = {}
    arguments: Dict[str, str] = {}
    sub_ui: List[Type['BaseUI']] = []
    group: str = None
    lang: str = all_langs[0]
    int_regex = r'^[-+]?[0-9]+$'
    float_regex = r'[-+]?(?:\d*\.*\d+)'
    bool_regex = r'^(T|t)rue$|^(F|f)alse$'

    @classmethod
    def build_ui(cls, base_tab: Type['BaseUI']):
        """Build UI"""
        global builder, base_builder
        cls.element_dict = {}
        old_builder = builder
        old_base_builder = base_builder
        builder = cls
        base_builder = base_tab
        cls.do_build_ui(base_tab)
        builder = old_builder
        base_builder = old_base_builder

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
    def default(cls, elem_id):
        """Get choice by elem_id"""
        for sub_ui in BaseUI.sub_ui:
            _choice = sub_ui.default(elem_id)
            if _choice:
                return _choice
        return cls.default_dict.get(elem_id, None)

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
    def argument(cls, elem_id):
        """Get argument by elem_id"""
        return cls.arguments.get(elem_id)

    @classmethod
    def set_lang(cls, lang):
        cls.lang = lang
        for sub_ui in cls.sub_ui:
            sub_ui.lang = lang

    @staticmethod
    def get_choices_from_dataclass(dataclass):
        choice_dict = {}
        for f in fields(dataclass):
            if 'choices' in f.metadata:
                choice_dict[f.name] = f.metadata['choices']
            if 'Literal' in str(f.type) and typing.get_args(f.type):
                choice_dict[f.name] = typing.get_args(f.type)
        return choice_dict

    @staticmethod
    def get_default_value_from_dataclass(dataclass):
        default_dict = {}
        for f in fields(dataclass):
            if hasattr(dataclass, f.name):
                default_dict[f.name] = getattr(dataclass, f.name)
            else:
                default_dict[f.name] = None
        return default_dict

    @staticmethod
    def get_argument_names(dataclass):
        arguments = {}
        for f in fields(dataclass):
            arguments[f.name] = f'--{f.name}'
        return arguments

    @staticmethod
    def get_custom_name_list():
        return list(set(MODEL_MAPPING.keys()) - set(ModelType.get_model_name_list()))
