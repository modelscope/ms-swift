# Copyright (c) Alibaba, Inc. and its affiliates.
import dataclasses
import os
import sys
import time
import typing
from collections import OrderedDict
from dataclasses import fields
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Type

import gradio as gr
import json
from gradio import Accordion, Audio, Button, Checkbox, Dropdown, File, Image, Slider, Tab, TabItem, Textbox, Video
from modelscope.hub.utils.utils import get_cache_dir

from swift.llm import TEMPLATE_MAPPING, BaseArguments, get_matched_model_meta

all_langs = ['zh', 'en']
builder: Type['BaseUI'] = None
base_builder: Type['BaseUI'] = None


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

        if base_builder and base_builder.default(elem_id) is not None and not kwargs.get('value'):
            kwargs['value'] = base_builder.default(elem_id)

        if builder is not None:
            if elem_id in builder.locales(builder.lang):
                values = builder.locale(elem_id, builder.lang)
                if 'info' in values:
                    kwargs['info'] = values['info']
                if 'value' in values:
                    kwargs['value'] = values['value']
                if 'label' in values:
                    kwargs['label'] = values['label']
                if hasattr(builder, 'visible'):
                    kwargs['visible'] = builder.visible
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
File.__init__ = update_data(File.__init__)
Image.__init__ = update_data(Image.__init__)
Video.__init__ = update_data(Video.__init__)
Audio.__init__ = update_data(Audio.__init__)


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
    cache_dir = os.path.join(get_cache_dir(), 'swift-web-ui')
    os.makedirs(cache_dir, exist_ok=True)
    quote = '\'' if sys.platform != 'win32' else '"'
    visible = True
    _locale = {
        'local_dir_alert': {
            'value': {
                'zh': '无法识别model_type和template,请手动选择',
                'en': 'Cannot recognize the model_type and template, please choose manully'
            }
        },
    }

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
        if cls is base_tab:
            for ui in cls.sub_ui:
                ui.after_build_ui(base_tab)

    @classmethod
    def after_build_ui(cls, base_tab: Type['BaseUI']):
        pass

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        """Build UI"""
        pass

    @classmethod
    def save_cache(cls, key, value):
        timestamp = str(int(time.time()))
        key = key.replace('/', '-')
        filename = os.path.join(cls.cache_dir, key + '-' + timestamp)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(value, f)

    @classmethod
    def list_cache(cls, key):
        files = []
        key = key.replace('/', '-')
        for _, _, filenames in os.walk(cls.cache_dir):
            for filename in filenames:
                if filename.startswith(key):
                    idx = filename.rfind('-')
                    key, ts = filename[:idx], filename[idx + 1:]
                    dt_object = datetime.fromtimestamp(int(ts))
                    formatted_time = dt_object.strftime('%Y/%m/%d %H:%M:%S')
                    files.append(formatted_time)
        return sorted(files, reverse=True)

    @classmethod
    def load_cache(cls, key, timestamp) -> BaseArguments:
        dt_object = datetime.strptime(timestamp, '%Y/%m/%d %H:%M:%S')
        timestamp = int(dt_object.timestamp())
        key = key.replace('/', '-')
        filename = key + '-' + str(timestamp)
        with open(os.path.join(cls.cache_dir, filename), 'r', encoding='utf-8') as f:
            return json.load(f)

    @classmethod
    def clear_cache(cls, key):
        key = key.replace('/', '-')
        for _, _, filenames in os.walk(cls.cache_dir):
            for filename in filenames:
                if filename.startswith(key):
                    os.remove(os.path.join(cls.cache_dir, filename))

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
        if elem_id in cls.default_dict:
            return cls.default_dict.get(elem_id)
        for sub_ui in BaseUI.sub_ui:
            _choice = sub_ui.default(elem_id)
            if _choice:
                return _choice
        return None

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
    def valid_elements(cls):
        valid_elements = OrderedDict()
        elements = cls.elements()
        for key, value in elements.items():
            if isinstance(value, (Textbox, Dropdown, Slider, Checkbox)) and key != 'train_record':
                valid_elements[key] = value
        return valid_elements

    @classmethod
    def element_keys(cls):
        return list(cls.elements().keys())

    @classmethod
    def valid_element_keys(cls):
        return [
            key for key, value in cls.elements().items()
            if isinstance(value, (Textbox, Dropdown, Slider, Checkbox)) and key != 'train_record'
        ]

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
            if f.default.__class__ is dataclasses._MISSING_TYPE:
                default_dict[f.name] = f.default_factory()
            else:
                default_dict[f.name] = f.default
            if isinstance(default_dict[f.name], list):
                try:
                    default_dict[f.name] = ' '.join(default_dict[f.name])
                except TypeError:
                    default_dict[f.name] = None
            if not default_dict[f.name]:
                default_dict[f.name] = None
        return default_dict

    @staticmethod
    def get_argument_names(dataclass):
        arguments = {}
        for f in fields(dataclass):
            arguments[f.name] = f'--{f.name}'
        return arguments

    @classmethod
    def update_input_model(cls, model, allow_keys=None, has_record=True, arg_cls=BaseArguments, is_ref_model=False):
        keys = cls.valid_element_keys()

        if not model:
            ret = [gr.update()] * (len(keys) + int(has_record))
            if len(ret) == 1:
                return ret[0]
            else:
                return ret

        model_meta = get_matched_model_meta(model)
        local_args_path = os.path.join(model, 'args.json')
        if model_meta is None and not os.path.exists(local_args_path):
            gr.Info(cls._locale['local_dir_alert']['value'][cls.lang])
            ret = [gr.update()] * (len(keys) + int(has_record))
            if len(ret) == 1:
                return ret[0]
            else:
                return ret

        if os.path.exists(local_args_path):
            try:
                if hasattr(arg_cls, 'resume_from_checkpoint'):
                    args = arg_cls(resume_from_checkpoint=model, load_data_args=True)
                else:
                    args = arg_cls(ckpt_dir=model, load_data_args=True)
            except ValueError:
                return [gr.update()] * (len(keys) + int(has_record))
            values = []
            for key in keys:
                if allow_keys is not None and key not in allow_keys:
                    continue
                arg_value = getattr(args, key, None)
                if arg_value and key != 'model':
                    if key in ('torch_dtype', 'bnb_4bit_compute_dtype'):
                        arg_value = str(arg_value).split('.')[1]
                    if isinstance(arg_value, list) and key != 'dataset':
                        try:
                            arg_value = ' '.join(arg_value)
                        except Exception:
                            arg_value = None
                    values.append(gr.update(value=arg_value))
                else:
                    values.append(gr.update())
            ret = [gr.update(choices=[])] * int(has_record) + values
            if len(ret) == 1:
                return ret[0]
            else:
                return ret
        else:
            values = []
            for key in keys:
                if allow_keys is not None and key not in allow_keys:
                    continue
                if key not in ('template', 'model_type', 'ref_model_type', 'system'):
                    values.append(gr.update())
                elif key in ('template', 'model_type', 'ref_model_type'):
                    if key == 'ref_model_type':
                        if is_ref_model:
                            values.append(gr.update(value=getattr(model_meta, 'model_type')))
                        else:
                            values.append(gr.update())
                    else:
                        values.append(gr.update(value=getattr(model_meta, key)))
                else:
                    values.append(gr.update(value=TEMPLATE_MAPPING[model_meta.template].default_system))

        if has_record:
            return [gr.update(choices=cls.list_cache(model))] + values
        else:
            if len(values) == 1:
                return values[0]
            return values

    @classmethod
    def update_all_settings(cls, model, train_record, base_tab):
        if not train_record:
            return [gr.update()] * len(cls.elements())
        cache = cls.load_cache(model, train_record)
        updates = []
        for key, value in cls.valid_elements().items():
            if key in cache:
                updates.append(gr.update(value=cache[key]))
            else:
                updates.append(gr.update())
        return updates
