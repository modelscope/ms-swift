import os

lang = os.environ.get('SWIFT_UI_LANG', 'zh')
locales = {}


def add_locale_labels(locale_dict, group):
    if group not in locales:
        locales[group] = {}
    for key, value in locale_dict.items():
        if key not in locales:
            locales[group][key] = {}
        for sub_key in value:
            locales[group][key][sub_key] = value[sub_key][lang]


def get_locale_by_group(group):
    return locales[group]

