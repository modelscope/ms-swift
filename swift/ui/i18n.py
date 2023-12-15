import os

lang = os.environ.get('SWIFT_UI_LANG', 'zh')
components = {}


def get_i18n_labels(i18n):
    for key, value in i18n.items():
        if key not in components:
            components[key] = {}
        for sub_key in value:
            components[key][sub_key] = value[sub_key][lang]
