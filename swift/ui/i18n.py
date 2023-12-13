import os

lang = os.environ.get('SWIFT_UI_LANG', 'zh')
components = {}


def get_i18n_labels(i18n):
    for key, value in i18n.items():
        if 'label' in value:
            components[key] = {
                'label': value['label'][lang],
            }
            if 'info' in value:
                components[key]['info'] = value['info'][lang]
