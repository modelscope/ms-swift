from dataclasses import fields
from swift.llm import SftArguments


all_choices = {}
all_default_values = {}


def get_choices(name):
    global all_choices
    if not all_choices:
        for f in fields(SftArguments):
            if 'choices' in f.metadata:
                all_choices[f.name] = f.metadata['choices']
    return all_choices.get(name, [])
