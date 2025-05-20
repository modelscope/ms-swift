# Copyright (c) Alibaba, Inc. and its affiliates.

BIN_EXTENSIONS = [
    '.*.bin',
    '.*.ts',
    '.*.pt',
    '.*.data-00000-of-00001',
    '.*.onnx',
    '.*.meta',
    '.*.pb',
    '.*.index',
]

PEFT_TYPE_KEY = 'peft_type'
SWIFT_TYPE_KEY = 'swift_type'
DEFAULT_ADAPTER = 'default'

DEFAULT_SYSTEM = ('A conversation between User and Assistant. The user asks a question, and the Assistant solves it. '
                  'The assistant first thinks about the reasoning process in the mind and then provides the user '
                  'with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> '
                  '</answer> tags, respectively, i.e., <think> reasoning process here </think>'
                  '<answer> answer here </answer>')


class Invoke(object):
    KEY = 'invoked_by'
    THIRD_PARTY = 'third_party'
    PRETRAINED = 'from_pretrained'
    PIPELINE = 'pipeline'
    TRAINER = 'trainer'
    LOCAL_TRAINER = 'local_trainer'
    PREPROCESSOR = 'preprocessor'
    SWIFT = 'swift'
