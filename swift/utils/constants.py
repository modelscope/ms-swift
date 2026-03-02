# Copyright (c) ModelScope Contributors. All rights reserved.

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


class Invoke(object):
    KEY = 'invoked_by'
    THIRD_PARTY = 'third_party'
    PRETRAINED = 'from_pretrained'
    PIPELINE = 'pipeline'
    TRAINER = 'trainer'
    LOCAL_TRAINER = 'local_trainer'
    PREPROCESSOR = 'preprocessor'
    SWIFT = 'swift'
