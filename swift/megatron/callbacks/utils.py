# Copyright (c) ModelScope Contributors. All rights reserved.


def rewrite_logs(logs):
    new_logs = {}
    for k, v in logs.items():
        if isinstance(v, str):
            continue
        k = k.replace('/', '_')
        if k.startswith('eval_'):
            k = k[len('eval_'):]
            k = f'eval/{k}'
        elif k.startswith('test_'):
            k = k[len('test_'):]
            k = f'test/{k}'
        else:
            k = f'train/{k}'
        new_logs[k] = v
    return new_logs
