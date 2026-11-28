# Copyright (c) ModelScope Contributors. All rights reserved.


def validate_infonce_config(config):
    if not isinstance(config, dict) or set(config) - {'use_in_batch_negatives', 'hard_negatives'}:
        raise ValueError('infonce_config accepts use_in_batch_negatives and hard_negatives only.')
    if 'use_in_batch_negatives' in config and type(config['use_in_batch_negatives']) is not bool:
        raise ValueError('use_in_batch_negatives must be a boolean.')
    negatives = config.get('hard_negatives')
    if negatives is not None and (type(negatives) is not int or negatives < 0):
        raise ValueError('hard_negatives must be a nonnegative integer or null.')
    if config.get('use_in_batch_negatives') is False and negatives == 0:
        raise ValueError('Disabling both in-batch and explicit negatives produces no training signal.')
    return config.copy()
