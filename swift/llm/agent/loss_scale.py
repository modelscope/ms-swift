# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import json

from .utils import calculate_loss_scale


def agentflan_loss_scale(query: str, response: str):
    loss_scale_config_path = 'agentflan.json'
    path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(path, '../agent', loss_scale_config_path)
    with open(config_path, 'r') as json_file:
        loss_scale_map = json.load(json_file)
    query_loss_scale_map = loss_scale_map['query']
    response_loss_scale_map = loss_scale_map['response']
    return calculate_loss_scale(query, response, response_loss_scale_map, query_loss_scale_map)


def react_loss_scale(query: str, response: str):
    loss_scale_config_path = 'default_loss_scale_config.json'
    path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(path, '../agent', loss_scale_config_path)
    with open(config_path, 'r') as json_file:
        loss_scale_map = json.load(json_file)
    return calculate_loss_scale(query, response, loss_scale_map)


def alpha_umi_loss_scale(query: str, response: str):
    loss_scale_config_path = 'alpha_umi_loss_scale_config.json'
    path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(path, '../agent', loss_scale_config_path)
    with open(config_path, 'r') as json_file:
        loss_scale_map = json.load(json_file)
    return calculate_loss_scale(query, response, loss_scale_map)


def default_loss_scale(query: str, response: str):
    return [response], [1.0]


loss_scale_map = {
    'agentflan': agentflan_loss_scale,
    'react': react_loss_scale,
    'alpha_umi': alpha_umi_loss_scale,
    'default': default_loss_scale,
}
