# Copyright (c) ModelScope Contributors. All rights reserved.
"""旧图升级的 one-shot 服务入口。

前端在读取 flows 后、渲染节点前调用（``POST /api/workflow/upgrade``，中间件
spawn ``python -m swift.dev.webui.upgrade``）：stdin 收一份画布声明 JSON，
stdout 回一份 GraphUpgrade.preview 的完整结果（candidate/changes/choices/
validation_error/can_confirm）。升级只在副本上展开，choices 未清零时
``can_confirm=False``，页面据此展示升级问题并禁止运行/自动保存。
"""
from __future__ import annotations

import json
import sys


def run(payload: dict) -> dict:
    from .graph import GraphUpgrade

    graph = payload.get('graph')
    if not isinstance(graph, dict) or not isinstance(graph.get('nodes'), list):
        return {'error': '升级需要画布声明（graph.nodes 数组）'}
    selections = payload.get('selections')
    if selections is None and isinstance(payload.get('selections_json'), str):
        # 声明必须可序列化往返：前端可选经 JSON 字符串透传用户选择。
        try:
            selections = json.loads(payload['selections_json'])
        except ValueError:
            return {'error': 'selections_json 不是合法 JSON'}
    try:
        return GraphUpgrade(graph).preview(selections if isinstance(selections, dict) else None)
    except (KeyError, TypeError, ValueError) as exc:
        return {'error': f'旧声明无法确定迁移：{exc}'}


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        json.dump({'error': 'stdin 不是合法 JSON'}, sys.stdout, ensure_ascii=False)
        return
    json.dump(run(payload), sys.stdout, ensure_ascii=False)


if __name__ == '__main__':
    main()
