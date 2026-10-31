# Copyright (c) ModelScope Contributors. All rights reserved.
"""画布与执行器共享的类型、端口及生命周期声明。"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

LIFECYCLES = ('Resource', 'Iterator', 'Stage', 'Task')


@dataclass(frozen=True)
class PortSpec:
    key: str
    kind: str
    role: str
    profiles: tuple[str, ...] = ()
    many: bool = False
    optional: bool = False
    #: 仅列出消费者真正接受的原生通道，不表示业务上的操作关系。
    accepted_kinds: tuple[str, ...] = ()
    access: str = 'read'
    #: 控制句柄不提取 Iterator 的下一批数据。
    control: bool = False
    #: 作用域可满足：未接线时沿非反馈边从上游 Resource 节点解析唯一同 kind
    #: 变量（连通即共享）；多个候选是歧义，必须显式连线，不运行期猜测。
    scope: bool = False
    enabled_by: str = ''
    #: 从输入端口继承实际 kind/profile；many 输入逐 slot 生成输出。
    mirror_input: str = ''
    profile_param: str = ''

    def accepts(self, producer_profiles: tuple[str, ...], producer_kind: str = '') -> bool:
        if producer_kind and producer_kind not in (self.kind, *self.accepted_kinds):
            return False
        return not self.profiles or bool(producer_profiles) and all(
            profile in self.profiles for profile in producer_profiles)

    def to_dict(self) -> dict:
        value = asdict(self)
        for key in ('profiles', 'accepted_kinds'):
            value[key] = list(value[key])
        return value


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    default: Any
    doc: str = ''
    options: tuple[Any, ...] = ()
    group: str = ''
    type: str = 'str'
    required: bool = False
    nullable: bool = False
    minimum: float | None = None
    maximum: float | None = None
    phase: str = 'construct'
    secret: bool = False
    item_type: str = ''
    item_options: tuple[Any, ...] = ()
    unique_items: bool = False

    def validate(self, value: Any) -> Any:
        if value is None:
            if self.nullable:
                return None
            raise ValueError(f'{self.name}: 不允许 null')
        types = {'str': str, 'code': str, 'bool': bool, 'int': int,
                 'float': (int, float), 'list': list, 'dict': dict, 'json': (dict, list)}
        expected = types.get(self.type)
        if expected is None:
            raise ValueError(f'{self.name}: 未声明的参数类型 {self.type!r}')
        if not isinstance(value, expected) or self.type in ('int', 'float') and isinstance(value, bool):
            raise ValueError(f'{self.name}: 需要 {self.type}，实际为 {type(value).__name__}')
        if self.required and isinstance(value, (str, list, dict)) and not value:
            raise ValueError(f'{self.name}: 必填参数不能为空')
        if self.options and value not in self.options:
            raise ValueError(f'{self.name}: 需要 {self.options!r}')
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f'{self.name}: 需要有限数值')
            if self.minimum is not None and value < self.minimum:
                raise ValueError(f'{self.name}: 不得小于 {self.minimum}')
            if self.maximum is not None and value > self.maximum:
                raise ValueError(f'{self.name}: 不得大于 {self.maximum}')
        if isinstance(value, list):
            for index, item in enumerate(value):
                if self.item_type:
                    ParamSpec(f'{self.name}[{index}]', self.label, None, type=self.item_type,
                              options=self.item_options).validate(item)
                elif self.item_options and item not in self.item_options:
                    raise ValueError(f'{self.name}[{index}]: 未声明选项 {item!r}')
            if self.unique_items and any(item in value[:index] for index, item in enumerate(value)):
                raise ValueError(f'{self.name}: 元素不得重复')
        self._validate_json(value)
        return copy.deepcopy(value)

    def _validate_json(self, value: Any, depth: int = 0) -> None:
        if depth > 32:
            raise ValueError(f'{self.name}: 嵌套深度超过 32')
        if value is None or type(value) in (str, bool, int):
            return
        if type(value) is float and math.isfinite(value):
            return
        if type(value) is list:
            for item in value:
                self._validate_json(item, depth + 1)
            return
        if type(value) is dict and all(type(key) is str for key in value):
            for item in value.values():
                self._validate_json(item, depth + 1)
            return
        raise ValueError(f'{self.name}: 需要有限数值与字符串键组成的 JSON 原生值')

    def to_dict(self) -> dict:
        result = asdict(self)
        result['options'] = list(self.options)
        if self.secret:
            result['default'] = None
        return result


@dataclass(frozen=True)
class ComponentDescriptor:
    type: str
    label: str
    lifecycle: str
    math_doc: str
    ports_in: tuple[PortSpec, ...] = field(default=())
    ports_out: tuple[PortSpec, ...] = field(default=())
    params: tuple[ParamSpec, ...] = field(default=())
    factory: str = ''
    client_factory: str = ''
    effects: tuple[str, ...] = field(default=())
    #: 每组必须恰好接入一个口。
    exclusive_inputs: tuple[tuple[str, ...], ...] = ()
    #: 参数等于指定值时，这些输入必需。
    required_when: tuple[tuple[str, Any, tuple[str, ...]], ...] = ()
    owner_kind: str = ''
    #: 计算执行位：driver（session 进程）或 remote（数值 worker，重载荷以
    #: 引用进出，session 只持句柄与小控制标量）。
    execution: str = 'driver'

    def __post_init__(self) -> None:
        if self.lifecycle not in LIFECYCLES:
            raise ValueError(f'{self.type}: 非法生命周期 {self.lifecycle!r}')
        if self.execution not in ('driver', 'remote'):
            raise ValueError(f'{self.type}: 非法执行位 {self.execution!r}')
        for key in ('ports_in', 'ports_out', 'params', 'effects', 'exclusive_inputs', 'required_when'):
            if not isinstance(getattr(self, key), tuple):
                raise TypeError(f'{self.type}.{key}: 需要 tuple')
        for ports in (self.ports_in, self.ports_out):
            if len({port.key for port in ports}) != len(ports):
                raise ValueError(f'{self.type}: 同一方向端口名重复')
        if len({param.name for param in self.params}) != len(self.params):
            raise ValueError(f'{self.type}: 参数名重复')

    def port(self, key: str, direction: str = '') -> PortSpec:
        if direction not in ('', 'in', 'out'):
            raise ValueError(f'未知端口方向 {direction!r}')
        ports = self.ports_in if direction == 'in' else self.ports_out if direction == 'out' else (
            self.ports_in + self.ports_out)
        candidates = [port for port in ports if port.key == key]
        if len(candidates) != 1:
            raise KeyError(f'{self.type}.{key}: 请指定存在且唯一的 {direction or "有方向"} 端口')
        return candidates[0]

    def required_params(self) -> tuple[ParamSpec, ...]:
        return tuple(param for param in self.params if param.required)

    def validate_params(self, values: Mapping[str, Any]) -> dict[str, Any]:
        declared = {param.name: param for param in self.params}
        unknown = set(values) - set(declared)
        if unknown:
            raise ValueError(f'{self.type}: 未声明参数 {sorted(unknown)}')
        return {name: param.validate(values[name] if name in values else param.default)
                for name, param in declared.items()}

    def output_ports(self, values: Mapping[str, Any]) -> tuple[PortSpec, ...]:
        return tuple(port for port in self.ports_out if not port.enabled_by
                     or port.key in values.get(port.enabled_by, ()))

    def to_dict(self) -> dict:
        return {'type': self.type, 'label': self.label, 'lifecycle': self.lifecycle,
                'math_doc': self.math_doc, 'ports_in': [port.to_dict() for port in self.ports_in],
                'ports_out': [port.to_dict() for port in self.ports_out],
                'params': [param.to_dict() for param in self.params], 'factory': self.factory,
                'client_factory': self.client_factory, 'effects': list(self.effects),
                'exclusive_inputs': [list(group) for group in self.exclusive_inputs],
                'required_when': [[key, value, list(ports)] for key, value, ports in self.required_when],
                'owner_kind': self.owner_kind, 'execution': self.execution}
