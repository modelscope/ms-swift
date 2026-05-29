import collections
from dataclasses import FrozenInstanceError, dataclass, field, fields
from typing import Any, Optional

# BaseConfig class inherits from collections.abc.Mapping, which means it can act like a dictionary


@dataclass
class BaseConfig(collections.abc.Mapping):
    """The BaseConfig provides dict-like interface for a dataclass config.

    By default all fields in the config is not mutable, unless specified in
    "_mutable_fields". The BaseConfig class implements the Mapping Abstract Base Class.
    This allows instances of this class to be used like dictionaries.
    """

    _mutable_fields = set()
    _target_: str = ''

    def __setattr__(self, name: str, value):
        """Set the value of an attribute. Check if the attr is mutable before setting the value."""
        # If the field already exists, it's considered frozen unless it's in _mutable_fields
        if name in self.__dict__ and name not in getattr(self, '_mutable_fields', set()):
            raise FrozenInstanceError(f"Field '{name}' is frozen and cannot be modified")
        super().__setattr__(name, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get the value associated with the given key. If the key does not exist, return the default value.

        Args:
            key (str): The attribute name to retrieve.
            default (Any, optional): The value to return if the attribute does not exist. Defaults to None.

        Returns:
            Any: The value of the attribute or the default value.
        """
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def __getitem__(self, key: str):
        """Implement the [] operator for the class. Allows accessing attributes like dictionary items.

        Args:
            key (str): The attribute name to retrieve.

        Returns:
            Any: The value of the attribute.

        Raises:
            AttributeError: If the attribute does not exist.
            TypeError: If the key type is not string
        """
        return getattr(self, key)

    def __iter__(self):
        """Implement the iterator protocol. Allows iterating over the attribute names of the instance.

        Yields:
            str: The name of each field in the dataclass.
        """
        for f in fields(self):
            yield f.name

    def __len__(self):
        """
        Return the number of fields in the dataclass.

        Returns:
            int: The number of fields in the dataclass.
        """
        return len(fields(self))


@dataclass
class ProfilerConfig(BaseConfig):
    """Worker profiler config.

    Args:
        discrete (bool): True for each task has its own database, False for all tasks in one training step
          share one database.
        all_ranks (bool): Whether to profile all ranks.
        ranks (list[int]): The ranks that will be profiled. Defaults to [].
        global_tool_config (Any): Global tool configuration for all profiling tools.
    """

    tool: Optional[str] = None
    enable: bool = False
    all_ranks: bool = False
    ranks: list[int] = field(default_factory=list)
    save_path: Optional[str] = None
    tool_config: Any = None
    global_tool_config: Optional[Any] = None  # Global tool configuration for all profiling tools

    def union(self, other: 'ProfilerConfig') -> 'ProfilerConfig':
        assert self.tool == other.tool, f"Cannot union ProfilerConfig with different tools: {self.tool} vs {other.tool}"
        return ProfilerConfig(
            tool=self.tool,
            enable=self.enable or other.enable,
            all_ranks=self.all_ranks or other.all_ranks,
            ranks=list(set(self.ranks or []) | set(other.ranks or [])),
            save_path=self.save_path or other.save_path,
            tool_config=self.tool_config or other.tool_config,
            global_tool_config=self.global_tool_config or other.global_tool_config,
        )

    def intersect(self, other: 'ProfilerConfig') -> 'ProfilerConfig':
        assert self.tool == other.tool, (
            f"Cannot intersect ProfilerConfig with different tools: {self.tool} vs {other.tool}")
        return ProfilerConfig(
            tool=self.tool,
            enable=self.enable and other.enable,
            all_ranks=self.all_ranks and other.all_ranks,
            ranks=list(set(self.ranks or []) & set(other.ranks or [])),
            save_path=self.save_path,
            tool_config=self.tool_config,
            global_tool_config=self.global_tool_config if self.global_tool_config else other.global_tool_config,
        )

    def __post_init__(self) -> None:
        """config validation logics go here"""
        assert isinstance(self.ranks,
                          (set, list, tuple)), (f"Profiler ranks must be of type list, got {type(self.ranks)}")


@dataclass
class TorchProfilerToolConfig(BaseConfig):
    """Torch profiler tool config."""

    # options: cuda, cpu, memory, shapes, stack
    contents: list[str] = field(default_factory=list)
    discrete: bool = False
    name: str = 'torch'

    def __post_init__(self) -> None:
        """config validation logics go here"""
        assert isinstance(self.contents, list), f"Profiler contents must be of type list, got {type(self.contents)}"
        __support_contents = ['cuda', 'cpu', 'memory', 'shapes', 'stack']
        for content in self.contents:
            assert content in __support_contents, (
                f"Profiler contents only supports {__support_contents}, but gets {content}")
