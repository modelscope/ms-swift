"""Process-wide runtime options shared by non-training CLI commands."""
from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    """Side-effectful run settings that do not belong to a business recipe."""

    seed: int = 42
