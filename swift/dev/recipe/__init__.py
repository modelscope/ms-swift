# Copyright (c) ModelScope Contributors. All rights reserved.
# The recipes: each subpackage is one visible, cookbook-style flow -- the training families, plus the
# non-training entries under infer / export (each binds a config in __init__ and runs in __call__,
# see base.Recipe). artifacts.py holds the typed outputs those entries hand downstream.
from __future__ import annotations
