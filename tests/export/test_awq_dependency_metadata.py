# Copyright (c) ModelScope Contributors. All rights reserved.
import subprocess
import sys
from importlib.metadata import PathDistribution
from packaging.requirements import Requirement
from pathlib import Path


def test_awq_backend_is_an_opt_in_dependency(tmp_path):
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, 'setup.py', 'egg_info', '--egg-base',
                    str(tmp_path)],
                   cwd=root,
                   check=True,
                   capture_output=True,
                   text=True)
    metadata = PathDistribution(next(tmp_path.glob('*.egg-info')))
    requirements = [Requirement(item) for item in metadata.requires or []]
    gptqmodel = [item for item in requirements if item.name.lower() == 'gptqmodel']
    assert len(gptqmodel) == 1
    assert str(gptqmodel[0].specifier) == '>=7.3.2'
    assert gptqmodel[0].marker.evaluate({'extra': 'awq'})
    assert not gptqmodel[0].marker.evaluate({'extra': ''})
    assert not gptqmodel[0].marker.evaluate({'extra': 'all'})
