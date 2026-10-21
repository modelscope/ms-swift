from __future__ import annotations
from typing import Any, Dict, Union

import torch
from twinkle.data_format import InputFeature, LossOutput, ModelOutput

__all__ = ["InputFeature", "ModelOutput", "LossOutput", "VISION_DUMMY_FLAG"]

# Marker on a collated batch: this forward carries an injected dummy image; zero the aligner.
VISION_DUMMY_FLAG = "_vision_keepalive_dummy"
