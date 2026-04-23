from __future__ import annotations

import os
import random

import numpy as np

from src.config import SEED


def set_global_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
