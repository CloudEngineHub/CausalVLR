from typing import Dict, Type
import torch.nn as nn

from .MRG import model_fn as MRG_MODELS

from .VQA import model_fns as VQA_MODELS

# Import submodules for direct access
from . import MRG
from . import VQA

# Unified model registry with task prefixes
ALL_MODELS: Dict[str, Type[nn.Module]] = {
    **{f"mrg_{k}": v for k, v in MRG_MODELS.items()},
    **{f"vqa_{k}": v for k, v in VQA_MODELS.items()}
}

__all__ = [
    "MRG_MODELS",
    "VQA_MODELS",
    "ALL_MODELS",
    "MRG",
    "VQA",
]
