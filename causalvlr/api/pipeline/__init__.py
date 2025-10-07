from typing import Dict, Type

from .VQA import VQA_PIPELINES
from .MRG import MRG_PIPELINES

ALL_PIPELINES: Dict[str, Type] = {
    **{f"mrg_{k}": v for k, v in MRG_PIPELINES.items()},
    **{f"vqa_{k}": v for k, v in VQA_PIPELINES.items()}
}

from . import VQA
from . import MRG

__all__ = [
    "MRG_PIPELINES",
    "VQA_PIPELINES",
    "ALL_PIPELINES",
    "MRG",
    "VQA",
]
