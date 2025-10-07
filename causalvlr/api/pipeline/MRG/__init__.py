from typing import Dict, Type

from .mrg_pipeline import MRGPipeline

MRG_PIPELINES: Dict[str, Type] = {
    'MRG': MRGPipeline,
}

__all__ = [
    "MRG_PIPELINES",
    "MRGPipeline",
]
