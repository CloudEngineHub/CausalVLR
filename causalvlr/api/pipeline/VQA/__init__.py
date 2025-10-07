from .tempclip_pipeline import TempCLIPPipeline
from .cra_pipeline import CRAPipeline

VQA_PIPELINES = {
    "TempCLIP": TempCLIPPipeline,
    "CRA": CRAPipeline,
    "tempclip": TempCLIPPipeline,
    "cra": CRAPipeline,
}

__all__ = [
    "TempCLIPPipeline",
    "CRAPipeline",
    "VQA_PIPELINES",
]
