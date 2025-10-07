from typing import Dict, Type

from .mrg import Trainer, PTrainer, FTrainer

TRAINERS: Dict[str, Type] = {
    'mrg_base': Trainer,
    'mrg_pretrain': PTrainer,
    'mrg_finetune': FTrainer,
}

from . import mrg

__all__ = [
    "Trainer",
    "PTrainer",
    "FTrainer",
    "TRAINERS",
    "mrg",
]
