from .mrg_loss import compute_lm_loss, compute_recon_loss, patchify
from .mrg_optimizers import build_optimizer, build_lr_scheduler
from .mrg_monitor import Monitor
from . import mrg_cvt_im_tensor as cvt_im_tensor
from . import mrg_tensor_utils as tensor_utils

loss_fn = {'lm': compute_lm_loss, 'recon': compute_recon_loss}

def _get_tokenizers_fn():
    """Lazy import to avoid circular dependency"""
    from causalvlr.data.MRG.mrg_tokenizers import Tokenizer, MixTokenizer
    return {'ori': Tokenizer, 'mix': MixTokenizer}

# Create a property-like getter
class _TokenizersFn:
    def __getitem__(self, key):
        return _get_tokenizers_fn()[key]
    
    def __repr__(self):
        return str(_get_tokenizers_fn())

tokenizers_fn = _TokenizersFn()

__all__ = [
    'loss_fn',
    'tokenizers_fn',
    'patchify',
    'Monitor',
    'cvt_im_tensor',
    'tensor_utils',
    'build_optimizer',
    'build_lr_scheduler',
]
