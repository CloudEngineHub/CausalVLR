"""
vision language model for MRG task.
It is a baseline model based on transformer (L3H8N512).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import misc
from modules.backbones.transformer import TextEmbed, Decoder, Encoder
from modules.backbones.vision_transformer import VisEmbed
from .beam_search import BeamSearch


def get_hv_mask(hv):
    """
    bidirectional-attention mask for image
    """
    v_masks = hv.new_ones(hv.shape[:2], dtype=torch.long)
    v_masks = v_masks.unsqueeze(-2)
    return v_masks


def get_ht_mask(seq=None):
    """
    unidirectional-attention mask for text
    """
    if seq is not None:
        # crop the last one
        seq = seq[:, :-1]
        seq_mask = (seq.data > 0)
        seq_mask[:, 0] += True
        seq_mask = seq_mask.unsqueeze(-2)
        seq_mask = seq_mask & misc.subsequent_mask(seq.size(-1)).to(seq_mask)
    else:
        seq_mask = None
    return seq_mask, seq


class Baseline(nn.Module):
    def __init__(self, cfgs, tokenizer):
        super(Baseline, self).__init__()
        self.cfgs = cfgs
        self.vocab_size = len(tokenizer.idx2token)
        self.tokenizer = tokenizer

        self.en_num_layers = cfgs["en_num_layers"]
        self.de_num_layers = cfgs["de_num_layers"]
        self.embed_dim = cfgs["embed_dim"]
        self.num_heads = cfgs["num_heads"]
        self.ff_dim = cfgs["embed_dim"] * 4
        self.dropout = cfgs["dropout"]

        # ------------------------------------------
        # Embedding
        self.vis_embed = VisEmbed(embed_dim=self.embed_dim, dropout=self.dropout)
        self.text_embed = TextEmbed(embed_dim=self.embed_dim, vocab_size=self.vocab_size + 1, dropout=self.dropout)

        # Encoder
        self.encoder = Encoder(embed_dim=self.embed_dim, num_layer=self.en_num_layers, num_heads=self.num_heads,
                               ff_dim=self.ff_dim, dropout=self.dropout)
        self.decoder = Decoder(embed_dim=self.embed_dim, num_layer=self.de_num_layers,
                               num_heads=self.num_heads, ff_dim=self.ff_dim, dropout=self.dropout)

        self.logit = nn.Linear(self.embed_dim, self.vocab_size + 1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        torch.nn.init.normal_(self.cls_token, std=.02)

        self.beam_search = BeamSearch(cfgs, self.vocab_size)

        # ------------------------------------------
        if cfgs["dataset_name"] == 'iu_xray':
            self.forward = self.forward_iu_xray
        elif cfgs["dataset_name"] == 'mimic_cxr':
            self.forward = self.forward_mimic_cxr
        elif cfgs["dataset_name"] == 'ffa_ir':
            self.forward = self.forward_ffa_ir
        else:
            raise ValueError

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        B = images.size(0)
        hv = self.vis_embed(images.reshape(B*2, 3, 224, 224))
        outputs = self._forward(hv, targets, mode, B)
        return outputs

    def forward_ffa_ir(self, images, targets=None, mode='train'):
        # B, N, C, H, W
        B, N, C, H, W = images.size()
        images = images.reshape([-1, C, H, W])
        hv = self.vis_embed(images)
        outputs = self._forward(hv, targets, mode, B)
        return outputs

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        hv = self.vis_embed(images)
        outputs = self._forward(hv, targets, mode, images.size(0))
        return outputs

    def _forward(self, hv, targets, mode, B):
        # append cls token
        hv = hv.reshape([B, -1, self.embed_dim])
        cls_token = self.cls_token + self.vis_embed.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(hv.shape[0], -1, -1)
        hv = torch.cat((cls_tokens, hv), dim=1)
        # encode
        hv_mask = get_hv_mask(hv)
        hv = self.encoder(hv, hv_mask)
        # decode
        if mode == 'train':
            ht_mask, targets = get_ht_mask(targets)
            ht = self.text_embed(targets)  # [B, L] -> [B, L, D]
            out = self.decoder(ht, hv, self_mask=ht_mask, cross_mask=hv_mask)
            outputs = [F.log_softmax(self.logit(out), dim=-1)]
        elif mode == 'sample':
            self.beam_search.load_model(self.sample_forward,
                                        self.logit)
            outputs, _ = self.beam_search.sample_beam(hv)
            self.beam_search.clean_model()
        else:
            raise ValueError
        return outputs

    def sample_forward(self, hv, ht, v_mask, t_mask):
        ht = self.text_embed(ht)
        return self.decoder(ht, hv, self_mask=t_mask, cross_mask=v_mask)
