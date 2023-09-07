"""
Visual Causal Intervention for Radiology Report Generation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.backbones.transformer import Encoder, DecoderLayer, LayerNorm
from .baseline import Baseline, get_ht_mask, get_hv_mask
from . import misc
from modules.layers.vlci import LocalSample, GlobalSample, VDM, LDM


def get_cross_mask(hv, seq):
    seq = seq[:, :-1]
    seq_mask = (seq.data > 0)
    seq_mask[:, 0] += True
    seq_mask = seq_mask.unsqueeze(-1)
    v_mask = hv.new_ones(hv.shape[:2], dtype=torch.long)
    v_mask = v_mask.unsqueeze(-2)

    cross_mask = seq_mask & v_mask
    return cross_mask


class CausalEncoder(Encoder):
    def __init__(self, embed_dim, num_layer, num_heads, ff_dim, dropout, mode='n'):
        super(CausalEncoder, self).__init__(embed_dim, num_layer, num_heads, ff_dim, dropout)
        self.mode = mode
        self.do = VDM(embed_dim, num_heads, ff_dim, dropout)

        self.global_sample = GlobalSample(embed_dim, num_heads, ff_dim, dropout)
        self.local_sample = LocalSample(embed_dim, num_heads, ff_dim, dropout)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, h, mask=None, pos=None, k=6, proj=False):
        attn = []
        for layer in self.layers:
            h = layer(h, mask)
            attn.append(layer.attn.attn)

        if h.size(1) > 197:
            h = h + torch.cat([pos, pos[:, 1:, :]], dim=1)
        else:
            h = h + pos

        fl = self.local_sample(h, attn, k)
        fg = self.global_sample(h)
        mediator = {"local": fl, "global": fg, 'attn': attn}
        h = self.do(h, fl=fl, fg=fg, mode=self.mode, proj=proj)

        h = self.norm(h)
        return h, mediator


class CausalDecoder(nn.Module):
    def __init__(self, embed_dim, num_layer, num_heads, ff_dim, dropout, mode='n'):
        super(CausalDecoder, self).__init__()
        self.mode = mode
        self.norm = LayerNorm(embed_dim)
        self.do = LDM(embed_dim)
        self.layers = nn.ModuleList(
            [DecoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layer)])
        self.ht = None

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, output, h, self_mask=None, cross_mask=None, z=None, fl=None, proj=False):

        output = self.do(output, z, fl, self.mode, proj)
        self.ht = output
        for i in range(len(self.layers)):
            output = self.layers[i](output, h, self_mask, cross_mask)
            # attn.append(self.layers[i].cross_attn.attn)

        output = self.norm(output)
        return output


class VLCI(Baseline):
    def __init__(self, cfgs, tokenizer):
        super(VLCI, self).__init__(cfgs, tokenizer)
        # Encoder
        self.z = None
        self.fl = None
        self.vocab = torch.arange(0, self.vocab_size + 1).unsqueeze(-1).long().cuda()
        self.z_norm = LayerNorm(self.embed_dim)
        self.fl_norm = LayerNorm(self.embed_dim)

        self.encoder = CausalEncoder(embed_dim=self.embed_dim, num_layer=self.en_num_layers, num_heads=self.num_heads,
                                     ff_dim=self.ff_dim, dropout=self.dropout, mode=cfgs["v_causal"])

        self.decoder = CausalDecoder(embed_dim=self.embed_dim, num_layer=self.de_num_layers, num_heads=self.num_heads,
                                     ff_dim=self.ff_dim, dropout=self.dropout, mode=cfgs["l_causal"])

    def _forward(self, hv, targets, mode, B):
        # append cls token
        hv = hv.reshape([B, -1, self.embed_dim])
        cls_token = self.cls_token + self.vis_embed.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(hv.shape[0], -1, -1)
        hv = torch.cat((cls_tokens, hv), dim=1)
        # encode
        hv_mask = get_hv_mask(hv)
        hv, mediator = self.encoder(hv, hv_mask, self.vis_embed.pos_embed, k=self.cfgs['local_sample'], proj=self.cfgs['v_causal_proj'])

        if self.cfgs["l_causal"]:
            z = self.text_embed.word_embed(self.vocab).reshape(1, -1, hv.size(-1)).detach()
            z = self.z_norm(z)
            self.fl = self.fl_norm(mediator['local'])
            self.z = z.repeat(hv.size(0), 1, 1)

        # decode
        if mode == 'train':
            cross_mask = get_cross_mask(hv, targets)
            ht_mask, targets = get_ht_mask(targets)
            ht = self.text_embed(targets)  # [B, L] -> [B, L, D]
            out = self.decoder(ht, hv, self_mask=ht_mask, cross_mask=cross_mask,
                               z=self.z, fl=self.fl, proj=self.cfgs['l_causal_proj'])
            outputs = F.log_softmax(self.logit(out), dim=-1)
            outputs = [outputs, mediator, {'ht': self.decoder.ht.detach(), 'hv': hv.detach()}]
        elif mode == 'sample':
            self.beam_search.load_model(self.sample_forward, self.logit)
            outputs, _ = self.beam_search.sample_beam(hv)
            self.beam_search.clean_model()
        else:
            raise ValueError
        return outputs

    def sample_forward(self, hv, ht, v_mask, t_mask):
        ht = self.text_embed(ht)
        v_mask = get_hv_mask(hv)
        if self.z is not None and self.z.size(0) != ht.size(0):
            self.z = misc.repeat_tensors(self.cfgs["beam_size"], self.z)
            self.fl = misc.repeat_tensors(self.cfgs["beam_size"], self.fl)
        out = self.decoder(ht, hv, self_mask=t_mask, cross_mask=v_mask,
                           z=self.z, fl=self.fl, proj=self.cfgs['l_causal_proj'])
        return out
