"""
modules of transformer
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.pos_embed import get_1d_sincos_pos_embed


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class WordEmbed(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(WordEmbed, self).__init__()
        self.lut = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.embed_dim)


class TextEmbed(nn.Module):
    """
    test embedding with 1d sin-cos embedding
    """

    def __init__(self, embed_dim, vocab_size, dropout=0.):
        super(TextEmbed, self).__init__()
        self.word_embed = WordEmbed(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 5000 + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(embed_dim)
        
        self.initialize_weights()
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def initialize_weights(self):
        # initialization
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x):
        x = self.word_embed(x)
        x = x + self.pos_embed[:, 1:x.size(1)+1]
        x = self.dropout(self.norm(x))
        return x


class SublayerConnection(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadedAttention(num_heads, embed_dim, dropout)
        self.feed_forward = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.sublayer_attn = SublayerConnection(embed_dim, dropout)
        self.sublayer_ff = SublayerConnection(embed_dim, dropout)

    def forward(self, x, mask=None):
        x = self.sublayer_attn(x, lambda x: self.attn(x, x, x, mask))
        x = self.sublayer_ff(x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_layer, num_heads, ff_dim, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, ff_dim, dropout)
                                     for _ in range(num_layer)])
        self.norm = LayerNorm(embed_dim)

    def forward(self, h, mask=None):
        for layer in self.layers:
            h = layer(h, mask)
        h = self.norm(h)
        return h


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(num_heads, embed_dim, dropout)
        self.cross_attn = MultiHeadedAttention(num_heads, embed_dim, dropout)
        self.feed_forward = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.sublayer_cross = SublayerConnection(embed_dim, dropout)
        self.sublayer_self = SublayerConnection(embed_dim, dropout)
        self.sublayer_ff = SublayerConnection(embed_dim, dropout)

    def forward(self, x, h, self_mask=None, cross_mask=None):
        x = self.sublayer_self(x, lambda x: self.self_attn(x, x, x, self_mask))
        x = self.sublayer_cross(x, lambda x: self.cross_attn(x, h, h, cross_mask))
        x = self.sublayer_ff(x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_layer, num_heads, ff_dim, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, ff_dim, dropout)
                                     for _ in range(num_layer)])
        self.norm = LayerNorm(embed_dim)

    def forward(self, x, h, self_mask=None, cross_mask=None):
        for i in range(len(self.layers)):
            x = self.layers[i](x, h, self_mask, cross_mask)
        x = self.norm(x)
        return x


class Transformer(nn.Module):

    arch_settings = {
        "nano": (3, 8, 512),
        "small": (6, 8, 512),
        "base": (12, 12, 768),
        "large": (24, 16, 1024),
        "huge": (32, 16, 1280)
    }

    def __init__(self):
        super().__init__()

    def forward(self):
        pass
