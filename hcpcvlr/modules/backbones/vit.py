"""
some modules of transformer
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


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



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
