import torch
from torch import nn
from torch.autograd import forward_ad
from modules.backbones.transformer import MultiHeadedAttention, PositionwiseFeedForward, LayerNorm
from torchvision.models import resnet101
from ..utils.pos_embed import get_2d_sincos_pos_embed


class PatchEmbed(nn.Module):
    """
    TODO add more selections for patch embedding
    resnet 1-3 block stem
    """

    def __init__(self, img_size=224, patch_size=16):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        model = resnet101(True)
        modules = list(model.children())[:-3]
        self.embed = nn.Sequential(*modules)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.embed(x).flatten(2).transpose(1, 2)
        return x


class VisEmbed(nn.Module):
    """
    image embedding with 2d sin-cos position embedding
    """
    def __init__(self, img_size=224, patch_size=16, embed_dim=512, dropout=0., cls_token=True):
        super(VisEmbed, self).__init__()

        # --------------------------------------------------------------------------
        # Patchfy and Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size)
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(1024, embed_dim)
        self.norm = LayerNorm(embed_dim)
        num_patches = self.patch_embed.num_patches
        # use 2d pos embed
        self.cls_token = cls_token
        if cls_token:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        # self.norm = norm_layer(embed_dim)
        self.initialize_weights()

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initialize_weights(self):
        # initialization
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=self.cls_token)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x):
        x = self.patch_embed(x)
        if self.cls_token:
            x = self.proj(x) + self.pos_embed[:, 1:, :]
        else:
            x = self.proj(x) + self.pos_embed
        x = self.dropout(self.norm(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def _init_weight(self):
        pass

    def forward(self):
        pass
