from abc import ABC
import torch
import math
from torch import nn, einsum
from functools import partial
from einops import rearrange
from einops.layers.torch import Rearrange


class PreNorm(nn.Module, ABC):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNorm2(nn.Module, ABC):
    def __init__(self, dim1, dim2, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim1)
        self.norm2 = nn.LayerNorm(dim2)
        self.fn = fn

    def forward(self, x, x_s, **kwargs):
        return self.fn(x, x_s, **kwargs)


class FeedForward(nn.Module, ABC):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module, ABC):
    def __init__(self, dim, spatial_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.spatial_transform = nn.Conv2d(spatial_dim, dim, kernel_size=1)

    def forward(self, freq_x, spatial_x):
        b, n, _, h = *freq_x.shape, self.heads
        #print('n.shape = {}'.format(n))
        #print('spatial_x.shape = {}'.format(spatial_x.shape))
        #print('freq_x.shape = {}'.format(freq_x.shape))

        # Transform spatial_x to have the same dimension as freq_x
        spatial_x = self.spatial_transform(spatial_x.unsqueeze(2))
        #print('spatial_x.shape = {}'.format(spatial_x.shape))

        spatial_x = spatial_x.squeeze(2)
        #print('spatial_x.shape = {}'.format(spatial_x.shape))

        # Compute Q from frequency domain input
        q = rearrange(self.to_q(freq_x), 'b n (h d) -> b h n d', h=h)
        #print('heads = {}'.format(h))

        spatial_x = spatial_x.permute(0, 2, 1).contiguous()
        #print('spatial_x.shape = {}'.format(spatial_x.shape))

        # Compute K and V from spatial domain input
        kv = self.to_kv(spatial_x).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), kv)
        #print('k.shape = {}'.format(k.shape))
        #print('v.shape = {}'.format(v.shape))


        # Attention mechanism
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Attention2(nn.Module, ABC):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out