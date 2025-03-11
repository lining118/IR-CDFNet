import torch
import torch.nn as nn
from utils.utils import Seg
from einops import rearrange
from config import Config
from utils.utils2 import PreNorm, Attention, Attention2, FeedForward, PreNorm2
import torch_dct as DCT
from models.modules.attention_blocks import SpatialAttention
import torch.nn.functional as F
import os



class channel_shuffle(nn.Module):
    def __init__(self,groups=4):
        super(channel_shuffle,self).__init__()
        self.groups=groups

    def forward(self,x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batchsize, self.groups,
               channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)
        return x


class FSF(nn.Module):
    def __init__(self, spatial_dim):
        super(FSF, self).__init__()
        self.config = Config()
        self.seg = Seg().to(self.config.device)
        self.vector_y = nn.Parameter(scaled_tensor1, requires_grad=True).to(self.config.device)
        self.vector_cb = nn.Parameter(scaled_tensor2, requires_grad=True).to(self.config.device)
        self.vector_cr = nn.Parameter(scaled_tensor3, requires_grad=True).to(self.config.device)
        self.shuffle = channel_shuffle()
        self.high_band = Transformer(dim=256, spatial_dim=spatial_dim, depth=1, heads=2, dim_head=128, mlp_dim=128 * 2, dropout=0)
        self.low_band = Transformer(dim=256, spatial_dim=spatial_dim, depth=1, heads=2, dim_head=128, mlp_dim=128 * 2, dropout=0)
        self.band = Transformer(dim=256, spatial_dim=spatial_dim, depth=1, heads=2, dim_head=128, mlp_dim=128 * 2, dropout=0)
        self.spatial = Transformer(dim=192, spatial_dim=spatial_dim, depth=1, heads=2, dim_head=64, mlp_dim=64 * 2, dropout=0)
        self.attn = SpatialAttention(96)
        # High-level decoder
        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )

        self.low_freq_conv = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=5, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
    def forward(self, x_dct, spatial_x):
        # 进行DCT预处理
        num_batchsize = x_dct.shape[0]
        size = x_dct.shape[2]
        x_dct = x_dct.reshape(num_batchsize, 3, size // 8, 8, size // 8, 8).permute(0, 2, 4, 1, 3, 5)
        x_dct = DCT.dct_2d(x_dct, norm='ortho')
        x_dct = x_dct.reshape(num_batchsize, size // 8, size // 8, -1).permute(0, 3, 1, 2)

        feat_y = x_dct[:, 0:64, :, :] * (self.seg + self.vector_y)
        feat_Cb = x_dct[:, 64:128, :, :] * (self.seg + self.vector_cb)
        feat_Cr = x_dct[:, 128:192, :, :] * (self.seg + self.vector_cr)

        origin_feat_DCT = torch.cat((torch.cat((feat_y, feat_Cb), 1), feat_Cr), 1)
        origin_feat_DCT = self.shuffle(origin_feat_DCT)

        high = torch.cat([feat_y[:, 32:, :, :], feat_Cb[:, 32:, :, :], feat_Cr[:, 32:, :, :]], 1)
        low = torch.cat([feat_y[:, :32, :, :], feat_Cb[:, :32, :, :], feat_Cr[:, :32, :, :]], 1)

        save_path = "high_low_dct.pth"
        if not os.path.exists(save_path):
            torch.save({
                'high_dct': high,
                'low_dct': low,
                'full_ycbcr_dct': x_dct  # 保存完整的 YCbCr DCT 变换结果
            }, save_path)

        b, n, h, w = high.shape
        high = torch.nn.functional.interpolate(high, size=(16, 16))
        low = torch.nn.functional.interpolate(low, size=(16, 16))

        #print('high.shape = {}'.format(high.shape))
        #print('low.shape = {}'.format(low.shape))
        # Apply convolution to high and low frequencies
        high = self.high_freq_conv(high)
        low = self.low_freq_conv(low)
        high = self.attn(high)

        high = rearrange(high, 'b n h w -> b n (h w)')
        low = rearrange(low, 'b n h w -> b n (h w)')

        spatial_x = rearrange(spatial_x, 'b n h w -> b n (h w)')

        #print('spatial_x.shape = {}'.format(spatial_x.shape))
        high = self.high_band(high, spatial_x)
        low = self.low_band(low, spatial_x)

        y_h, b_h, r_h = torch.split(high, 32, 1)
        y_l, b_l, r_l = torch.split(low, 32, 1)
        feat_y = torch.cat([y_l, y_h], 1)
        feat_Cb = torch.cat([b_l, b_h], 1)
        feat_Cr = torch.cat([r_l, r_h], 1)
        feat_DCT = torch.cat((torch.cat((feat_y, feat_Cb), 1), feat_Cr), 1)
        feat_DCT = self.band(feat_DCT, spatial_x)
        feat_DCT = feat_DCT.transpose(1, 2)
        feat_DCT = self.spatial(feat_DCT, spatial_x)
        feat_DCT = feat_DCT.transpose(1, 2)
        feat_DCT = rearrange(feat_DCT, 'b n (h w) -> b n h w', h=16)
        feat_DCT = torch.nn.functional.interpolate(feat_DCT, size=(h, w))
        feat_DCT = origin_feat_DCT + feat_DCT
        return feat_DCT



# 高频信息处理- 空间注意力
class Transformer(nn.Module):
    def __init__(self, dim, spatial_dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm2(dim, spatial_dim, Attention(dim, spatial_dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, freq_x, spatial_x):
        for attn, ff in self.layers:
            freq_x = attn(freq_x, spatial_x) + freq_x
            freq_x = ff(freq_x) + freq_x
        freq_x = self.norm(freq_x)
        return freq_x




# 生成随机张量
random_tensor1 = torch.rand(1, 64, 1, 1)

# 将范围从 [0, 1) 转换为 [-1, 1)
scaled_tensor1 = 2 * random_tensor1 - 1

random_tensor2 = torch.rand(1, 64, 1, 1)

# 将范围从 [0, 1) 转换为 [-1, 1)
scaled_tensor2 = 2 * random_tensor2 - 1

random_tensor3 = torch.rand(1, 64, 1, 1)

# 将范围从 [0, 1) 转换为 [-1, 1)
scaled_tensor3 = 2 * random_tensor3 - 1
