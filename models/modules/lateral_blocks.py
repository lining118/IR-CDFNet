import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
from config import Config

config = Config()


class two_ConvBnRule(nn.Module):

    def __init__(self, in_chan, out_chan):
        super(two_ConvBnRule, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN1 = nn.BatchNorm2d(out_chan)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN2 = nn.BatchNorm2d(out_chan)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, mid=False):
        feat = self.conv1(x)
        feat = self.BN1(feat)
        feat = self.relu1(feat)

        if mid:
            feat_mid = feat

        feat = self.conv2(feat)
        feat = self.BN2(feat)
        feat = self.relu2(feat)

        if mid:
            return feat, feat_mid
        return feat


class GussianLatBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, index=0):
        super(GussianLatBlk, self).__init__()
        self.conv = nn.Conv2d(1, out_channels, 1, 1, 0)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.index = index

    def forward(self, x):


        dog = difference_of_gaussians(x = x, kernel_size=3, sigma1=3.0 * (0.9 ** self.index), sigma2=3.0 * (0.9**(self.index + 1)), index=self.index)
        dog = self.conv(dog)
        dog = 2 * dog.sigmoid()
        return dog

def difference_of_gaussians(x, kernel_size, sigma1, sigma2, index):
    blur1 = gaussian_conv(x, kernel_size=kernel_size, sigma=sigma1, id=index)
    blur2 = gaussian_conv(x, kernel_size=kernel_size, sigma=sigma2, id=index)
    dog = blur1 - blur2
    return dog


def gaussian_conv(x, kernel_size, sigma, id):
    channels = config.channels_list[id]
    if config.gus_ker_type == '2d':
        gaussian_kernel = _get_2d_kernel(kernel_size, sigma)
        gaussian_kernel = np.float32(gaussian_kernel)
        gaussian_kernel = np.repeat(gaussian_kernel[np.newaxis, np.newaxis, ...], channels, axis=1)
        gaussian_kernel = torch.from_numpy(gaussian_kernel)

        gaussian_kernel = gaussian_kernel.to(x.device)

        x_smoothed = F.conv2d(x, gaussian_kernel, padding=kernel_size // 2, groups=1)
        x_smoothed = min_max_norm(x_smoothed)  # normalization
        return x_smoothed
    else:
        gaussian_kernel = _get_3d_kernel(sigma, kernel_size, channels).to(dtype=x.dtype, device=x.device)
        x_smoothed = F.conv2d(x, gaussian_kernel, padding=kernel_size // 2, groups=1)
        x_smoothed = min_max_norm(x_smoothed)  # normalization
        return x_smoothed




def _get_2d_kernel(size, sigma):
    interval = (2 * sigma + 1.) / size
    x = np.linspace(-sigma - interval / 2., sigma + interval / 2., size + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def _get_3d_kernel(sigma, size, channels):
    """
    Generate a 3D Gaussian kernel.

    Args:
    - sigma: standard deviation of the Gaussian distribution.
    - size: size of the kernel in each spatial dimension (should be odd).
    - channels: number of channels.

    Returns:
    - kernel: 3D Gaussian kernel.
    """
    kernel = np.zeros((channels, size, size))
    center = size // 2
    z_center = channels // 2
    cov = np.diag([sigma ** 2, sigma ** 2, sigma ** 2])  # Covariance matrix
    for z in range(channels):
        for x in range(size):
            for y in range(size):
                # Calculate the coordinates relative to the center
                coord = np.array([x - center, y - center, z - z_center])
                # Calculate the value of the Gaussian function at this point
                kernel[z, x, y] = st.multivariate_normal.pdf(coord, mean=np.zeros(3), cov=cov)
    kernel /= np.sum(kernel)
    return torch.tensor(kernel).unsqueeze(0)

def min_max_norm(in_):
    """
        normalization
    :param: in_
    :return:
    """
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_ - min_ + 1e-8)
