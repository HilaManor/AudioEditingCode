"""Based on LPIPS from https://github.com/richzhang/PerceptualSimilarity/"""

from __future__ import absolute_import

import torch
import torch.nn as nn
from . import pretrained_networks as pn


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat/(norm_factor+eps)


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([1, 2], keepdim=keepdim)


def upsample(in_tens, out_HW=(64, 64)):  # assumes scale factor is same for H and W
    # in_H, in_W = in_tens.shape[1], in_tens.shape[2]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)


# Learned perceptual metric
class LPAPS(nn.Module):
    def __init__(self, net: str = 'clap', device: str = 'cpu',
                 net_kwargs: dict = {'model_arch': 'HTSAT-base',
                             'chkpt': 'music_speech_epoch_15_esc_89.25.pt',
                             'enable_fusion': False},
                 checkpoint_path='clap/pretrained', req_grad=False, spatial=False):
        
        """ Initializes a perceptual loss torch.nn.Module

        :param str net: the network to use
        :param str device: the device to use
        :param dict net_kwargs: the network keyword arguments
        :param str checkpoint_path: the checkpoint path
        :param bool req_grad: whether to require gradients
        :param bool spatial: whether to use spatial information
        """

        super(LPAPS, self).__init__()

        self.pnet_type = net
        self.extra_kwargs = net_kwargs
        self.req_grad = req_grad

        self.spatial = spatial

        if (self.pnet_type in 'clap'):
            net_type = pn.CLAP_base

        self.net = net_type(requires_grad=self.req_grad, **self.extra_kwargs, checkpoint_path=checkpoint_path,
                            device=device)
        self.L = self.net.get_num_layers()
        self.eval()

    def forward(self, in0, in1, sample_rates0, sample_rates1, retPerLayer=False):
        outs0, outs1 = self.net.forward(in0, sample_rates0), self.net.forward(in1, sample_rates1)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if (self.spatial):
            res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
        else:
            res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]

        val = 0
        for l in range(self.L):
            val += res[l]

        if (retPerLayer):
            return (val, res)
        else:
            return val
