# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads#每个Head输出32维向量
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)#每个head每个level特征图得到4个相对ref点的横纵坐标offset，权重初始化为0
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)#[[cos0*2pi,sin0*2pi],[cos1/8 *2pi,sin1/8 *2pi]...[cos1*2pi,sin1*2pi]]
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)#(8,4,4,2)#8对数绝对值化后每对选最大值，用来给每对数被除(变成0到1范围内)，每个Head的16个采样点offset的scale一样
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1#每层采样4个点，4个点的bias依次增大，初始化在-k到k之间
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))#bias初始化，可能通过这样的bias希望学习出由近到远的4个采样点
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 1 / (self.n_levels * self.n_points))
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        #encoder的自注意力中query是input_flatten+pos emb+level emb；input_flatten是将不同尺寸特征图按h*w叠加,即(bs,h1w1+h2w2+...,256)。
        #encoder中参考点是h1w1+h2w2+...的每个像素点，并对每个像素点在4个特征图中分别找到对应位置的相对坐标。
        #encoder中通过对query直接MLP得到每个参考点在各个特征图中的采样点offset，并得到采样点之间的注意力权重。通过插值求得所有采样点的value，根据注意力权重融合，返回#(bs,h1w1+h2w2+...,256)
        #decoder中的交叉注意力中query是tgt + query pos（初始都是直接用的embedding）。input_flatten与encoder一样，即(bs,h1w1+h2w2+...,256)，仅用来插值求采样点的value。
        #decoder中参考点是对每个tgt对应的query embedding直接MLP预测得到参考点相对坐标，并得到参考点在4个特征图中对应位置的相对坐标。和encoder只是参考点数量不同。
        #decoder中同样通过对query直接MLP得到每个参考点在各个特征图中的采样点offset，然后采样点值融合，返回#(bs,num_query,256)
        N, Len_q, _ = query.shape#bs,num_query,256
        N, Len_in, _ = input_flatten.shape#bs,h1w1+h2w2+...,256
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)#(bs,h1w1+h2w2+...,256) 特征图Project为256维value, 不包含embedding
        if input_padding_mask is not None:#(bs,h1w1+h2w2+...)
            value = value.masked_fill(input_padding_mask[..., None], float(0))#(bs,h1w1+h2w2+...,256)中padding部分的值变为0，这样如果采样点在padding区域就不会被用到
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)#划分为8头 (bs,h1w1+h2w2+...,8,32)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)#加了pos embedding的特征图用于生成offset，(bs,num_query,256)project为(bs,h1w1+h2w2+...,8,4,4,2)。某个特征图的一个参考点可以同时生成4个特征图的采样点
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)#加了pos embedding的特征图用于生成8*4*4个采样点的注意力权重
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)#4个特征图共16个采样点的注意力权重softmax标准化，即对每个head的16个采样点按注意力权重加权平均，多层特征融合
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)#(4,2)其中第一个是宽，第二个是高
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]#(b,h1w1+h2w2+...,1,4,1,2) + (bs,h1w1+h2w2+...,8,4,4,2) / (1,1,1,4,1,2)#预测的是实际坐标的差距，将每个特征图的4个采样点offsets除以对应特征图的宽和高，使得offset也标准化到01之间
        elif reference_points.shape[-1] == 4:#由于初始化时offset在-k到k（4个点）之间，因此这里除以n_points进行scale（初始时可归一化），再乘以bbox宽和高的一半，使得参考点的移动还在bbox范围内，不会移动太多
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5#(b,h1w1+h2w2+...,1,4,1,(cx,cy)) + (bs,h1w1+h2w2+...,8,4,4,2) / 4 * (b,h1w1+h2w2+...,1,4,1,(w,h)) * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output


'''
对多尺度特征图中的每个点找到它在所有特征图中的对应点，即每个点找4个参考点。这里的对应点不光看在全图中的相对位置，而是找对应实际目标中的相对位置，这样这几个参考点的信息才是适合融合的。
第一个特征图某个像素点，的第一个参考点的坐标，(ref_x/w1, ref_y/h1)该像素点相对第一个特征图的相对位置
加上对应像素点，的第一个head，的第一个特征图，的第一个offset相对第一个特征图的坐标
加上对应像素点，的第一个head，的第一个特征图，的第二个offset相对第一个特征图的坐标
加上对应像素点，的第一个head，的第一个特征图，的第三个offset相对第一个特征图的坐标
加上对应像素点，的第一个head，的第一个特征图，的第四个offset相对第一个特征图的坐标
。。。到这里没错

第一个特征图某个像素点，的第二个参考点的坐标，(ref_x * w2'/(w1'*w2), ...)该像素点相对第二个特征图的相对位置，其中ref_x * w2'/w1'是该对应位置的实际坐标
加上对应像素点，的第一个head，的第二个特征图，的第一个offset相对第二个特征图的坐标。
加上对应像素点，的第一个head，的第二个特征图，的第二个offset相对第二个特征图的坐标
加上对应像素点，的第一个head，的第二个特征图，的第三个offset相对第二个特征图的坐标
加上对应像素点，的第一个head，的第二个特征图，的第四个offset相对第二个特征图的坐标

这里关键点在于作者没有将同一像素点在四个图中的相对坐标使用同一个数(ref_x/w1, ref_y/h1)表示，而是通过在非padding区域相对坐标转换，参见get reference point。因为作者认为重要的不是像素点在全图中的相对位置，而是在真实图片（非padding区域中的相对位置），这也是在另一个特征图中的真实图片中的相对位置，再根据另一个特征图padding区域和全图之间的面积比例关系scale为在全图中的相对位置。
比如一个点在原图片小狗的正中间，那该点在另一个特征图中也应该对应小狗特征的正中间，这样才能说这两个特征图中分别对应的采样点是相关的，可以融合。
'''
