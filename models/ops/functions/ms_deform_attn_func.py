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

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import MultiScaleDeformableAttention as MSDA


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):#当你自定义的函数不可导时，在写backward函数时，就需要使用@once_differentiable
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape# (bs,h1w1+h2w2+...,8,32)
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape#(bs,h1w1+h2w2+...,8,4,4,2)，其中第一个是宽坐标，第二个是高坐标，坐标在01之间
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)#将value分成4个特征图的tensor(chunk)，(tensor(bs,h1w1,8,32),tensor(bs,h2w2,8,32)...)
    sampling_grids = 2 * sampling_locations - 1#从0到1之间的相对坐标，坐标系变换为从-1到1之间的相对坐标。这样的相对坐标形式是grid_sample要求的
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):#lvl, (h,w)
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)#(bs,8*32,h1w1) -> (bs*8,32,h1,w1)，这样的维度(N,c,Hin,Win)是grid_sample要求的
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)#(bs*8,h1w1+h2w2+...,4,2)某特征图中4个采样点的坐标
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)#目标grid尺寸为(h1w1+h2w2+...,4)，目标grid坐标（dim=-1）都在-1到1之间，即每个坐标表示-1到1坐标系下原特征图(h1,w1)的相对坐标。grid_sample会将-1到1的相对坐标重新变换为0到1的相对坐标，再插值得到value，最后得到(bs*8,32,h1w1+h2w2+...,4)
        sampling_value_list.append(sampling_value_l_)#对每个参考点分别得到4个特征图中的4个采样点的32维值，4个(bs*8,32,h1w1+h2w2+...,4)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)#(bs*8,1,h1w1+h2w2+...,4*4)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)#(bs*8,32,h1w1+h2w2+...,4,4) -> (bs*8,32,h1w1+h2w2+...,4*4)，某像素点对应的4*4个采样点分别乘对应的注意力权重，依此类推，得(bs*8,32,h1w1+h2w2+...,4*4)。加权求和后为(bs*8,32,h1w1+h2w2+...)，多头concat为(bs,8*32,h1w1+h2w2+...)
    return output.transpose(1, 2).contiguous()#(bs,h1w1+h2w2+...,256)
