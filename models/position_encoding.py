# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):#输入的是backbone输出的某个尺寸特征图的数据NestedTensor(x, mask)
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask#按mask取反，padding区域全0，其他区域全1
        y_embed = not_mask.cumsum(1, dtype=torch.float32)#按h维度计算累加值
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale#真实图像高度坐标从0.5到w*-0.5，pad部分为-0.5，这样embedding就能够反映高度坐标和哪些部分是pad。再除以h方向之和标准化到01之间，再乘2pi
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale#反映真实图像宽度坐标和pad区域

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)#0到127
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)#(2 * (dim_t // 2) / self.num_pos_feats)是小于等于1的128维逐渐递增序列 0 0 1/128 1/128 126/128 126/128 -> 10000**后是从1开始逐渐递增逼近10000的序列

        pos_x = x_embed[:, :, :, None] / dim_t#(b,h,w,featuredim)，对于b*h*w中每个像素形成128维深度，且同一个像素的不同dim有不同取值
        pos_y = y_embed[:, :, :, None] / dim_t#pos_x能反映不同的深度、width和pad区域
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)#256维深度一半sin，一半cos。这样就算相邻两个数几乎相同也可以区分开
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)#pos_y和pos_x按深度d叠加，256维
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)#一般横纵坐标不超过50，建立一个包含50个向量的词库
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)#(w,d)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),#(1,w,d)按y方向复制成(h,w,d)
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)#pos_y和pos_x按深度d叠加,(1,d,h,w)按batch中数据数量复制(b,d,h,w)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2#"Size of the embeddings (dimension of the transformer)",这里除以2是为了将pos_x和pos_y叠加
    if args.position_embedding in ('v2', 'sine'):#"Type of positional embedding to use on top of the image features"
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
