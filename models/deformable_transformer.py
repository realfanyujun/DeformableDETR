# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        super().__init__()

        self.d_model = d_model#dimension of the transformer
        self.nhead = nhead#Number of attention heads inside the transformer's attentions
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))#(4,256)

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)#如果不是两阶段，那么每层特征图的ref点通过线性模块直接得到

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)#[0,1,...,127]
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)#[0,0,1,1,2,...,63,63] -> 10000 ** [0,0,2/128,2/128,4/128,...,126/128,126/128]
        # N, L, 4
        proposals = proposals.sigmoid() * scale#proposal bbox变成相对坐标后，*2pi
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t#(bs, 300, 4, [相对坐标*2pi*scale,128个数])
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos#(bs, 300, 512)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape##(bs,h1w1+h2w2+...,256)
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)#把lvl特征图的信息拿出来，#(bs,h1w1+h2w2+...)中(bs,h1w1) -> (bs, h1, w1, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)#(bs, h1(w1=0对应的列)) -> (bs)该尺寸特征图每个sample的有效高度
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)#(bs)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)#grid_y是(h1,w1)，每一列都是linspace(0, H_ - 1, H_); grid_x是(h1,w1)，每一行都是linspace(0, W_ - 1, W_)。cat后是(h1,w1,2)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)#(bs,1) cat后是(bs,2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale#(1,h1,w1,2) -> (bs, h1, w1, 2)，加0.5后grid又变成高在0.5到H-0.5间、宽在0.5到W-0.5之间的h*w个坐标（与之前取参考点和做pos embedding的坐标一样），再scale为有效目标中的相对位置
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)#proposal对应原图像中的相对w或h，最大0.4。特征图越大，越接近原图像；特征图越小，对应原图像的范围也翻倍（感知野）。实际上对应原图像的同一个点，得到了4个面积不同的框
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)#(bs, h1*w1, 4),每个像素点包括(相对宽坐标，相对高坐标，0.05 * (2.0 ** lvl), 0.05 * (2.0 ** lvl))
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)#(bs, h1w1+h2w2+..., 4)
        #proposal差不多可认为是原图片中非padding区域的所有相对坐标位置
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)#(bs, h1w1+h2w2+..., 1)最后的bool值表示该像素点是否符合范围
        output_proposals = torch.log(output_proposals / (1 - output_proposals))#sigmoid的逆函数#相对坐标和wh越小，值也越小（负无穷到正无穷）,padding区域会变成nan
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))#padding区域像素点对应的proposal不要了，用Inf填充(bs,h1w1+h2w2+..., 4)
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))#进一步按范围筛选proposal (bs,h1w1+h2w2+..., 4)

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))#padding区域用0填充
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):#(b,h,w)
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)#(b,h,只取w的第一列)按h相加，得到(b)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H#该尺寸特征图下，真实图片高度像素/特征图总高度像素
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)#(b,2)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):#self.query_embed = nn.Embedding(num_queries, hidden_dim*2)，query_embeds = self.query_embed.weight
        assert self.two_stage or query_embed is not None#如果不是两阶段，目标检测query要有embed

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):#不同尺寸的特征图
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)#(bs,h*w,c)
            mask = mask.flatten(1)#(bs,h*w)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)#(bs,h*w,c)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)#(bs,h*w,256)+(1,1,256) -> (bs,h*w,256)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)#将不同尺寸特征图按h*w叠加,即(bs,h1w1+h2w2+...,c)
        mask_flatten = torch.cat(mask_flatten, 1)#(bs,h1w1+h2w2+...)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)#(bs,h1w1+h2w2+...,c)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)#(4,2)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))#[0,h1w1,h1w1+h2w2,hw...],分别是每个level开始的索引
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)#(b,4,2)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)#(bs,h1w1+h2w2+...,256)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)#每个特征图中每个点得到1个proposal，按是否为padding进行了筛选。(bs,h1w1+h2w2+..., 256)(bs,h1w1+h2w2+..., 4)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)#(bs,h1w1+h2w2+...,num_class)#class_embed的最后一层（对应decoder层数之后的一层）通过将256转化为num_class，对每个proposal的信息预测分类
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals#(bs,h1w1+h2w2+...,4)#对每个proposal的信息预测bbox offset，加上proposal就是预测bbox（其中proposal已提前逆sigmoid）#padding区域还是inf

            topk = self.two_stage_num_proposals#300
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]#类别为0的概率最大的300个proposals的索引，(bs, 300)
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))#(bs,300,4)对每个样本，拿出300个proposal bbox #topk proposals: (bs,300,1) -> (bs, 300, 4)
                                                            #本来padding区域对应的proposal也有可能被选到topk中，但padding区域的memory已设为0，所有class的logit都是0.01，被选到的可能比较小（但仍然有可能）
            topk_coords_unact = topk_coords_unact.detach()#已经对用于预测proposal bbox offset的bbox_embed增加了loss函数，会让预测的proposal bbox更接近真实，因此不需要从后面decoder往这个bbox_embed传递梯度了。
            reference_points = topk_coords_unact.sigmoid()#每个query对应的参考点，相对坐标，包括宽和高(bs, 300, 4)
                                                            #padding区域的proposal bbox之前设为Inf，被sigmoid后相对坐标是(1,1,1,1)，即padding区域被选到的proposals其实是整个原图像（不包含padding区域）
                                                            #因此，所有proposals的bbox（参考点）都在不含padding区域的原图像内
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))#(bs, 300, 256*2), 300个embed各不相同，且其中包含了proposal bbox信息。对这300个embed投射后norm
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)#300个query, embed和tgt值各占一半维度(bs, 300, 256)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)#2个(num_queries, 256)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)#(bs,num_queries, 256)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)#(bs,num_queries, 256)
            reference_points = self.reference_points(query_embed).sigmoid()#预测的参考点是相对于原图像真实目标的相对位置(bs,num_queries, 2),decoder中还会进一步转换成4层特征图中的相对位置(bs,num_queries, 4, 2)用于attention采样。参考点还是预测的bbox基准点
            init_reference_out = reference_points#这里直接用query_embed去预测参考点，因为觉得只要每个num_query各自对应不同参考点就行

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):#(bs,h1w1+h2w2+...,c)+(bs,h1w1+h2w2+...,c)
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):#(4,2)

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)#得到的是(b,h*w每个参考点的高度x在真实图片像素高度中的比例)。前面被除的是每个参考点的高度x (1,h*w)；后面除的是(b,1#该尺寸特征图下真实图片高度像素)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)#(b,h*w) 将原参考点的坐标在真实图片像素中的比例作为新的参考点坐标，即scale到0与1之间，padding部分坐标超过了1。先算真实图片像素的相对坐标的作用是：可以使一个像素点在多个特征图中的参考点是实际目标的同一个对应位置，这样参考点是描述同一个位置，才适合融合
            ref = torch.stack((ref_x, ref_y), -1)#(b,h*w,2)
            reference_points_list.append(ref)#(4个(b,h*w,2)) 第一个是宽的坐标，第二个是高的坐标
        reference_points = torch.cat(reference_points_list, 1)#(b,4*h*w,2) 4个特征图h*w个参考点的坐标
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]#(b,4个特征图h*w之和,1,2)*(b,1,4,2)，得到(b,4个特征图h*w之和,4,2),进一步scale,这样其实还是坐标除以总像素大小。对第一个特征图的参考点坐标，得到4个坐标，(ref_x/w1, ref_y/h1), (ref_x * w2'/(w1'*w2), ref_x * h2'/(h1'*h2))。每个像素是一个参考点，在4个特征图对应位置分别得到一个相对坐标，用于分别在4个特征图中根据offset找采样点
        return reference_points
    '''
    这里关键点在于作者没有将同一像素点在四个图中的相对坐标使用同一个数(ref_x/w1, ref_y/h1)表示，而是通过在非padding区域相对坐标转换，参见get reference point。因为作者认为重要的不是像素点在全图中的相对位置，而是在真实图片（非padding区域中的相对位置），这也是在另一个特征图中的真实图片中的相对位置，再根据另一个特征图padding区域和全图之间的面积比例关系scale为在全图中的相对位置。
比如一个点在原图片小狗的正中间，那该点在另一个特征图中也应该对应小狗特征的正中间，这样才能说这两个特征图中分别对应的采样点是相关的，可以融合。

也就是说，4个特征图h*w之和中任意一个像素点都是参考点，对于每个参考点不光给出在对应特征图中的相对坐标位置，还给出在其它特征图中对应位置（描述物体同一位置）的相对坐标，这样每个参考点就有4个相对坐标。
大致学习的逻辑：算法对于一个给定参考点的信息，可以在其它特征图中找到对应位置（无论value还是位置肯定是和给定参考点相关度最高），然后对每个特征图中的对应点找到最相关的4个采样点。因此给定一个参考点可以通过MLP直接得到4个特征图中的采样点，这些采样点都与参考点描述的位置很相关。同样可给出相互的注意力权重。
由于这些采样点都与给定参考点描述的很相关，因此可以进行融合。对于深层小尺寸特征图中的小目标，可能特征已经很不明显了，但通过融合大尺寸特征图中对应位置的相关特征，可以让目标特征更明确，有助于小目标检测
    '''

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)#(bs, num_query, 256)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)#参数不共享
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt#(bs,num_queries, 256)，tgt是query的value；query_pos是前面的query_embed，作为tgt的pos embed

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:#参考点是300个proposal（按class 0概率排序，padding区域像素对应的proposal是1 1 1 1），且使用bbox_embed修正过 #tgt和query embed都与参考点坐标有关
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]#(bs,num_queries, 1, 4) * (b,1,4,4)#bbox参考点转成每个特征图的相对坐标
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]##(bs,num_queries, 2) -> (bs,num_queries, 1, 2) * (b,1,4,2)，对每个query，参考点坐标（预测的应该是在真实目标中的相对位置）乘4个特征图的valid ratio，变成参考点在对应特征图中的相对位置坐标，(b, num_query, 4, 2)
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)#上一层decoder的输出和参考点，输入下一层decoder。后面会对每层decoder的输出预测（相对输入参考点）的bbox offset

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)#两阶段时预测的宽和高也是参考点的offset
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)#这一层decoder的输出预测bbox offset，这个offset是对这一层decoder输入的参考点的修正
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()#得到了新的修正参考点，最后的Box_embed对最后的参考点预测offset。基于新的参考点的后续计算不计算梯度，之前的参数梯度不计入这些参考点的梯度贡献
                '''
                最后一层bbox_embed被优化于使在给定输入的参考点时，最后bbox_embed输出的offset使bbox误差越小。如果不detach new ref，那么优化最后一层bbox_embed时也会将梯度传递到上一层bbox_embed，
                但我们希望上一层bbox_embed被用于优化于给定上上一层输出的参考点时，上一层输出的offset使bbox误差越小。即每一层Bbox_embed仅对其输出的offset负责，而不管其他层的offset。
                最后iterative bbox refine下，每层decoder输出的bbox都趋向于真实目标
                '''

            if self.return_intermediate:
                intermediate.append(output)#这一层decoder的输出(bs,num_queries, 256)
                intermediate_reference_points.append(reference_points)#下一层decoder输入的参考点(bs,num_queries, 4, 2)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries)


