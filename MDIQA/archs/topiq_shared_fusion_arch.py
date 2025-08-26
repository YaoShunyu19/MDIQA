"""TOP-IQ metric, proposed by

TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment.
Chaofeng Chen, Jiadi Mo, Jingwen Hou, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, Weisi Lin.
Arxiv 2023.

Paper link: https://arxiv.org/abs/2308.03060

TOPIQ baseline的网络结构
resnet50 + conv fusion + 2 branch + weighted average
"""

from MDIQA.archs.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import timm
from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from pyiqa.archs.arch_util import dist_to_mos, load_pretrained_network, random_crop

import copy
import open_clip
from torchvision import transforms
import math
from MDIQA.utils.registry import ARCH_REGISTRY
from MDIQA.archs.module_utils import *
from MDIQA.archs.topiq_arch import TransformerEncoderLayer

default_model_urls = {
    'cfanet_fr_kadid_res50': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/cfanet_fr_kadid_res50-2c4cc61d.pth',
    'cfanet_fr_pipal_res50': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/cfanet_fr_pipal_res50-69bbe5ba.pth',
    'cfanet_nr_flive_res50': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/cfanet_nr_flive_res50-ded1c74e.pth',
    'cfanet_nr_koniq_res50': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/cfanet_nr_koniq_res50-9a73138b.pth',
    'cfanet_nr_spaq_res50': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/cfanet_nr_spaq_res50-a7f799ac.pth',
    'cfanet_iaa_ava_res50': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/cfanet_iaa_ava_res50-3cd62bb3.pth',
    'cfanet_iaa_ava_swin': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/cfanet_iaa_ava_swin-393b41b4.pth',
    'topiq_nr_gfiqa_res50': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/topiq_nr_gfiqa_res50-d76bf1ae.pth',
}


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


@ARCH_REGISTRY.register()
class CFANet_base_unshared(nn.Module):
    def __init__(self,
                 semantic_model_name='resnet50',
                 model_name='cfanet_base_nr_koniq_res50',
                 backbone_pretrain=True,
                 in_size=None,
                 use_ref=True,
                 quality_map=False,
                 num_class=1,
                 num_crop=1,
                 crop_size=256,
                 inter_dim=256,
                 activation='gelu',
                 pretrained_model_path=None,
                 out_act=False,
                 test_img_size=None,
                 default_mean=IMAGENET_DEFAULT_MEAN,
                 default_std=IMAGENET_DEFAULT_STD,
                 ):
        super().__init__()

        self.in_size = in_size

        self.model_name = model_name
        self.semantic_model_name = semantic_model_name
        self.semantic_level = -1
        self.crop_size = crop_size
        self.use_ref = use_ref
        self.quality_map = quality_map
        self.num_class = num_class
        self.test_img_size = test_img_size

        # =============================================================
        # define semantic backbone network
        # =============================================================

        # assert semantic_model_name in ['resnet50', 'mobilenetv3_large_100.ra_in1k', 'convnextv2_tiny.fcmae_ft_in22k_in1k_384']
        self.semantic_model = timm.create_model(semantic_model_name, pretrained=backbone_pretrain, features_only=True)
        feature_dim_list = self.semantic_model.feature_info.channels()
        self.fix_bn(self.semantic_model)

        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)

        # =============================================================
        # define self-attention and cross scale attention blocks
        # =============================================================

        self.act_layer = nn.GELU() if activation == 'gelu' else nn.ReLU()

        linear_dim = inter_dim * len(feature_dim_list)

        self.dim_reduce = nn.ModuleList()
        for _, dim in enumerate(feature_dim_list):
            dim = dim * 3 if use_ref else dim
            self.dim_reduce.append(nn.Sequential(
                nn.Conv2d(dim, inter_dim, 1, 1),
                self.act_layer,
                )
            )
        self._init_linear(self.dim_reduce)

        self.score_linear = [
            nn.LayerNorm(linear_dim),
            nn.Linear(linear_dim, linear_dim),
            self.act_layer,
            nn.Linear(linear_dim, 1),
        ]
        self.weight_linear = [
            nn.LayerNorm(linear_dim),
            nn.Linear(linear_dim, linear_dim),
            self.act_layer,
            nn.Linear(linear_dim, 1),
        ]

        # make sure output is positive, useful for 2AFC datasets with probability labels
        if out_act:
            self.score_linear.append(nn.Softplus())

        self.weight_linear.append(nn.Sigmoid())

        self.score_linear = nn.Sequential(*self.score_linear)
        self.weight_linear = nn.Sequential(*self.weight_linear)

        if self.quality_map:
            self.conv_fusion = nn.Sequential(
                nn.Conv2d(linear_dim, inter_dim, 1, 1, 0),
                self.act_layer,
                nn.Conv2d(inter_dim, inter_dim, 1, 1, 0),
            )
            self.quality_map_linear = [
                nn.LayerNorm(inter_dim),
                nn.Linear(inter_dim, inter_dim),
                self.act_layer,
                nn.Linear(inter_dim, 1),
                nn.Sigmoid()
            ]
            self.quality_map_linear = nn.Sequential(*self.quality_map_linear)

        self.eps = 1e-8
        self.crops = num_crop

        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, False, weight_keys='params')

    def _init_linear(self, m):
        for module in m.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                nn.init.constant_(module.bias.data, 0)

    def preprocess(self, x):
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        return x

    def fix_bn(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False
                m.eval()

    def forward_single_dim(self, x, y=None, if_quality_map=False):

        # resize image when testing
        if not self.training:
            if self.model_name == 'cfanet_iaa_ava_swin':
                x = TF.resize(x, [384, 384], antialias=True)  # swin require square inputs
            elif self.test_img_size is not None:
                x = TF.resize(x, self.test_img_size, antialias=True)

        x = self.preprocess(x)  # B,3,384,384
        if self.use_ref:
            y = self.preprocess(y)

        dist_feat_list = self.semantic_model(x)
        if self.use_ref:
            ref_feat_list = self.semantic_model(y)
        self.fix_bn(self.semantic_model)
        self.semantic_model.eval()

        start_level = 0
        end_level = len(dist_feat_list)   # [B,ci,384/2^i,384/2^i]

        _, _, th, tw = dist_feat_list[end_level - 2].shape  # TODO

        if self.quality_map and if_quality_map:
            tmp_dist_feat_list = []
            for i in range(start_level, end_level):
                tmp_feat = dist_feat_list[i]
                tmp_feat = self.dim_reduce[i](tmp_feat)
                if tmp_feat.shape[2] != th and tmp_feat.shape[3] != tw:
                    tmp_feat = F.adaptive_avg_pool2d(tmp_feat, (th, tw))
                tmp_dist_feat_list.append(tmp_feat)
            dist_feat = torch.cat(tmp_dist_feat_list, dim=1)  # B,5c,h,w
            local_feature = self.conv_fusion(dist_feat)
            local_feature_ = local_feature.permute(0, 2, 3, 1)
            quality_map = self.quality_map_linear(local_feature_)[:, :, :, 0]  # B,h,w
            return torch.mean(quality_map, dim=[-2, -1]).unsqueeze(1), local_feature, quality_map
        else:
            tmp_dist_feat_list = []
            for i in range(start_level, end_level):
                tmp_feat = dist_feat_list[i]
                tmp_feat = self.dim_reduce[i](tmp_feat)
                if tmp_feat.shape[2] != th and tmp_feat.shape[3] != tw:
                    tmp_feat = F.adaptive_avg_pool2d(tmp_feat, (th, tw))
                tmp_dist_feat_list.append(tmp_feat)
            dist_feat = torch.cat(tmp_dist_feat_list, dim=1)  # B,5c,h,w
            local_feature = dist_feat
            dist_feat = dist_feat.permute(0, 2, 3, 1)
            score = self.score_linear(dist_feat)  # B,h,w,1
            weight = self.weight_linear(dist_feat)  # B,h,w,1
            global_score = torch.mean(weight * score, dim=[-3, -2]) / (torch.mean(weight, dim=[-3, -2]) + self.eps)
            # B,1  B,c,h,w
            return global_score, local_feature, None

    def forward(self, x, y=None, if_quality_map=False):
        if self.use_ref:
            assert y is not None, f'Please input y when use reference is True.'
        else:
            y = None

        if self.crops > 1 and not self.training:
            bsz = x.shape[0]
            if y is not None:
                x, y = random_crop([x, y], self.crop_size, self.crops)
            else:
                x = random_crop([x], self.crop_size, self.crops)  # B*N,3,h,w
            global_scores, local_scores, quality_map = self.forward_single_dim(x, y, if_quality_map)  # B,1  B,h,w  B,h,w
            global_scores = global_scores.reshape(bsz, self.crops, 1)
            global_scores = global_scores.mean(dim=1)
        else:
            global_scores, local_scores, quality_map = self.forward_single_dim(x, y, if_quality_map)  # B,1  B,h,w  B,h,w

        return global_scores, local_scores, quality_map


class InstanceSpecificWeightedBranch(CFANet_base_unshared):
    def __init__(self,
                 semantic_model_name='resnet50',
                 model_name='cfanet_base_nr_koniq_res50',
                 backbone_pretrain=True,
                 in_size=None,
                 use_ref=True,
                 quality_map=False,
                 num_class=1,
                 num_crop=1,
                 crop_size=256,
                 inter_dim=256,
                 activation='gelu',
                 pretrained_model_path=None,
                 out_act='softmax',
                 test_img_size=None,
                 default_mean=IMAGENET_DEFAULT_MEAN,
                 default_std=IMAGENET_DEFAULT_STD):
        super().__init__(semantic_model_name,
                         model_name,
                         backbone_pretrain,
                         in_size,
                         use_ref,
                         quality_map,
                         num_class,
                         num_crop,
                         crop_size,
                         inter_dim,
                         activation,
                         pretrained_model_path,
                         out_act,
                         test_img_size,
                         default_mean,
                         default_std)

        feature_dim_list = self.semantic_model.feature_info.channels()
        linear_dim = inter_dim * len(feature_dim_list)
        score_linear = [
            nn.LayerNorm(linear_dim),
            nn.Linear(linear_dim, linear_dim),
            self.act_layer,
            nn.Linear(linear_dim, num_class),
        ]
        if out_act == 'softmax':
            self.out_act = nn.Softmax(dim=-1)
        elif out_act == 'softplus':
            self.out_act = nn.Softplus()
        else:
            self.out_act = nn.Identity()
        self.score_linear = nn.Sequential(*score_linear)

        del self.weight_linear

    def forward(self, x, y=None, weight_ratio=None):
        # resize image when testing
        if not self.training:
            if self.model_name == 'cfanet_iaa_ava_swin':
                x = TF.resize(x, [384, 384], antialias=True)  # swin require square inputs
            elif self.test_img_size is not None:
                x = TF.resize(x, self.test_img_size, antialias=True)

        x = self.preprocess(x)  # B,3,384,384
        if self.use_ref:
            y = self.preprocess(y)

        dist_feat_list = self.semantic_model(x)
        if self.use_ref:
            ref_feat_list = self.semantic_model(y)
        self.fix_bn(self.semantic_model)
        self.semantic_model.eval()

        start_level = 0
        end_level = len(dist_feat_list)   # [B,ci,384/2^i,384/2^i]

        _, _, th, tw = dist_feat_list[end_level - 2].shape  # TODO

        tmp_dist_feat_list = []
        for i in range(start_level, end_level):
            tmp_feat = dist_feat_list[i]
            tmp_feat = self.dim_reduce[i](tmp_feat)
            if tmp_feat.shape[2] != th and tmp_feat.shape[3] != tw:
                tmp_feat = F.adaptive_avg_pool2d(tmp_feat, (th, tw))
            tmp_dist_feat_list.append(tmp_feat)

        dist_feat = torch.cat(tmp_dist_feat_list, dim=1)  # B,5c,h,w

        score = self.score_linear(dist_feat.permute(0, 2, 3, 1))  # B,h,w,C
        weight = torch.mean(score, dim=[-3, -2])  # B,C
        weight = self.out_act(weight)
        if weight_ratio is not None:
            weight = weight * weight_ratio
        
        return weight


class MLP_fusion_feature(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.in_ln = nn.Linear(in_channels, hidden_channels, bias=False)
        self.out_ln = nn.Linear(hidden_channels, out_channels, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return self.out_ln(self.dropout(self.gelu(self.in_ln(x))))


class LocalScoreModule(nn.Module):
    def __init__(self, num_dim, aes_num, inter_dim=256, semantic_feat=False):
        super().__init__()
        self.tech_num = num_dim - aes_num
        self.fusion_before_attn = nn.ModuleList()
        self.attn_pools = nn.ModuleList()
        self.score_linears = nn.ModuleList()

        for i in range(num_dim):
            if semantic_feat:
                self.fusion_before_attn.append(MLP_fusion_feature(2*inter_dim, inter_dim//2, inter_dim))
            else:
                self.fusion_before_attn.append(MLP_fusion_feature(inter_dim, inter_dim//2, inter_dim))

            self.attn_pools.append(TransformerEncoderLayer(inter_dim, nhead=4, dim_feedforward=1024, normalize_before=True, dropout=0.1, activation='gelu'))
            score_linear = [
                nn.LayerNorm(inter_dim),
                nn.Linear(inter_dim, inter_dim),
                nn.GELU(),
                nn.LayerNorm(inter_dim),
                nn.Linear(inter_dim, inter_dim),
                nn.GELU(),
                nn.Linear(inter_dim, 1),
            ]
            score_linear = nn.Sequential(*score_linear)
            self.score_linears.append(score_linear)

        net_inter_dim = 4 * num_dim
        self.net = [
            nn.Linear(num_dim, net_inter_dim),
            nn.GELU(),
            nn.Linear(net_inter_dim, net_inter_dim),
            nn.GELU(),
            nn.Linear(net_inter_dim, 1)
        ]
        self.net = nn.Sequential(*self.net)
        self._positive_initialize_weights(self.net)

        self._init_linear(self.attn_pools)
        self._init_linear(self.score_linears)

    def _init_linear(self, m):
        for module in m.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                nn.init.constant_(module.bias.data, 0)

    def _positive_initialize_weights(self, m):
        for module in m.modules():
            if isinstance(module, nn.Linear):
                init.normal_(module.weight, mean=0.0, std=0.1)
                module.weight.data = torch.abs(module.weight.data)
                if module.bias is not None:
                    init.normal_(module.bias, mean=0.0, std=0.1)
                    module.bias.data = torch.abs(module.bias.data)

    def forward(self, global_scores, local_scores, final_tokens, weight, semantic_feature):
        # B,d, [(B,c,h,w)*d], B,d  c=256 d:num_dim
        token_feat_list = []
        for i, feat in enumerate(local_scores):
            tmp_feat = feat
            c, h, w = tmp_feat.shape[1], tmp_feat.shape[2], tmp_feat.shape[3]

            tmp_feat = tmp_feat.flatten(2).permute(2, 0, 1)  # B,c,h,w->h*w,B,c
            res_feat = tmp_feat

            if semantic_feature is not None:
                semantic_feature_ = F.interpolate(semantic_feature, size=(h, w), mode='bicubic', align_corners=False).flatten(2).permute(2, 0, 1)
                tmp_feat = torch.cat([tmp_feat, semantic_feature_], dim=-1)

            tmp_feat = self.fusion_before_attn[i](tmp_feat) + res_feat
            tmp_feat = self.attn_pools[i](tmp_feat).mean(dim=0)  # B,c
            tmp_feat = self.score_linears[i](tmp_feat)  # B,1
            token_feat_list.append(tmp_feat)

        token_feat = torch.cat(token_feat_list, dim=1) + global_scores  # B,d
        local_weighted_score = token_feat * weight
        weighted_score = local_weighted_score

        out_score = self.net(weighted_score)  # B,1
        return out_score


@ARCH_REGISTRY.register()
class MultiDimFusionWeightedMLP(nn.Module):
    def __init__(self,
                 num_dim=1,
                 aes_num=0,
                 inter_dim=18,
                 activation='gelu',
                 dataset_specific_weighted=0,
                 instance_specific_weighted=False,
                 semantic_model_name='resnet50',
                 local_score=False,
                 semantic_feat='without',
                 out_act='softmax',
                 ):
        super().__init__()
        self.act_layer = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.local_score = local_score
        if self.local_score is not False:
            if_semantic_feat = False if semantic_feat == 'without' else True
            self.local_score_module = LocalScoreModule(num_dim, aes_num, semantic_feat=if_semantic_feat)
        else:
            inter_dim = 4 * num_dim
            self.net = [
                nn.Linear(num_dim, inter_dim),
                self.act_layer,
                nn.Linear(inter_dim, inter_dim),
                self.act_layer,
                nn.Linear(inter_dim, 1)
            ]
            self.net = nn.Sequential(*self.net)
            self._positive_initialize_weights(self.net)

        self.dataset_specific_weights = nn.ParameterList()
        self.dataset_specific_weights.append(nn.Parameter(torch.ones(num_dim), requires_grad=False))
        for _ in range(dataset_specific_weighted):
            weight = nn.Parameter(torch.ones(num_dim), requires_grad=True)
            self.dataset_specific_weights.append(weight)

        self.instance_specific_weighted = instance_specific_weighted
        if self.instance_specific_weighted is not False:
            self.instance_specific_weighted_branch = InstanceSpecificWeightedBranch(semantic_model_name=semantic_model_name,
                                                                                    use_ref=False,
                                                                                    quality_map=False,
                                                                                    num_class=num_dim,
                                                                                    out_act=out_act)

        self.semantic_feat = semantic_feat
        if 'clip' in self.semantic_feat:
            model_name = self.semantic_feat.split('_')[-1]
            self.semantic_model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=f'/root/.cache/clip/{model_name}.pt')
            steps_to_retain = [transforms.Resize, transforms.CenterCrop, transforms.Normalize]
            new_preprocess = [t for t in preprocess.transforms if any(isinstance(t, step_type) for step_type in steps_to_retain)]
            self.semantic_preprocess = transforms.Compose(new_preprocess)
            self.semantic_feat_conv = nn.Conv2d(1024, 256, 1, 1, 0)
            for p in self.semantic_model.parameters():
                p.requires_grad = False

    def _positive_initialize_weights(self, m):
        for module in m.modules():
            if isinstance(module, nn.Linear):
                init.normal_(module.weight, mean=0.0, std=0.1)
                module.weight.data = torch.abs(module.weight.data)
                if module.bias is not None:
                    init.normal_(module.bias, mean=0.0, std=0.1)
                    module.bias.data = torch.abs(module.bias.data)

    def forward(self, x, scores, weight_ratio=None, weights_idx=0):
        # scores: B,d / [(B,c,h,w)*d]  d: num_dim
        global_scores, local_scores, final_tokens = scores[0], scores[1], scores[2]

        semantic_feature = None
        with torch.no_grad():
            if 'clip' in self.semantic_feat:
                semantic_feature = self.semantic_model.encode_image(self.semantic_preprocess(x))[1:, :, :]
                w = int(math.sqrt(semantic_feature.shape[0]))
                semantic_feature = semantic_feature.reshape(w, w, x.shape[0], -1).permute(2, 3, 0, 1)  # B,d,h,w
                semantic_feature = self.semantic_feat_conv(semantic_feature)

        if self.local_score and self.instance_specific_weighted:
            weight = self.instance_specific_weighted_branch(x, weight_ratio=weight_ratio)  # B,d
            score = self.local_score_module(global_scores, local_scores, final_tokens, weight, semantic_feature)
        elif self.local_score and not self.instance_specific_weighted:
            weight = self.dataset_specific_weights[weights_idx][None, ...]  # 1,d
            score = self.local_score_module(global_scores, local_scores, final_tokens, weight, semantic_feature)
        elif not self.local_score and self.instance_specific_weighted:
            weight = self.instance_specific_weighted_branch(x, weight_ratio=weight_ratio)
            weighted_scores = global_scores * weight  # B,d * B,d
            score = self.net(weighted_scores)
        else:
            weight = self.dataset_specific_weights[weights_idx][None, ...]
            weighted_scores = global_scores * weight  # B,d * B,d
            score = self.net(weighted_scores)
        return score
