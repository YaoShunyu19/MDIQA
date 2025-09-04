from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F

import timm
from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .topiq_arch import TransformerEncoderLayer




class InstanceSpecificWeightedBranchv2(nn.Module):
    def __init__(self,
                 semantic_model_name='resnet50',
                 backbone_pretrain=True,
                 num_class=1,
                 default_mean=IMAGENET_DEFAULT_MEAN,
                 default_std=IMAGENET_DEFAULT_STD):
        super().__init__()

        self.semantic_model = timm.create_model(semantic_model_name, pretrained=backbone_pretrain)
        self.semantic_model.fc = torch.nn.Linear(self.semantic_model.fc.in_features, num_class)
        self.fix_bn(self.semantic_model)

        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)

    def fix_bn(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False
                m.eval()

    def preprocess(self, x):
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        return x

    def forward(self, x):
        x = self.preprocess(x)  # B,3,384,384
        weight = self.semantic_model(x)
        self.fix_bn(self.semantic_model)
        self.semantic_model.eval()
        weight = torch.softmax(weight, dim=-1)

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



class LocalScoreModule_v4(nn.Module):
    def __init__(self, num_dim, aes_num, inter_dim=256, semantic_feat=False, mask_feature=False):
        super().__init__()
        self.tech_num = num_dim - aes_num
        self.fusion_before_attn = nn.ModuleList()
        self.attn_pools = nn.ModuleList()
        self.score_linears = nn.ModuleList()

        for i in range(num_dim):
            if (semantic_feat & mask_feature) & (i < 10000):  # TODO  i < self.tech_num
                self.fusion_before_attn.append(MLP_fusion_feature(3*inter_dim, inter_dim//2, inter_dim))
            elif (semantic_feat ^ mask_feature) & (i < 10000):  # TODO  i < self.tech_num
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

    def forward(self, global_scores, local_scores, final_tokens, weight, semantic_feature, mask_feature=None):
        # B,d, [(B,c,h,w)*d], B,d  c=256 d:num_dim
        token_feat_list = []
        for i, feat in enumerate(local_scores):
            tmp_feat = feat
            c, h, w = tmp_feat.shape[1], tmp_feat.shape[2], tmp_feat.shape[3]

            tmp_feat = tmp_feat.flatten(2).permute(2, 0, 1)  # B,c,h,w->h*w,B,c
            res_feat = tmp_feat

            if i < 10000:  # TODO  i < self.tech_num
                if semantic_feature is not None:
                    semantic_feature_ = F.interpolate(semantic_feature, size=(h, w), mode='bicubic', align_corners=False).flatten(2).permute(2, 0, 1)
                    tmp_feat = torch.cat([tmp_feat, semantic_feature_], dim=-1)
                if mask_feature is not None:
                    mask_feature_ = F.interpolate(mask_feature, size=(h, w), mode='bicubic', align_corners=False).flatten(2).permute(2, 0, 1)
                    tmp_feat = torch.cat([tmp_feat, mask_feature_], dim=-1)

            tmp_feat = self.fusion_before_attn[i](tmp_feat) + res_feat
            tmp_feat = self.attn_pools[i](tmp_feat).mean(dim=0)  # B,c
            tmp_feat = self.score_linears[i](tmp_feat)  # B,1
            token_feat_list.append(tmp_feat)

        token_feat = torch.cat(token_feat_list, dim=1) + global_scores  # B,d
        # token_feat = torch.cat(token_feat_list, dim=1)
        local_weighted_score = token_feat * weight  # Uncomment if weight before final MLP else comment
        weighted_score = local_weighted_score

        out_score = self.net(weighted_score)  # B,1
        return out_score

