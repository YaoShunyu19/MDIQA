"""TOP-IQ metric, proposed by

TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment.
Chaofeng Chen, Jiadi Mo, Jingwen Hou, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, Weisi Lin.
Arxiv 2023.

Paper link: https://arxiv.org/abs/2308.03060

"""

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
from .clip_model import load
from .topiq_swin import create_swin

from MDIQA.utils.registry import ARCH_REGISTRY
from MDIQA.archs.topiq_arch import CFANet, GatedConv, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


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


class CFANet_Backbone(nn.Module):
    def __init__(self,
                 setting=1,
                 semantic_model_name='resnet50',
                 model_name='cfanet_nr_koniq_res50',
                 backbone_pretrain=True,
                 inter_dim=256,
                 num_heads=4,
                 num_attn_layers=1,
                 dprate=0.1,
                 activation='gelu',
                 pretrained=True,
                 pretrained_model_path=None,
                 default_mean=IMAGENET_DEFAULT_MEAN,
                 default_std=IMAGENET_DEFAULT_STD,
                 ):
        super().__init__()
        self.setting = setting
        self.model_name = model_name
        self.semantic_model_name = semantic_model_name
        self.semantic_level = -1

        # =============================================================
        # define semantic backbone network
        # =============================================================

        if 'swin' in semantic_model_name:
            self.semantic_model = create_swin(semantic_model_name, pretrained=True, drop_path_rate=0.0)
            feature_dim = self.semantic_model.num_features
            feature_dim_list = [int(self.semantic_model.embed_dim * 2 ** i) for i in range(self.semantic_model.num_layers)]
            feature_dim_list = feature_dim_list[1:] + [feature_dim]
        elif 'clip' in semantic_model_name:
            semantic_model_name = semantic_model_name.replace('clip_', '')
            self.semantic_model = [load(semantic_model_name, 'cpu')]
            feature_dim_list = self.semantic_model[0].visual.feature_dim_list
            default_mean, default_std = OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
        else:
            self.semantic_model = timm.create_model(semantic_model_name, pretrained=backbone_pretrain, features_only=True)
            # self.semantic_model = timm.create_model(semantic_model_name, pretrained=False, features_only=True, checkpoint_path='/root/.cache/huggingface/hub/models--timm--mobilenetv3_large_100.ra_in1k')
            feature_dim_list = self.semantic_model.feature_info.channels()
            feature_dim = feature_dim_list[self.semantic_level]
            self.fix_bn(self.semantic_model)
        self.feature_dim_list = feature_dim_list
        
        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)

        # =============================================================
        # define self-attention and cross scale attention blocks
        # =============================================================

        sa_layers = num_attn_layers 

        act_layer = nn.GELU() if activation == 'gelu' else nn.ReLU()
        dim_feedforward = min(4 * inter_dim, 2048)

        # gated local pooling and self-attention
        tmp_layer = TransformerEncoderLayer(inter_dim, nhead=num_heads, dim_feedforward=dim_feedforward, normalize_before=True, dropout=dprate, activation=activation)
        
        if self.setting in [2, 3]:
            self.dim_reduce = nn.ModuleList()
            self.weight_pool = nn.ModuleList()
        if self.setting in [3]:
            self.sa_attn_blks = nn.ModuleList()

        for idx, dim in enumerate(feature_dim_list):
            if self.setting in [2, 3]:
                self.weight_pool.append(GatedConv(dim))
                    
                self.dim_reduce.append(nn.Sequential(
                    nn.Conv2d(dim, inter_dim, 1, 1),
                    act_layer,
                    )
                )
            if self.setting in [3]:
                self.sa_attn_blks.append(TransformerEncoder(tmp_layer, sa_layers))

        if self.setting in [2, 3]:
            self._init_linear(self.dim_reduce)
        if self.setting in [3]:
            self._init_linear(self.sa_attn_blks)
            self.h_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 32, 1))
            self.w_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 1, 32))
            nn.init.trunc_normal_(self.h_emb.data, std=0.02)
            nn.init.trunc_normal_(self.w_emb.data, std=0.02)


        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True, weight_keys='params')
        elif pretrained:
            load_pretrained_network(self, default_model_urls[model_name], False, weight_keys='params')
            
        self.eps = 1e-8

    def get_feature_dim_list(self):
        return self.feature_dim_list
    
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
        
    def get_swin_feature(self, model, x):
        b, c, h, w = x.shape
        x = model.patch_embed(x)
        if model.absolute_pos_embed is not None:
            x = x + model.absolute_pos_embed
        x = model.pos_drop(x)
        feat_list = []
        for ly in model.layers:
            x = ly(x)
            feat_list.append(x)

        h, w = h // 8, w // 8
        for idx, f in enumerate(feat_list):
            feat_list[idx] = f.transpose(1, 2).reshape(b, f.shape[-1], h, w)
            if idx < len(feat_list) - 2: 
                h, w = h // 2, w // 2

        return feat_list
    
    def forward_cross_attention(self, x):

        x = self.preprocess(x)

        if 'swin' in self.semantic_model_name:
            dist_feat_list = self.get_swin_feature(self.semantic_model, x)
            self.semantic_model.eval()
        elif 'clip' in self.semantic_model_name:
            visual_model = self.semantic_model[0].visual.to(x.device)
            dist_feat_list = visual_model.forward_features(x)
        else:
            self.fix_bn(self.semantic_model)
            self.semantic_model.eval()
            dist_feat_list = self.semantic_model(x)
        
        start_level = 0
        end_level = len(dist_feat_list) 

        b, c, th, tw = dist_feat_list[end_level - 1].shape
        if self.setting in [3]:
            pos_emb = torch.cat((self.h_emb.repeat(1, 1, 1, self.w_emb.shape[3]), self.w_emb.repeat(1, 1, self.h_emb.shape[2], 1)), dim=1) 

        # backbone feature 1
        token_feat_list = dist_feat_list

        if self.setting in [2, 3]:
            token_feat_list = []
            for i in range(start_level, end_level):
                tmp_dist_feat = dist_feat_list[i]
                
                # backbone feature 2/3  gated local pooling
                tmp_feat = self.weight_pool[i](tmp_dist_feat)

                if tmp_feat.shape[2] > th and tmp_feat.shape[3] > tw: 
                    tmp_feat = F.adaptive_avg_pool2d(tmp_feat, (th, tw))

                tmp_feat = self.dim_reduce[i](tmp_feat)
                
                if self.setting in [3]:
                    # backbone feature 3  self attention
                    tmp_pos_emb = F.interpolate(pos_emb, size=tmp_feat.shape[2:], mode='bicubic', align_corners=False)
                    tmp_pos_emb = tmp_pos_emb.flatten(2).permute(2, 0, 1)
                    
                    tmp_feat = tmp_feat.flatten(2).permute(2, 0, 1)
                    tmp_feat = tmp_feat + tmp_pos_emb

                    tmp_feat = self.sa_attn_blks[i](tmp_feat)

                token_feat_list.append(tmp_feat)
        
        # setting 1: [B,di,hi,wi]
        # setting 2: [B,d,h,w]
        # setting 3: [hxw,B,d]
        return token_feat_list, th, tw
    
    def forward(self, x):
        backbone_features, th, tw = self.forward_cross_attention(x)        
        return backbone_features, th, tw


class CFANet_Head(nn.Module):
    def __init__(self,
                 setting=1,
                 model_name='cfanet_nr_koniq_res50',
                 feature_dim_list=[64, 256, 512, 1024, 2048],
                 inter_dim=256,
                 num_heads=4,
                 num_attn_layers=1,
                 dprate=0.1,
                 activation='gelu',
                 pretrained=True,
                 pretrained_model_path=None,
                 ):
        super().__init__()
        self.setting = setting
        self.model_name = model_name

        # =============================================================
        # define self-attention and cross scale attention blocks
        # =============================================================

        ca_layers = sa_layers = num_attn_layers 

        self.act_layer = nn.GELU() if activation == 'gelu' else nn.ReLU()
        dim_feedforward = min(4 * inter_dim, 2048)

        # gated local pooling and self-attention
        tmp_layer = TransformerEncoderLayer(inter_dim, nhead=num_heads, dim_feedforward=dim_feedforward, normalize_before=True, dropout=dprate, activation=activation)
        
        if self.setting in [1]:
            self.weight_pool = nn.ModuleList()
            self.dim_reduce = nn.ModuleList()
        if self.setting in [1, 2]:
            self.sa_attn_blks = nn.ModuleList()

        for idx, dim in enumerate(feature_dim_list):
            if self.setting in [1]:
                self.weight_pool.append(GatedConv(dim))
                    
                self.dim_reduce.append(nn.Sequential(
                    nn.Conv2d(dim, inter_dim, 1, 1),
                    self.act_layer,
                    )
                )
            if self.setting in [1, 2]:
                self.sa_attn_blks.append(TransformerEncoder(tmp_layer, sa_layers))


        # cross scale attention
        self.attn_blks = nn.ModuleList()
        tmp_layer = TransformerDecoderLayer(inter_dim, nhead=num_heads, dim_feedforward=dim_feedforward, normalize_before=True, dropout=dprate, activation=activation)
        for i in range(len(feature_dim_list) - 1):
            self.attn_blks.append(TransformerDecoder(tmp_layer, ca_layers))

        # attention pooling and MLP layers 
        self.attn_pool = TransformerEncoderLayer(inter_dim, nhead=num_heads, dim_feedforward=dim_feedforward, normalize_before=True, dropout=dprate, activation=activation)

        linear_dim = inter_dim
        self.score_linear = [
            nn.LayerNorm(linear_dim),
            nn.Linear(linear_dim, linear_dim),
            self.act_layer,
            nn.LayerNorm(linear_dim),
            nn.Linear(linear_dim, linear_dim),
            self.act_layer,
            nn.Linear(linear_dim, 1),
        ]

        self.score_linear = nn.Sequential(*self.score_linear)
        

        if self.setting in [1]:
            self._init_linear(self.dim_reduce)
        if self.setting in [1, 2]:
            self.h_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 32, 1))
            self.w_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 1, 32))
            nn.init.trunc_normal_(self.h_emb.data, std=0.02)
            nn.init.trunc_normal_(self.w_emb.data, std=0.02)
            self._init_linear(self.sa_attn_blks)
        
        self._init_linear(self.attn_blks)
        self._init_linear(self.attn_pool)

        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True, weight_keys='params')
        elif pretrained:
            load_pretrained_network(self, default_model_urls[model_name], False, weight_keys='params')
            
        self.eps = 1e-8
    
    def _init_linear(self, m):
        for module in m.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                nn.init.constant_(module.bias.data, 0)

    def forward_cross_attention(self, backbone_features, th, tw):
        # setting 1: [B,di,hi,wi]
        # setting 2: [B,d,h,w]
        # setting 3: [hxw,B,d]

        dist_feat_list = backbone_features
        start_level = 0
        end_level = len(dist_feat_list) 

        # setting 3
        token_feat_list = backbone_features

        if self.setting in [1, 2]:
            pos_emb = torch.cat((self.h_emb.repeat(1, 1, 1, self.w_emb.shape[3]), self.w_emb.repeat(1, 1, self.h_emb.shape[2], 1)), dim=1) 
            token_feat_list = []
            for i in reversed(range(start_level, end_level)):
                tmp_feat = dist_feat_list[i]

                if self.setting in [1]:
                    # setting 1  gated local pooling
                    tmp_feat = self.weight_pool[i](tmp_feat)

                    if tmp_feat.shape[2] > th and tmp_feat.shape[3] > tw: 
                        tmp_feat = F.adaptive_avg_pool2d(tmp_feat, (th, tw))

                    tmp_feat = self.dim_reduce[i](tmp_feat)

                # setting 1/2  self attention
                tmp_pos_emb = F.interpolate(pos_emb, size=tmp_feat.shape[2:], mode='bicubic', align_corners=False)
                tmp_pos_emb = tmp_pos_emb.flatten(2).permute(2, 0, 1)
                
                tmp_feat = tmp_feat.flatten(2).permute(2, 0, 1)
                
                tmp_feat = tmp_feat + tmp_pos_emb
                
                tmp_feat = self.sa_attn_blks[i](tmp_feat)
                
                token_feat_list.append(tmp_feat)
        
        # high level -> low level: coarse to fine 
        query = token_feat_list[0]
        query_list = [query]
        for i in range(len(token_feat_list) - 1):
            key_value = token_feat_list[i + 1] 
            query = self.attn_blks[i](query, key_value)
            query_list.append(query)

        b = query.shape[1]
        query_feat = query.permute(1, 2, 0).reshape(b, -1, th, tw)
        final_token = self.attn_pool(query)  # hw,B,d
        out_score = self.score_linear(final_token.mean(dim=0))  # B,1

        return out_score, query_feat, final_token.mean(dim=0)
    
    def forward(self, backbone_features, th, tw):
        out_score, query_feat, final_token = self.forward_cross_attention(backbone_features, th, tw)        
        return out_score, query_feat, final_token


@ARCH_REGISTRY.register()
class CFANet_Shared(nn.Module):
    def __init__(self,
                 setting=1,
                 semantic_model_name='resnet50',
                 model_name='cfanet_nr_koniq_res50',
                 backbone_pretrain=True,
                 num_class=1,
                 num_crop=1,
                 crop_size=256,
                 inter_dim=256,
                 num_heads=4,
                 num_attn_layers=1,
                 dprate=0.1,
                 activation='gelu',
                 pretrained=True,
                 pretrained_model_path=None,
                 default_mean=IMAGENET_DEFAULT_MEAN,
                 default_std=IMAGENET_DEFAULT_STD,
                 ):
        super().__init__()
        self.num_class = num_class

        pretrained_backbone_path = None
        pretrained_heads_path = [None] * self.num_class
        if isinstance(pretrained_model_path, list):
            pretrained_backbone_path = pretrained_model_path[0]
            pretrained_heads_path = pretrained_model_path[1:]
        
        self.cfanet_backbone = CFANet_Backbone(setting=setting,
                                               semantic_model_name=semantic_model_name,
                                               model_name=model_name,
                                               backbone_pretrain=backbone_pretrain,
                                               inter_dim=inter_dim,
                                               num_heads=num_heads,
                                               num_attn_layers=num_attn_layers,
                                               dprate=dprate,
                                               activation=activation,
                                               pretrained=pretrained,
                                               pretrained_model_path=pretrained_backbone_path,
                                               default_mean=default_mean,
                                               default_std=default_std
                                               )
        
        self.cfanet_heads = nn.ModuleList([
            CFANet_Head(setting=setting,
                        model_name=model_name,
                        feature_dim_list=self.cfanet_backbone.get_feature_dim_list(),
                        inter_dim=inter_dim,
                        num_heads=num_heads,
                        num_attn_layers=num_attn_layers,
                        dprate=dprate,
                        activation=activation,
                        pretrained=pretrained,
                        pretrained_model_path=pretrained_heads_path[i])
         for i in range(num_class)])
        
        if isinstance(pretrained_model_path, str):
            load_pretrained_network(self, pretrained_model_path, True, weight_keys='params')

    def forward(self, x, idx=None):
        backbone_features, th, tw = self.cfanet_backbone(x)
        out_score_list, query_feat_list, final_token_list = [], [], []
        if idx is None:
            for head in self.cfanet_heads:
                out_score, query_feat, final_token = head(backbone_features, th, tw)
                out_score_list.append(out_score)
                query_feat_list.append(query_feat)
                final_token_list.append(final_token)
            # [B,1], [B,d,h,w], [B,d]
            return out_score_list, query_feat_list, final_token_list
        else:
            out_score, query_feat, final_token = self.cfanet_heads[idx](backbone_features, th, tw)
            # B,1, B,d,h,w, B,d
            return out_score, query_feat, final_token

    def freeze_backbone(self):
        for name, p in self.named_parameters():
            if 'cfanet_backbone' in name:
                p.requires_grad = False

    def get_backbone(self):
        return self.cfanet_backbone
    
    def get_heads(self, idx=None):
        if idx is None:
            return self.cfanet_heads
        else:
            return self.cfanet_heads[idx]
