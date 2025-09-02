import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from basicsr.losses.topiq_base_shared_arch import CFANet_Shared
from basicsr.losses.topiq_base_unshared_arch import LocalScoreModule_v4, InstanceSpecificWeightedBranchv2
import open_clip
from torchvision import transforms
import math
import timm
import torch.nn.init as init
import pyiqa

from basicsr.utils.registry import LOSS_REGISTRY

_reduction_modes = ['none', 'mean', 'sum']


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
                 mask_feat=True,
                 ):
        super().__init__()
        self.act_layer = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.local_score = local_score
        if self.local_score is not False:
            if_semantic_feat = False if semantic_feat == 'without' else True
            self.local_score_module = LocalScoreModule_v4(num_dim, aes_num, semantic_feat=if_semantic_feat, mask_feature=mask_feat)

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
            self.instance_specific_weighted_branch = InstanceSpecificWeightedBranchv2(num_class=num_dim)

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
        elif 'dino' in self.semantic_feat:
            self.semantic_model = timm.create_model(self.semantic_feat, pretrained=True, num_classes=0)
            self.semantic_model = self.semantic_model.eval()
            data_config = timm.data.resolve_model_data_config(self.semantic_model)
            preprocess = timm.data.create_transform(**data_config, is_training=False)
            steps_to_retain = [transforms.Resize, transforms.CenterCrop, transforms.Normalize]
            new_preprocess = [t for t in preprocess.transforms if any(isinstance(t, step_type) for step_type in steps_to_retain)]
            self.semantic_preprocess = transforms.Compose(new_preprocess)
            self.semantic_feat_conv = nn.Conv2d(768, 256, 1, 1, 0)
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

    def forward(self, x, mask, scores, weight_ratio=None, weights_idx=0):
        # scores: B,d / [(B,c,h,w)*d]  d: num_dim
        global_scores, local_scores = scores[0], scores[1]

        semantic_feature = None
        with torch.no_grad():
            if 'clip' in self.semantic_feat:
                semantic_feature = self.semantic_model.encode_image(self.semantic_preprocess(x))[1:, :, :]
                w = int(math.sqrt(semantic_feature.shape[0]))
                semantic_feature = semantic_feature.reshape(w, w, x.shape[0], -1).permute(2, 3, 0, 1)  # B,d,h,w
                semantic_feature = self.semantic_feat_conv(semantic_feature)
            elif 'dino' in self.semantic_feat:
                semantic_feature = self.semantic_model.forward_features(self.semantic_preprocess(x))[:, 1:, :]
                w = int(math.sqrt(semantic_feature.shape[1]))
                semantic_feature = semantic_feature.reshape(x.shape[0], w, w, -1).permute(0, 3, 1, 2)  # B,d,h,w
                semantic_feature = self.semantic_feat_conv(semantic_feature)


        if self.local_score and self.instance_specific_weighted:
            weight = self.instance_specific_weighted_branch(x, weight_ratio=weight_ratio)  # B,d

            score = self.local_score_module(global_scores, local_scores, None, weight, semantic_feature)
        elif self.local_score and not self.instance_specific_weighted:
            weight = self.dataset_specific_weights[weights_idx][None, ...]  # 1,d
            score = self.local_score_module(global_scores, local_scores, None, weight, semantic_feature)
        elif not self.local_score and self.instance_specific_weighted:
            weight = self.instance_specific_weighted_branch(x, weight_ratio=weight_ratio)
            weighted_scores = global_scores * weight  # B,d * B,d
            score = self.net(weighted_scores)
        else:
            weight = self.dataset_specific_weights[weights_idx][None, ...]
            weighted_scores = global_scores * weight  # B,d * B,d
            score = self.net(weighted_scores)
        return score

    def get_weight(self, x, weight_ratio=None, weights_idx=0):
        with torch.no_grad():
            if self.local_score and self.instance_specific_weighted:
                weight = self.instance_specific_weighted_branch(x, weight_ratio=weight_ratio)  # B,d
            elif self.local_score and not self.instance_specific_weighted:
                weight = self.dataset_specific_weights[weights_idx][None, ...]  # 1,d
            elif not self.local_score and self.instance_specific_weighted:
                weight = self.instance_specific_weighted_branch(x, weight_ratio=weight_ratio)
            else:
                weight = self.dataset_specific_weights[weights_idx][None, ...]
            return weight



class MultiDimSharedIQAModel(nn.Module):
    def __init__(self,
                 multidim_fusion_net_pretrained_model_path,
                 weight_model_name='resnet50'):
        super().__init__()
        technology_pretrained_model_path = ['net_freeze_backbone.pth',
                                            'net_best_koniq10k_val_head0_head0.pth',
                                            'net_best_koniq10k_val_head1_head1.pth',
                                            'net_best_koniq10k_val_head2_head2.pth',
                                            'net_best_koniq10k_val_head3_head3.pth',
                                            'net_best_koniq10k_val_head4_head4.pth']
        self.technology_shared_model = CFANet_Shared(setting=2,
                                                     num_class=5,
                                                     semantic_model_name='resnet50',
                                                     pretrained_model_path=technology_pretrained_model_path)

        aesthetic_pretrained_model_path = 'net_best_PARA_val_color.pth'
        self.aesthetic_shared_model = CFANet_Shared(setting=2,
                                                    num_class=4,
                                                    semantic_model_name='resnet50',
                                                    pretrained_model_path=aesthetic_pretrained_model_path)

        self.multidim_fusion_net = MultiDimFusionWeightedMLP(num_dim=9,
                                                             dataset_specific_weighted=False,
                                                             instance_specific_weighted=True,
                                                             semantic_model_name=weight_model_name,
                                                             local_score='v4',  # false v4
                                                             semantic_feat='clip_RN50',  # without clip_RN50
                                                             out_act='softmax',
                                                             mask_feat=False)

        missing_keys, unexpected_keys = self.multidim_fusion_net.load_state_dict(torch.load(multidim_fusion_net_pretrained_model_path)['params'], strict=True)
        print('missing keys: ', missing_keys)
        print('unexpected keys: ', unexpected_keys)

    def forward(self, x, weight_ratio=None):
        self.eval()

        tech_global_score_list, tech_local_score_list, _ = self.technology_shared_model(x)
        aes_global_score_list, aes_local_score_list, _ = self.aesthetic_shared_model(x)

        global_score_list = []
        local_score_list = []
        for tgs in tech_global_score_list:
            global_score_list.append(tgs)
        for ags in aes_global_score_list:
            global_score_list.append(ags)
        for tls in tech_local_score_list:
            local_score_list.append(tls)
        for als in aes_local_score_list:
            local_score_list.append(als)

        global_scores = torch.cat(global_score_list, dim=1)  # B,(N1+N2+...)
        scores = [global_scores, local_score_list]

        output = self.multidim_fusion_net(x, None, scores, weight_ratio, None)
        return global_scores, output

    def forward_feature(self, x, weight_ratio=None, return_weight=False):
        self.eval()

        tech_global_score_list, tech_local_score_list, _ = self.technology_shared_model(x)
        aes_global_score_list, aes_local_score_list, _ = self.aesthetic_shared_model(x)

        global_score_list = []
        local_score_list = []  # [B,C,H,W]*N
        for tgs in tech_global_score_list:
            global_score_list.append(tgs)
        for ags in aes_global_score_list:
            global_score_list.append(ags)
        for tls in tech_local_score_list:
            local_score_list.append(tls)
        for als in aes_local_score_list:
            local_score_list.append(als)

        if return_weight:
            weight = self.multidim_fusion_net.get_weight(x, weight_ratio, None)  # B,N
            return local_score_list, weight
        return local_score_list


@LOSS_REGISTRY.register()
class MultiDimSharedNRIQALoss(nn.Module):
    def __init__(self,
                 multidim_fusion_net_pretrained_model_path,
                 weight_model_name='resnet50',
                 loss_weight=1.0,
                 reduction='mean',
                 weight_ratio=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                 ):
        super(MultiDimSharedNRIQALoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.weight_ratio = torch.tensor(weight_ratio, dtype=torch.float32)

        self.model = MultiDimSharedIQAModel(multidim_fusion_net_pretrained_model_path, weight_model_name=weight_model_name)
        self.model.eval()

    def forward(self, pred, target):
        self.model.eval()
        weight_ratio = self.weight_ratio.to(pred.device)
        _, score = self.model(pred, weight_ratio=weight_ratio)  # B,1
        score = torch.mean(score, dim=-1)  # B
        if self.reduction in ['none', 'mean']:
            score = torch.mean(score, dim=0)
        else:
            score = torch.sum(score, dim=0)
        return -1 * self.loss_weight * score


@LOSS_REGISTRY.register()
class MultiDimSharedFRIQALoss(nn.Module):
    def __init__(self,
                 multidim_fusion_net_pretrained_model_path,
                 weight_model_name='resnet50',
                 loss_weight=1.0,
                 reduction='mean',
                 weight_ratio=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]):
        super(MultiDimSharedFRIQALoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.weight_ratio = torch.tensor(weight_ratio, dtype=torch.float32)

        self.model = MultiDimSharedIQAModel(multidim_fusion_net_pretrained_model_path, weight_model_name=weight_model_name)
        self.model.eval()

        self.criterion = torch.nn.L1Loss()

    def forward(self, pred, target):
        self.model.eval()
        weight_ratio = self.weight_ratio.to(pred.device)
        local_features_list_pred, weight = self.model.forward_feature(pred, weight_ratio, True)  # [B,C,h,w]*N
        local_features_list_target = self.model.forward_feature(target.detach(), weight_ratio, False)  # [B,C,h,w]*N

        if weight_ratio is not None:
            weighted_local_features_list_pred = [local_features_list_pred[i] * weight_ratio[i] * weight[:, i, None, None, None] for i in range(len(local_features_list_pred))]
            weighted_local_features_list_target = [local_features_list_target[i] * weight_ratio[i] * weight[:, i, None, None, None] for i in range(len(local_features_list_target))]
        else:
            weighted_local_features_list_pred = [local_features_list_pred[i] * weight[:, i, None, None, None] for i in range(len(local_features_list_pred))]
            weighted_local_features_list_target = [local_features_list_target[i] * weight[:, i, None, None, None] for i in range(len(local_features_list_target))]

        local_features_pred = torch.cat(weighted_local_features_list_pred, dim=1)
        local_features_target = torch.cat(weighted_local_features_list_target, dim=1)

        percep_loss = self.criterion(local_features_pred, local_features_target)
        return self.loss_weight * percep_loss

