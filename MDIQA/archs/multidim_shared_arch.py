import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from MDIQA.archs.topiq_backbone_head_arch import CFANet_Shared
from MDIQA.archs.topiq_shared_fusion_arch import MultiDimFusionWeightedMLP

class MultiDimSharedIQAModel(nn.Module):
    def __init__(self):
        super().__init__()
        technology_pretrained_model_path = ['/root/ysy/IQA-PyTorch/local_training_20240727/experiments/shared/CFANet_shared_fix_spaq_5_tech_backboneiter1500_setting2_load_unshared_sharpness_init/models/net_freeze_backbone.pth',
                                            '/root/ysy/IQA-PyTorch/local_training_20240727/experiments/shared/CFANet_shared_fix_spaq_5_tech_backboneiter1500_setting2_load_unshared_sharpness_init/models/net_best_koniq10k_val_head0_head0.pth',
                                            '/root/ysy/IQA-PyTorch/local_training_20240727/experiments/shared/CFANet_shared_fix_spaq_5_tech_backboneiter1500_setting2_load_unshared_sharpness_init/models/net_best_koniq10k_val_head1_head1.pth',
                                            '/root/ysy/IQA-PyTorch/local_training_20240727/experiments/shared/CFANet_shared_fix_spaq_5_tech_backboneiter1500_setting2_load_unshared_sharpness_init/models/net_best_koniq10k_val_head2_head2.pth',
                                            '/root/ysy/IQA-PyTorch/local_training_20240727/experiments/shared/CFANet_shared_fix_spaq_5_tech_backboneiter1500_setting2_load_unshared_sharpness_init/models/net_best_koniq10k_val_head3_head3.pth',
                                            '/root/ysy/IQA-PyTorch/local_training_20240727/experiments/shared/CFANet_shared_fix_spaq_5_tech_backboneiter1500_setting2_load_unshared_sharpness_init/models/net_best_koniq10k_val_head4_head4.pth']
        self.technology_shared_model = CFANet_Shared(setting=2,
                                                     num_class=5,
                                                     semantic_model_name='resnet50',
                                                     pretrained_model_path=technology_pretrained_model_path)
        
        aesthetic_pretrained_model_path = '/root/ysy/IQA-PyTorch/local_training_20240727/experiments/shared/CFANet_shared_fix_para_4_aes_setting2/models/net_best_PARA_val_color.pth'
        self.aesthetic_shared_model = CFANet_Shared(setting=2,
                                                    num_class=4,
                                                    semantic_model_name='resnet50',
                                                    pretrained_model_path=aesthetic_pretrained_model_path)
        
        self.multidim_fusion_net = MultiDimFusionWeightedMLP(num_dim=9,
                                                             dataset_specific_weighted=False,
                                                             instance_specific_weighted=True,
                                                             local_score=False,
                                                             semantic_feat='without',
                                                             mask_feat=False)
        
        multidim_fusion_net_pretrained_model_path = '/root/ysy/IQA-PyTorch/local_training_20240727/experiments/shared/CFANet_shared_fix_5t_finetune2_4a_setting2_koniq_insweight_fixaesprocess/models/net_best.pth'
        self.multidim_fusion_net.load_state_dict(torch.load(multidim_fusion_net_pretrained_model_path)['params'], strict=True)

    def forward(self, x):
        self.eval()

        tech_global_score_list, tech_local_score_list = self.technology_shared_model(x)
        aes_global_score_list, aes_local_score_list = self.aesthetic_shared_model(x)

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

        output = self.multidim_fusion_net(x, None, scores, None)
        return output


if __name__ == '__main__':
    device = 'cuda:0'
    model = MultiDimSharedIQAModel().to(device)
    img_path = '/root/ysy/IQA-PyTorch/datasets/koniq10k/512x384/10043785683.jpg'
    img_pil = Image.open(img_path).convert('RGB')
    img = torch.from_numpy(np.array(img_pil)).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.
    print(img.shape)
    score = model(img)
    print(score)