import os
import pandas as pd
import torch
from torch.utils import data as data
import torchvision.transforms as tf
from PIL import Image
from pyiqa.data.transforms import transform_mapping, PairedToTensor
from pyiqa.data.general_nr_dataset import GeneralNRDataset
from pyiqa.utils import get_root_logger
from MDIQA.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class KonIQ(GeneralNRDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.get_transforms_aes(opt)

    def get_transforms_aes(self, opt):
        transform_list = []
        augment_dict = opt.get('augment_aes', None)
        if augment_dict is not None:
            for k, v in augment_dict.items():
                transform_list += transform_mapping(k, v)

            transform_list += [
                    PairedToTensor(),
                    ]
            self.trans_aes = tf.Compose(transform_list)
        else:
            self.trans_aes = None

    def __getitem__(self, index):

        img_path = self.paths_mos[index][0]
        mos_label = float(self.paths_mos[index][1])
        img_pil = Image.open(img_path).convert('RGB')

        img_tensor = self.trans(img_pil)
        if self.trans_aes is None:
            img_tensor_aes = img_tensor
        else:
            img_tensor_aes = self.trans_aes(img_pil)
        
        img_tensor = img_tensor * self.img_range
        img_tensor_aes = img_tensor_aes * self.img_range
        mos_label_tensor = torch.Tensor([mos_label])
        
        return {'img': img_tensor, 'img_aes': img_tensor_aes, 'mos_label': mos_label_tensor, 'img_path': img_path}
