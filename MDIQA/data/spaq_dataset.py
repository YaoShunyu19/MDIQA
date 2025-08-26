import os
import pandas as pd
from torch.utils import data as data
import torchvision.transforms as tf
from PIL import Image
from pyiqa.data.transforms import transform_mapping, PairedToTensor
from pyiqa.utils import get_root_logger
from MDIQA.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SPAQ(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.dataroot = opt['dataroot']
        self.paths_list = self.get_meta_info(opt)
        self.get_transforms(opt)
        self.get_transforms_aes(opt)

    def get_transforms(self, opt):
        transform_list = []
        augment_dict = opt.get('augment', None)
        if augment_dict is not None:
            for k, v in augment_dict.items():
                transform_list += transform_mapping(k, v)

        self.img_range = opt.get('img_range', 1.0)
        transform_list += [
                PairedToTensor(),
                ]
        self.trans = tf.Compose(transform_list)

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

    def get_meta_info(self, opt):
        dataframe = pd.read_csv(opt['meta_info_file'])
        dist_name_list = list(dataframe['dist_name'])
        label_list = self.mos_normalize(list(dataframe[opt['label_type']]))

        paths_list = []
        for i in range(len(dataframe)):
            dist_path = os.path.join(self.dataroot, dist_name_list[i])
            paths_list.append([dist_path, label_list[i]])

        return paths_list
    
    def mos_normalize(self, mos):
        return [s / 100. for s in mos]

    def __len__(self):
        return len(self.paths_list)
    
    def __getitem__(self, index):
        dist_path = self.paths_list[index][0]
        dist_pil = Image.open(dist_path).convert('RGB')

        dist_tensor = self.trans(dist_pil)
        if self.trans_aes is None:
            dist_aes_tensor = dist_tensor
        else:
            dist_aes_tensor = self.trans_aes(dist_pil)

        dist_tensor = dist_tensor * self.img_range
        dist_aes_tensor = dist_aes_tensor * self.img_range
        label = self.paths_list[index][1]

        return {'dist': dist_tensor, 'dist_aes': dist_aes_tensor, 'label': label, 'dist_path': dist_path}
