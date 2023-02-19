import random
import pathlib
import scipy.io as sio
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from utils import normalize_zero_to_one,kspace2image, image2kspace, complex2pseudo, pseudo2real, pseudo2complex
 
from torch.utils import data as Data 
from dataprocess import complex2pseudo,kspace2image

def build_loader(dataset, batch_size,is_shuffle,    
                 num_workers=4):
    loader=Data.DataLoader(dataset, batch_size=batch_size , shuffle=is_shuffle, num_workers=num_workers)
    return loader

class IXIData(Dataset):
    def __init__(self, data_path, u_mask_path, s_mask_up_path, s_mask_down_path):
        super(IXIData, self).__init__()
        self.data_path = data_path
        self.u_mask_path = u_mask_path
        self.s_mask_up_path = s_mask_up_path
        self.s_mask_down_path = s_mask_down_path
        # self.s_mask_over_path = '/home/liuchun/Desktop/ovlm_parallel_02/mask/selecting_mask/mask_over.mat'

        self.examples = []
        
        data_dict = np.load(data_path)
        # loading dataset
        kspace = data_dict['kspace']  # List[ndarray]
        self.images=kspace2image(kspace)
        self.images=complex2pseudo(self.images)
        self.examples = self.images.astype(np.float32)

        self.mask_under = np.array(sio.loadmat(self.u_mask_path)['mask'])
        self.s_mask_up = np.array(sio.loadmat(self.s_mask_up_path)['mask'])
        self.s_mask_down = np.array(sio.loadmat(self.s_mask_down_path)['mask'])
        #new
        # self.over_mask = np.array(sio.loadmat(self.s_mask_over_path)['data'])

        self.mask_net_up = self.mask_under * self.s_mask_up
        self.mask_net_down = self.mask_under * self.s_mask_down

        self.mask_under = np.stack((self.mask_under, self.mask_under), axis=-1)
        self.mask_under = torch.from_numpy(self.mask_under).float()
        self.mask_net_up = np.stack((self.mask_net_up, self.mask_net_up), axis=-1)
        self.mask_net_up = torch.from_numpy(self.mask_net_up).float()
        self.mask_net_down = np.stack((self.mask_net_down, self.mask_net_down), axis=-1)
        self.mask_net_down = torch.from_numpy(self.mask_net_down).float()
        #new
        # self.over_mask = np.stack((self.over_mask, self.over_mask), axis=-1)
        # self.over_mask = torch.from_numpy(self.over_mask).float()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        label = self.examples[item]
        label = normalize_zero_to_one(label, eps=1e-6)  #归一化部分 是否需要额外进行归一化？ 
        label = torch.from_numpy(label)

        return label, self.mask_under,self.mask_net_up, self.mask_net_down #, file.name, slice_id
