import torch
import torch.nn as nn
from mri_tools import  ifft2,fft2

class Memc_LOUPE(nn.Module):
    def __init__(self, input_shape, slope, sample_slope, device, sparsity):
        super(Memc_LOUPE, self).__init__()
        # self.gt=gt
        self.input_shape = input_shape
        self.slope = 5  #、？？
        self.device = device
        self.add_weight_real = nn.Parameter(- torch.log(1. / torch.rand(self.input_shape, dtype=torch.float32) - 1.) / self.slope, requires_grad=True)        
        self.add_weight_imag = nn.Parameter(- torch.log(1. / torch.rand(self.input_shape, dtype=torch.float32) - 1.) / self.slope, requires_grad=True)
        self.sample_slope = 200
        # self.sparsity = sparsity
        self.sparsity = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        # self.noise_val = noise_val
        self.conv = nn.Conv2d(4, 2, 1, 1, 0)

    def calculate_Mask(self, kspace_mc, option):

        logit_weights_real = 0 * kspace_mc[:,0, :, :] + self.add_weight_real
        logit_weights_imag = 0 * kspace_mc[:,1, :, :] + self.add_weight_imag
        prob_mask_tensor = torch.cat((logit_weights_real, logit_weights_imag), dim=1)
        prob_mask_tensor = self.conv(prob_mask_tensor)
        prob_mask_tensor = torch.sigmoid(self.slope * prob_mask_tensor)
       
        # print('self.sparsity:',self.sparsity,'self.sparsity.grad:',self.sparsity.grad)
        xbar = torch.mean(prob_mask_tensor)
        r = self.sparsity / xbar
        beta = (1 - self.sparsity) / (1 - xbar)
        le = (torch.less_equal(r, 1)).to(dtype=torch.float32)
        prob_mask_tensor = le * prob_mask_tensor * r + (1 - le) * (1 - (1 - prob_mask_tensor) * beta)

        threshs = torch.rand(prob_mask_tensor.size(), dtype=torch.float32).to(device=self.device)
        thresh_tensor = 0 * prob_mask_tensor + threshs
        # print('option:',option)
        if option:
            last_tensor_mask = torch.sigmoid(self.sample_slope * (prob_mask_tensor - thresh_tensor))
        else:
            last_tensor_mask = (prob_mask_tensor > thresh_tensor) + 0

        return last_tensor_mask.to(device=self.device)
    ##交集部分的前向传播的实现
    # def forward(self,mask,gt,overmask):
    #     B,C,H,W=gt.shape
    #     assert B==1 and C==2 and H==256 and W==256
    #     kspace_mc=fft2(gt)
 
    #     dcmask = self.calculate_Mask(kspace_mc, option=True)#inital work
    #     dcmask=dcmask*mask
    #     loss_mask=mask-dcmask

    #     #将交集部分置为1
    #     # print('overlap:',overlap)dcmask
    #     dcmask[overmask==1] = 1
    #     loss_mask[overmask==1] = 1

    #     return loss_mask, dcmask

    def forward(self,mask,gt,overlap):
        B,C,H,W=gt.shape
        assert B==1 and C==2 and H==256 and W==256
        kspace_mc=fft2(gt)
 
        dcmask = self.calculate_Mask(kspace_mc, option=True)#inital work
        dcmask=dcmask*mask
        loss_mask=mask-dcmask

        #将部分中心数据置为1  没有交集
        # print('overlap:',overlap)dcmask
        # dcmask[:,:,128 - overlap//2:128 + overlap//2,128 - overlap//2:128 + overlap//2] = 1
        # loss_mask[:,:,128 - overlap//2:128 + overlap//2,128 - overlap//2:128 + overlap//2] = 1

        return loss_mask, dcmask
