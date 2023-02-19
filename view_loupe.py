import torch
import torch.nn as nn


class Memc_LOUPE(nn.Module):
    def __init__(self, input_shape, slope, sample_slope, device, sparsity, noise_val):
        super(Memc_LOUPE, self).__init__()
        self.input_shape = input_shape
        self.slope = slope
        self.device = device
        self.add_weight_real = nn.Parameter(- torch.log(1. / torch.rand(self.input_shape, dtype=torch.float32) - 1.) / self.slope, requires_grad=True)
        self.add_weight_imag = nn.Parameter(- torch.log(1. / torch.rand(self.input_shape, dtype=torch.float32) - 1.) / self.slope, requires_grad=True)
        self.sample_slope = sample_slope
        self.sparsity = sparsity
        self.noise_val = noise_val
        self.conv = nn.Conv3d(32, 16, 1, 1, 0) #why 3d??

    def calculate_Mask(self, kspace_mc, option):
        logit_weights_real = 0 * kspace_mc[:, 0::2, :, :, :] + self.add_weight_real  #扩充维度，可学习参数同时更新？
        logit_weights_imag = 0 * kspace_mc[:, 1::2, :, :, :] + self.add_weight_imag
        prob_mask_tensor = torch.cat((logit_weights_real, logit_weights_imag), dim=1)
        prob_mask_tensor = self.conv(prob_mask_tensor) #感知机
        prob_mask_tensor = torch.sigmoid(self.slope * prob_mask_tensor) #激活函数的作用

        #实现概率值的约束?
        xbar = torch.mean(prob_mask_tensor)
        r = self.sparsity / xbar
        beta = (1 - self.sparsity) / (1 - xbar)
        le = (torch.less_equal(r, 1)).to(dtype=torch.float32)
        prob_mask_tensor = le * prob_mask_tensor * r + (1 - le) * (1 - (1 - prob_mask_tensor) * beta)  #（1-le)表示网络是1的位置 

        threshs = torch.rand(prob_mask_tensor.size(), dtype=torch.float32).to(device=self.device)
        thresh_tensor = 0 * prob_mask_tensor + threshs

        if option:
            last_tensor_mask = torch.sigmoid(self.sample_slope * (prob_mask_tensor - thresh_tensor)) #激活函数输出作为最终的结果
        else:
            last_tensor_mask = (prob_mask_tensor > thresh_tensor) + 0

        return last_tensor_mask.to(device=self.device)

    def get_noise(self, kspace):

        spower = torch.sum(kspace ** 2) / kspace.size
        npower = self.noise_val[0] / (1 - self.noise_val[0]) * spower
        noise = torch.randn(0, self.noise_val[1] ** 0.5, kspace.shape) * torch.sqrt(npower)

        return kspace + noise

    def forward(self, kspace_mc, option):

        Kspace = torch.zeros(1, 2, int(kspace_mc.size()[1] / 2), kspace_mc.size()[3], kspace_mc.size()[4]).to(device=self.device, dtype=torch.float32)

        rec_mid_img = torch.zeros(1, 2, int(kspace_mc.size()[1] / 2), kspace_mc.size()[3], kspace_mc.size()[4]).to(device=self.device, dtype=torch.float32)

        # Get Mask
        last_tensor_mask = self.calculate_Mask(kspace_mc, option=option)

        # Get Coil Under_sampled K_space
        real = kspace_mc[:, 0::2, :, :, :] * last_tensor_mask
        imag = kspace_mc[:, 1::2, :, :, :] * last_tensor_mask

        real = real.view(-1, kspace_mc.size()[3], kspace_mc.size()[4])
        imag = imag.view(-1, kspace_mc.size()[3], kspace_mc.size()[4])

        # Get Coil Under_sampled Image
        image_mc = torch.fft.ifft2(real[0::1, :, :] + 1j * imag[0::1, :, :]).to(dtype=torch.complex64)
        image_mc = image_mc.view(int(kspace_mc.size()[1] / 2), kspace_mc.size()[2], kspace_mc.size()[3], kspace_mc.size()[4])

        # coil combine
        combine_image = torch.sqrt(torch.sum(torch.square(torch.abs(image_mc[0::1, :, :, :])), dim=1))

        # Sensitivity Map
        SensitivityMap = image_mc / (combine_image.view(int(kspace_mc.size()[1] / 2), 1, kspace_mc.size()[3], kspace_mc.size()[4]))

        # Input K_space
        Input_K = torch.fft.fft2(combine_image[0::1, :, :])

        Kspace[0, 0, :, :, :] = Input_K.real
        Kspace[0, 1, :, :, :] = Input_K.imag

        # Input Image
        Input_I = torch.fft.ifft2(Kspace[0, 0, 0::1, :, :] + 1j * Kspace[0, 1, 0::1, :, :])

        rec_mid_img[0, 0, :, :, :] = Input_I.real
        rec_mid_img[0, 1, :, :, :] = Input_I.imag

        return last_tensor_mask, Kspace, rec_mid_img, SensitivityMap
