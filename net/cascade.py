import torch
import torch.nn as nn
import numpy as np
 
import sys
sys.path.append("/home/liuchun/Desktop/add_dc/dual_domain/Networks")
from .dudor_gen import get_generator
from dataprocess import imsshow
from mri_tools import  ifft2,fft2

def image2kspace(x):
    if isinstance(x, np.ndarray):
        x = np.fft.ifftshift(x, axes=(-2, -1))
        x = np.fft.fft2(x)
        x = np.fft.fftshift(x, axes=(-2, -1))
        return x
    elif isinstance(x, torch.Tensor):
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.fft2(x)
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x
    else:
        raise RuntimeError("Unsupported type.")

def pseudo2complex(x):
    """
    :param x:  [..., C=2, H, W]
    :return: [..., H, W] Complex
    """
    return x[..., 0, :, :] + x[..., 1, :, :] * 1j  #x[..., 0, :, :] + x[..., 1, :, :] * 1j

def kspace2image(x):
    
    if isinstance(x, np.ndarray):
        x = np.fft.ifftshift(x, axes=(-2, -1))
        x = np.fft.ifft2(x)
        x = np.fft.fftshift(x, axes=(-2, -1))
        return x
    elif isinstance(x, torch.Tensor):
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.ifft2(x)
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x
    else:
        raise RuntimeError("Unsupported type.")
def complex2pseudo(x):
    """
    :param x: [..., H, W] Complex
    :return: [...., C=2, H, W]
    """
    if isinstance(x, np.ndarray):
        return np.stack([x.real, x.imag], axis=-3)
    elif isinstance(x, torch.Tensor):
        return torch.stack([x.real, x.imag], dim=-3)
    else:
        raise RuntimeError("Unsupported type.")

class DataConsistencyLayer(nn.Module):
    """
    This class support different types k-space data consistency
    """

    def __init__(self, is_data_fidelity=True):
        super().__init__()
        self.is_data_fidelity = is_data_fidelity

    def forward(self, im_recon, mask, k0):
        """
        set is_data_fidelity=True to complete the formulation
        
        :param k: input k-space (reconstructed kspace, 2D-Fourier transform of im) complex
        :param k0: initially sampled k-space complex
        :param mask: sampling pattern
        """
        # k=fft2(im_recon)
        k=complex2pseudo(image2kspace(pseudo2complex(im_recon)))
        k_dc = (1 - mask) * k + mask * k0  
        im_dc = complex2pseudo(kspace2image(pseudo2complex(k_dc)))  # [B, C=2, H, W] 
        # im_dc = ifft2(k_dc)  # [B, C=2, H, W]   
        return im_dc


class CascadeMRIReconstructionFramework(nn.Module):
    def __init__(self,  n_cascade: int):
        super().__init__()
        self.cnn = get_generator('DRDN')######使用新的
        # self.cnn=UNet(n_channels=2, n_classes=2, bilinear=True)
        self.n_cascade = n_cascade

        assert n_cascade > 0
        dc_layers = [DataConsistencyLayer() for _ in range(n_cascade)]
        self.dc_layers = nn.ModuleList(dc_layers)

    def forward(self,im_recon, mask,k0):
        
        B, C, H, W = im_recon.shape
        B, C, H, W = k0.shape
        assert C == 2
        assert (B,C, H, W) == tuple(mask.shape)
        for dc_layer in self.dc_layers:
             #1 256 256
            # before= torch.abs(pseudo2complex(ifft2(k0)))
            im_recon=self.cnn(im_recon)    
            im_recon = dc_layer(im_recon, mask, k0)
            # after= torch.abs(pseudo2complex(im_recon))#1 256 256 
            # sub=before-after 
            # # after=torch.abs(pseudo2complex(ifft2(k0))) 
            # im_show=torch.cat((before,after,sub),0)
            # imsshow(im_show.cpu().detach().numpy(),titles=['before','after','sub'],ncols=3,cmap='gray',is_colorbar=True)
            
        return im_recon



        # assert im_recon.dtype==
        # print('mask.shape:',mask.shape)#1 2 256 256
       
        
        #显示输出
        # mask= mask#1 256 256
        # im_recon= torch.abs(pseudo2complex(im_recon))#1 256 256
        # k0k= torch.abs(pseudo2complex(k0))#1 256 256
        # k02img=torch.abs(  pseudo2complex(ifft2(k0)))
        # im_show=torch.cat((torch.log(k0k),mask,im_recon,k02img),0)
        # imsshow(im_show.cpu().detach().numpy(),titles=['k0','mask','im_recon','k02img'],num_col=4,cmap='gray',is_colorbar=True)


        
        # im_recon =  complex2pseudo( kspace2image( pseudo2complex(k_und)))
      
