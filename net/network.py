from net.net_parts import *
from .cascade import CascadeMRIReconstructionFramework
from mri_tools import  ifft2,fft2
from dataprocess import imsshow,pseudo2complex
from .memc_loupe import Memc_LOUPE
class ParallelNetwork(nn.Module):
    def __init__(self,method, rank,slope=5,sample_slope=200):
        super(ParallelNetwork, self).__init__()
        self.method=method
        self.rank = rank
        self.up_network = CascadeMRIReconstructionFramework(
            n_cascade=5  #the formor is 5
        )
        self.down_network = CascadeMRIReconstructionFramework(
            n_cascade=5  #the formor is 5
        )
        if(self.method=="loupe"):
            input_shape=[1,2,256,256]
            self.Memc_LOUPE_Model = Memc_LOUPE(input_shape, slope=slope, sample_slope=sample_slope, device=self.rank, sparsity=0.5)

    def forward(self,par1,par2,par3,par4,mode):
        if(self.method=="loupe"):
            mask=par1
            gt=par2
            overlap=par3
            option=par4
            if option:
                loss_mask, dc_mask=self.Memc_LOUPE_Model(mask,gt,overlap)
            else:
                loss_mask=dc_mask=mask  
            #形成统一的mask
            # if mode=='test':
            if mode != 'train':
                loss_mask=dc_mask=mask

            k0_up=fft2(gt)*dc_mask
            under_img_up=ifft2(k0_up)
            k0_down=fft2(gt)*loss_mask
            under_img_down=ifft2(k0_down)      
            output_up  = self.up_network(under_img_up, dc_mask,k0_up)
            output_down  = self.down_network(under_img_down, loss_mask,k0_down)
            return  output_up,output_down,dc_mask,loss_mask
        elif(self.method=="baseline"):
            under_img_up=par1
            mask_up=par2
            under_img_down=par3
            mask_down=par4
            k0_up=fft2(under_img_up)
            k0_down=fft2(under_img_down)
            output_up  = self.up_network(under_img_up, mask_up,k0_up)
            output_down  = self.down_network(under_img_down, mask_down,k0_down)
            return output_up,  output_down 
