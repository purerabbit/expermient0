from audioop import mul
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
import torchvision.utils
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.color import rgb2gray
import os
from functools import reduce
import operator
import skimage.metrics
def get_mask(u_mask_path):
    former_path_select='/home/liuchun/Desktop/0_experiment/mask/selecting_mask/' #选择mask路径前缀
    former_path_under='/home/liuchun/Desktop/0_experiment/mask/undersampling_mask/' #欠采mask路径前缀
    #简化传入参数
    if(u_mask_path=='vd'):
        u_mask_path=former_path_under+'vd_mask_under.mat'
        s_mask_up_path=former_path_select+'vd_mask_up.mat'
        s_mask_down_path=former_path_select+'vd_mask_down.mat'
    elif(u_mask_path=='c'):
        u_mask_path=former_path_under+'cartesian_mask_under.mat'
        s_mask_up_path=former_path_select+'cartesian_mask_up.mat'
        s_mask_down_path=former_path_select+'cartesian_mask_down.mat'

    elif(u_mask_path=='r'):
        u_mask_path=former_path_under+'random_mask_under.mat'
        s_mask_up_path=former_path_select+'random_mask_up.mat'
        s_mask_down_path=former_path_select+'random_mask_down.mat'

    else:
        u_mask_path='/home/liuchun/Desktop/0_experiment/mask/undersampling_mask/mask_4.00x_acs24.mat'
        s_mask_up_path='/home/liuchun/Desktop/0_experiment/mask/selecting_mask/mask_2.00x_acs16.mat'
        s_mask_down_path='/home/liuchun/Desktop/0_experiment/mask/selecting_mask/mask_2.50x_acs16.mat'
    return u_mask_path,s_mask_up_path,s_mask_down_path

def save_images(under_img,net_img_up,net_img_down,output_up,output_down,mask_under,mask_net_up,mask_net_down,out_mean,label,iter_num,path,mode):
        img_show=torch.cat((pseudo2real(under_img),pseudo2real(net_img_up),\
                    pseudo2real(net_img_down) ,pseudo2real(output_up),\
                    pseudo2real(output_down),pseudo2real(mask_under),\
                    pseudo2real(mask_net_up),pseudo2real(mask_net_down),pseudo2real(out_mean),\
                    pseudo2real(label)),0) 
        psnr_under=compute_psnr_q(pseudo2real(label),pseudo2real(under_img))
        ssim_under=compute_ssim(pseudo2real(label),pseudo2real(under_img))
        psnr_up=compute_psnr_q(pseudo2real(label),pseudo2real(net_img_up))
        ssim_up=compute_ssim(pseudo2real(label),pseudo2real(net_img_up))
        psnr_down=compute_psnr_q(pseudo2real(label),pseudo2real(net_img_down))
        ssim_down=compute_ssim(pseudo2real(label),pseudo2real(net_img_down))
        psnr_show_up=compute_psnr_q(pseudo2real(label),pseudo2real(output_up))
        ssim_show_up=compute_ssim(pseudo2real(label),pseudo2real(output_up))
        psnr_show_down=compute_psnr_q(pseudo2real(label),pseudo2real(output_down))
        ssim_show_down=compute_ssim(pseudo2real(label),pseudo2real(output_down))
        psnr_show_mean=compute_psnr_q(pseudo2real(label),pseudo2real(out_mean))
        ssim_show_mean=compute_ssim(pseudo2real(label),pseudo2real(out_mean))

        ratio_mask_under_real=torch.sum(mask_under[0,0,:,:])/(256*256)
        ratio_mask_under_imag=torch.sum(mask_under[0,1,:,:])/(256*256)
        ratio_mask_select_up_real=torch.sum(mask_net_up[0,0,:,:])/(256*256)
        ratio_mask_select_up_imag=torch.sum(mask_net_up[0,1,:,:])/(256*256)
        ratio_mask_select_down_real=torch.sum(mask_net_down[0,0,:,:])/(256*256)
        ratio_mask_select_down_imag=torch.sum(mask_net_down[0,1,:,:])/(256*256)
        if mode=='train':
            filename2save=f'/home/liuchun/Desktop/0_experiment/train_imgs/{path}'
        elif mode=='test':
            filename2save=f'/home/liuchun/Desktop/0_experiment/test_images/{path}'
        if not os.path.exists(filename2save):
            os.makedirs(filename2save)
        imsshow(img_show.data.cpu().numpy(),['underssim: {:.3f} underpsnr: {:.3f}'.format(ssim_under, psnr_under),\
                                            'sussim: {:.3f} supsnr: {:.3f}'.format(ssim_up, psnr_up),\
                                            'sdssim: {:.3f} sdpsnr: {:.3f}'.format(ssim_down, psnr_down),\
                                            'upssim: {:.3f} uppsnr: {:.3f}'.format(ssim_show_up, psnr_show_up),\
                                            'dssim: {:.3f} dpsnr: {:.3f}'.format(ssim_show_down, psnr_show_down),\
                                            'under-real:{:.3f} under-imag:{:.3f}'.format(ratio_mask_under_real, ratio_mask_under_imag),\
                                            'up-real:{:.3f} up-imag:{:.3f}'.format(ratio_mask_select_up_real, ratio_mask_select_up_imag),\
                                            'down-real:{:.3f} down-imag:{:.3f}'.format(ratio_mask_select_down_real, ratio_mask_select_down_imag),\
                                            'meanssim: {:.3f} meanpsnr: {:.3f}'.format(ssim_show_mean, psnr_show_mean),\
                                                'label'],ncols=5,cmap='gray',is_colorbar=True,filename2save=f'{filename2save}/{iter_num}.png')
        

def center_crop(data, shape):
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def normalize_zero_to_one(data, eps=0.):
    data_min = float(data.min())
    data_max = float(data.max())
    return (data - data_min) / (data_max - data_min + eps)


#####################################################################

def plot_loss(loss):
    plt.figure()
    plt.plot(loss)
    plt.show()
    plt.close('all')


def imgshow(im, cmap=None, rgb_axis=None, dpi=1000, figsize=(6.4, 4.8)):
    if isinstance(im, torch.Tensor):
        im = im.to('cpu').detach().cpu().numpy()
    if rgb_axis is not None:
        im = np.moveaxis(im, rgb_axis, -1)
        im = rgb2gray(im)

    plt.figure(dpi=dpi, figsize=figsize)
    norm_obj = Normalize(vmin=im.min(), vmax=im.max())
    plt.imshow(im, norm=norm_obj, cmap=cmap)
    plt.colorbar()
    plt.show()
    plt.close('all')

def imsshow(imgs, titles=None, ncols=5, dpi=100, cmap=None, is_colorbar=False, is_ticks=False,
            col_width=4, row_width=3, margin_ratio=0.1, n_images_max=50, filename2save=None, **imshow_kwargs):
    '''
    assume imgs is Sequence[ndarray[Nx, Ny]]
    '''
    num_imgs = len(imgs)

    if num_imgs > n_images_max:
        print(
            f"[WARNING] Too many images ({num_imgs}), clip to argument n_images_max({n_images_max}) for performance reasons.")
        imgs = imgs[:n_images_max]
        num_imgs = n_images_max

    if isinstance(cmap, list):
        assert len(cmap) == len(imgs)
    else:
        cmap = [cmap, ] * num_imgs

    nrows = math.ceil(num_imgs / ncols)

    # compute the figure size, compute necessary size first, then add margin
    figsize = (ncols * col_width, nrows * row_width)
    figsize = (figsize[0] * (1 + margin_ratio), figsize[1] * (1 + margin_ratio))
    fig = plt.figure(dpi=dpi, figsize=figsize)
    for i in range(num_imgs):
        ax = plt.subplot(nrows, ncols, i + 1)
        im = ax.imshow(imgs[i], cmap=cmap[i], **imshow_kwargs)
        if titles:
            plt.title(titles[i])
        if is_colorbar:
            cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.01, ax.get_position().height])
            plt.colorbar(im, cax=cax)
        if not is_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
    if filename2save is not None:
        fig.savefig(filename2save)
    else:
        plt.show()
    plt.close('all')

# def imsshow(imgs, titles=None, num_col=5, dpi=100, cmap=None, is_colorbar=False, is_ticks=False):
#     '''
#     assume imgs's shape is (Nslice, Nx, Ny)
#     '''
#     num_imgs = len(imgs)
#     num_row = math.ceil(num_imgs / num_col)
#     fig_width = num_col * 3
#     if is_colorbar:
#         fig_width += num_col * 1.5
#     fig_height = num_row * 3
#     fig = plt.figure(dpi=dpi, figsize=(fig_width, fig_height))
#     for i in range(num_imgs):
#         ax = plt.subplot(num_row, num_col, i + 1)
#         im = ax.imshow(imgs[i], cmap=cmap)
#         if titles:
#             plt.title(titles[i])
#         if is_colorbar:
#             cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.01, ax.get_position().height])
#             plt.colorbar(im, cax=cax)
#         if not is_ticks:
#             ax.set_xticks([])
#             ax.set_yticks([])
#     plt.show()
#     plt.close('all')


def make_grid_and_show(ims, nrow=5, cmap=None):
    if isinstance(ims, np.ndarray):
        ims = torch.from_numpy(ims)

    B, C, H, W = ims.shape
    grid_im = torchvision.utils.make_grid(ims, nrow=nrow)
    fig_h, fig_w = nrow * 2 + 1, (B / nrow) + 1
    imgshow(grid_im, cmap=cmap, rgb_axis=0, dpi=200, figsize=(fig_h, fig_w))


def int2preetyStr(num: int):
    s = str(num)
    remain_len = len(s)
    while remain_len - 3 > 0:
        s = s[:remain_len - 3] + ',' + s[remain_len - 3:]
        remain_len -= 3
    return s


def compute_num_params(module, is_trace=False):
    print(int2preetyStr(sum([p.numel() for p in module.parameters()])))
    if is_trace:
        for item in [f"[{int2preetyStr(info[1].numel())}] {info[0]}:{tuple(info[1].shape)}"
                     for info in module.named_parameters()]:
            print(item)


def tonp(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    else:
        return x


def pseudo2real(x):
    """
    :param x: [..., C=2, H, W]
    :return: [..., H, W]
    """
    return (x[..., 0, :, :] ** 2 + x[..., 1, :, :] ** 2) ** 0.5


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


def pseudo2complex(x):
    """
    :param x:  [..., C=2, H, W]
    :return: [..., H, W] Complex
    """
    return x[..., 0, :, :] + x[..., 1, :, :] * 1j


# ================================
# Preprocessing
# ================================
def minmax_normalize(x, eps=1e-8):
    min = x.min()
    max = x.max()
    return (x - min) / (max - min + eps)


# ================================
# kspace and image domain transform
# reference: [ismrmrd-python-tools/transform.py at master · ismrmrd/ismrmrd-python-tools · GitHub](https://github.com/ismrmrd/ismrmrd-python-tools/blob/master/ismrmrdtools/transform.py)
# ================================
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


# ======================================
# Metrics
# ======================================
def compute_mse(x, y):
    """
    REQUIREMENT: `x` and `y` can be any shape, but their shape have to be same
    """
    assert x.dtype == y.dtype and x.shape == y.shape, \
        'x and y is not compatible to compute MSE metric'

    if isinstance(x, np.ndarray):
        mse = np.mean(np.abs(x - y) ** 2)

    elif isinstance(x, torch.Tensor):
        mse = torch.mean(torch.abs(x - y) ** 2)

    else:
        raise RuntimeError(
            'Unsupported object type'
        )
    return mse

def compute_psnr_q(gt, pred, maxval=None):
    # gt1=minmax_normalize(gt)
    # pred1=minmax_normalize(pred)
    assert type(gt) == type(pred)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    max_val = gt.max() if maxval is None else maxval
    PSNR = peak_signal_noise_ratio(gt, pred, data_range=max_val)  
    return PSNR
def compute_ssim(pred, gt):
    # gt=minmax_normalize(gt2)
    # pred=minmax_normalize(pred2)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    # target_im=minmax_normalize(target_im)
    # reconstructed_im=minmax_normalize(reconstructed_im)
    ssim=structural_similarity(gt.squeeze(), pred.squeeze(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    return ssim 


# def compute_psnr_q(gt, pred, maxval=None):
#     gt=minmax_normalize(gt)
#     pred=minmax_normalize(pred)
#     assert type(gt) == type(pred)
#     if type(pred) is torch.Tensor:
#         gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
#     max_val = gt.max() if maxval is None else maxval
#     PSNR = peak_signal_noise_ratio(gt, pred, data_range=max_val)
    
#     return PSNR

# def compute_psnr(reconstructed_im, target_im, peak='normalized', is_minmax=False):
#     '''
#     Image must be of either Integer [0, 255] or Float value [0,1]
#     :param peak: 'max' or 'normalize', max_intensity will be the maximum value of target_im if peek == 'max.
#           when peek is 'normalized', max_intensity will be the maximum value depend on data representation (in this
#           case, we assume your input should be normalized to [0,1])
#     REQUIREMENT: `x` and `y` can be any shape, but their shape have to be same
#     '''
#     assert target_im.dtype == reconstructed_im.dtype and target_im.shape == reconstructed_im.shape, \
#         'target_im and reconstructed_im is not compatible to compute PSNR metric'
#     assert peak in {'max', 'normalized'}, \
#         'peak mode is not supported'

#     eps = 1e-8  # to avoid math error in log(x) when x=0  try 1e-16

#     if is_minmax:
#         reconstructed_im = minmax_normalize(reconstructed_im, eps)
#         target_im = minmax_normalize(target_im, eps)

#     if isinstance(target_im, np.ndarray):
#         max_intensity = 255 if target_im.dtype == np.uint8 else 1.0
#         max_intensity = np.max(target_im).item() if peak == 'max' else max_intensity
#         psnr = 20 * math.log10(max_intensity) - 10 * np.log10(compute_mse(reconstructed_im, target_im) + eps)

#     elif isinstance(target_im, torch.Tensor):
#         max_intensity = 255 if target_im.dtype == torch.uint8 else 1.0
#         max_intensity = torch.max(target_im).item() if peak == 'max' else max_intensity
#         psnr = 20 * math.log10(max_intensity) - 10 * torch.log10(compute_mse(reconstructed_im, target_im) + eps)

#     else:
#         raise RuntimeError(
#             'Unsupported object type'
#         )
#     return psnr

# def axes_to_last_order(ndim, axes_idx):
#     """
#     Generate axes order of move the axes specifed by `axes_idx` to last in a n-dimension axes,
#     return indices can used to permute(array, indices) and inverse_indices can used
#     to restore the data by permute(array, inverse_indices)
#     eg.
#         (Nb, Nx, Ny, Ntime) --move(2,3)--> (Nb, Ntime, Nx, Ny)
#         (..., Nx, Ny, Ntime, Nslice) --move(-4,-3)--> (..., Ntime, Nslice, Nx, Ny)
#         shape (1,2,3,4,5) --move(3,2)--> shape (1,2,5,4,3)
#     """
#     # convert reversed index to ordinal index
#     axes_need_move = [(axis if axis >= 0 else ndim + axis) for axis in axes_idx]

#     # generate forward-indices
#     forward_indices = tuple([axis for axis in range(ndim) if axis not in axes_need_move] + axes_need_move)

#     # generate inverse-indices
#     inverse_indices = np.argsort(forward_indices).tolist()

#     return forward_indices, inverse_indices

# def compute_psnr_q(gt, pred, maxval=None):
#     assert type(gt) == type(pred)
#     if type(pred) is torch.Tensor:
#         gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    
#     max_val = gt.max() if maxval is None else maxval
#     PSNR = peak_signal_noise_ratio(gt.squeeze(), pred.squeeze(), data_range=max_val)
    
#     return PSNR

# def compute_ssim(reconstructed_im, target_im, image_axes=(-2, -1)):
#     ssim1=compute_ssim1(reconstructed_im, target_im)
    # ssim2=compute_ssim2(reconstructed_im, target_im)
    # print('ssim1:',ssim1)
    # print('ssim2:',ssim2)
    # return ssim1

# def compute_ssim(reconstructed_im, target_im, image_axes=(-2, -1)):
#     target_im=minmax_normalize(target_im)
#     reconstructed_im=minmax_normalize(reconstructed_im)
#     target_im = target_im.detach().to('cpu').numpy()
#     reconstructed_im = reconstructed_im.detach().to('cpu').numpy()
#     ssim=structural_similarity(target_im.squeeze(), reconstructed_im.squeeze(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
#     return ssim 

# def compute_ssim(reconstructed_im, target_im, image_axes=(-2, -1)):
#     """
#     Compute structural similarity index between two batches using skimage library,
#     which only accept 2D-image input. We have to specify where is image's axes.

#     WARNING: this method using skimage's implementation, DOES NOT SUPPORT GRADIENT
#     """

#     assert target_im.dtype == reconstructed_im.dtype and target_im.shape == reconstructed_im.shape, \
#         'target_im and reconstructed_im is not compatible to compute SSIM metric'

#     if isinstance(target_im, np.ndarray):
#         pass
#     elif isinstance(target_im, torch.Tensor):
#         target_im = target_im.detach().to('cpu').numpy()
#         reconstructed_im = reconstructed_im.detach().to('cpu').numpy()
#     else:
#         raise RuntimeError(
#             'Unsupported object type'
#         )
#     # b c h w 
#     axes_indices, _ = axes_to_last_order(target_im.ndim, image_axes)

#     rec_im_seq = np.transpose(reconstructed_im, axes_indices).reshape(-1, target_im.shape[-2], target_im.shape[-1])
#     target_im_seq = np.transpose(target_im, axes_indices).reshape(-1, target_im.shape[-2], target_im.shape[-1])
#     sequence_length = len(target_im_seq)

#     ssim_acc = sum(
#         [structural_similarity(target_im_seq[i, :, :], rec_im_seq[i, :, :],
#                                gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
#          for i in range(sequence_length)]
#     )

#     return ssim_acc / sequence_length
# def compute_psnr_q(xs: torch.Tensor, ys: torch.Tensor) -> np.ndarray:
#     xs=xs.squeeze()
#     ys=ys.squeeze()

#     psnr = skimage.metrics.peak_signal_noise_ratio(
#             xs.cpu().numpy(),
#             ys.cpu().numpy(),
#             data_range=ys.cpu().numpy().max(),
#         )
#     return psnr
# def compute_mse_q(x, y, axis=None):
#     """
#     REQUIREMENT: `x` and `y` can be any shape, but their shape have to be same
#     """
#     assert x.dtype == y.dtype and x.shape == y.shape, \
#         'x and y is not compatible to compute MSE metric'

#     if axis:
#         mse = ((x - y) ** 2).mean(axis=axis)
#     else:
#         mse = ((x - y) ** 2).mean()

#     return mse

# def compute_psnr_q(x, y, peak='normalized', image_axes=(-2, -1), reduction='mean', eps=1e-16):
# def compute_psnr_q(x, y, peak='max', image_axes=(-2, -1), reduction='mean', eps=1e-16):
#     '''
#     Image must be of either Integer [0, 255] or Float value [0,1]
#     :param peak: 'max' or 'normalize', max_intensity will be the maximum value of target_im if peek == 'max.
#           when peek is 'normalized', max_intensity will be the maximum value depend on data representation (in this
#           case, we assume your input should be normalized to [0,1])
#     :param image_axes: specify the image's axes (H, W)
#     :param eps: to avoid math error in log(x) when x=0
#     REQUIREMENT: `x` and `y` is [..., H, W, ...], their shape have to be same
#     '''
#     # x=minmax_normalize(x)
#     # y=minmax_normalize(y)
#     assert len(x.shape) >= 2
#     assert y.dtype == x.dtype and y.shape == x.shape, \
#         f"shape of x {x.shape} and shape of y {y.shape} is not compatible to compute PSNR metric"
#     assert peak in {'max', 'normalized'}, \
#         'peak mode is not supported'

#     axes_permutation_order, _ = axes_to_last_order(len(x.shape), image_axes)
#     H, W = x.shape[image_axes[0]], x.shape[image_axes[1]]
#     aa=reduce(operator.mul, x.shape, 1)
#     bb=reduce(operator.mul, [H, W], 1)
#     N=aa//bb
#     # N = math.prod(x.shape) // math.prod((H, W))

#     if isinstance(y, np.ndarray):
#         max_intensity = 255 if y.dtype == np.uint8 else 1.0
#         max_intensity = np.max(y).item() if peak == 'max' else max_intensity
#         fn_log10 = np.log10

#         x = x.transpose(axes_permutation_order).reshape(N, H, W)
#         y = y.transpose(axes_permutation_order).reshape(N, H, W)
#         psnr = np.zeros(N)
#     elif isinstance(y, torch.Tensor):
#         max_intensity = 255 if y.dtype == torch.uint8 else 1.0
#         max_intensity = torch.max(y).item() if peak == 'max' else max_intensity
#         fn_log10 = torch.log10

#         x = x.permute(axes_permutation_order).reshape(N, H, W)
#         y = y.permute(axes_permutation_order).reshape(N, H, W)
#         psnr = torch.zeros(N)
#     else:
#         raise RuntimeError('Unsupported object type')

#     for i in range(N):
#         psnr[i] = 20 * math.log10(max_intensity) - 10 * fn_log10(compute_mse_q(x[i, ...], y[i, ...]) + eps)

#     if reduction == 'mean':
#         psnr = psnr.mean()  # [1,]
#     elif reduction == 'sum':
#         psnr = psnr.sum()  # [1,]
#     elif reduction == 'none':
#         pass  # [N, ]
#     else:
#         raise RuntimeError('Unsupported reduction type')

#     return psnr
    #former code begin at #b c h w
    # eps = 1e-8  # to avoid math error in log(x) when x=0

    # if is_minmax:
    #     reconstructed_im = minmax_normalize(reconstructed_im, eps)
    #     target_im = minmax_normalize(target_im, eps)
  
    # target_im1=target_im#.transpose(0,2,3,1)
    # reconstructed_im1=reconstructed_im#.transpose(0,2,3,1)
  
    # for i in range(target_im1.shape[0]):
    #     ssim_value = structural_similarity(target_im1[i], reconstructed_im1[i], \
    #         gaussian_weights=True, sigma=1.5, use_sample_covariance=True,multichannel=True,channel_axis=-1)

    # return ssim_value