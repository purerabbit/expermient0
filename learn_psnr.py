from typing import Sequence, Tuple
import numpy as np
import math
import torch



def compute_mse_q(x, y, axis=None):
    """
    REQUIREMENT: `x` and `y` can be any shape, but their shape have to be same
    """
    assert x.dtype == y.dtype and x.shape == y.shape, \
        'x and y is not compatible to compute MSE metric'

    if axis:
        mse = ((x - y) ** 2).mean(axis=axis)
    else:
        mse = ((x - y) ** 2).mean()

    return mse


def axes_to_last_order(ndim, axes_idx: Sequence[int]):
    """
    根据数组尺寸长度和最后两个轴的索引计算出 permute的索引顺序
    Generate axes order of move the axes specifed by `axes_idx` to last in a n-dimension axes,
    return indices can used to permute(array, indices) and inverse_indices can used
    to restore the data by permute(array, inverse_indices)
    eg.
        (Nb, Nx, Ny, Ntime) --move(2,3)--> (Nb, Ntime, Nx, Ny) may be wrong
        (..., Nx, Ny, Ntime, Nslice) --move(-4,-3)--> (..., Ntime, Nslice, Nx, Ny)
        shape (1,2,3,4,5) --move(3,2)--> shape (1,2,5,4,3)
    """
    # convert reversed index to ordinal index eg: -1,-2  ---> 3,2(when the len of shape is 4)
    axes_need_move = [(axis if axis >= 0 else ndim + axis) for axis in axes_idx]

    # generate forward-indices
    forward_indices = tuple([axis for axis in range(ndim) if axis not in axes_need_move] + axes_need_move)

    # generate inverse-indices
    inverse_indices = np.argsort(forward_indices).tolist() #argsort 将数组中的元素按照从小到大进行排序 输出对应排序后的索引

    return forward_indices, inverse_indices

def compute_psnr_q(x, y, peak='normalized', image_axes=(-2, -1), reduction='mean', eps=1e-16):
    '''
    Image must be of either Integer [0, 255] or Float value [0,1]
    :param peak: 'max' or 'normalize', max_intensity will be the maximum value of target_im if peek == 'max.
          when peek is 'normalized', max_intensity will be the maximum value depend on data representation (in this
          case, we assume your input should be normalized to [0,1])
    :param image_axes: specify the image's axes (H, W)
    :param eps: to avoid math error in log(x) when x=0
    REQUIREMENT: `x` and `y` is [..., H, W, ...], their shape have to be same
    '''
    assert len(x.shape) >= 2
    assert y.dtype == x.dtype and y.shape == x.shape, \
        f"shape of x {x.shape} and shape of y {y.shape} is not compatible to compute PSNR metric"
    assert peak in {'max', 'normalized'}, \
        'peak mode is not supported'

    axes_permutation_order, _ = axes_to_last_order(len(x.shape), image_axes)
    H, W = x.shape[image_axes[0]], x.shape[image_axes[1]]
    N = math.prod(x.shape) // math.prod((H, W))

    if isinstance(y, np.ndarray):
        max_intensity = 255 if y.dtype == np.uint8 else 1.0
        max_intensity = np.max(y).item() if peak == 'max' else max_intensity
        fn_log10 = np.log10

        x = x.transpose(axes_permutation_order).reshape(N, H, W)
        y = y.transpose(axes_permutation_order).reshape(N, H, W)
        psnr = np.zeros(N)
    elif isinstance(y, torch.Tensor):
        max_intensity = 255 if y.dtype == torch.uint8 else 1.0
        max_intensity = torch.max(y).item() if peak == 'max' else max_intensity
        fn_log10 = torch.log10

        x = x.permute(axes_permutation_order).reshape(N, H, W)
        y = y.permute(axes_permutation_order).reshape(N, H, W)
        psnr = torch.zeros(N)
    else:
        raise RuntimeError('Unsupported object type')

    for i in range(N):
        psnr[i] = 20 * math.log10(max_intensity) - 10 * fn_log10(compute_mse_q(x[i, ...], y[i, ...]) + eps)

    if reduction == 'mean':
        psnr = psnr.mean()  # [1,]
    elif reduction == 'sum':
        psnr = psnr.sum()  # [1,]
    elif reduction == 'none':
        pass  # [N, ]
    else:
        raise RuntimeError('Unsupported reduction type')

    return psnr

def minmax_normalize(x, eps=1e-8):
    min = x.min()
    max = x.max()
    return (x - min) / (max - min + eps)


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


def compute_psnr(reconstructed_im, target_im, peak='normalized', is_minmax=False):
    '''
    Image must be of either Integer [0, 255] or Float value [0,1]
    :param peak: 'max' or 'normalize', max_intensity will be the maximum value of target_im if peek == 'max.
          when peek is 'normalized', max_intensity will be the maximum value depend on data representation (in this
          case, we assume your input should be normalized to [0,1])
    REQUIREMENT: `x` and `y` can be any shape, but their shape have to be same
    '''
    assert target_im.dtype == reconstructed_im.dtype and target_im.shape == reconstructed_im.shape, \
        'target_im and reconstructed_im is not compatible to compute PSNR metric'
    assert peak in {'max', 'normalized'}, \
        'peak mode is not supported'

    eps = 1e-8  # to avoid math error in log(x) when x=0

    if is_minmax:
        reconstructed_im = minmax_normalize(reconstructed_im, eps)
        target_im = minmax_normalize(target_im, eps)

    if isinstance(target_im, np.ndarray):
        max_intensity = 255 if target_im.dtype == np.uint8 else 1.0
        max_intensity = np.max(target_im).item() if peak == 'max' else max_intensity
        psnr = 20 * math.log10(max_intensity) - 10 * np.log10(compute_mse(reconstructed_im, target_im) + eps)

    elif isinstance(target_im, torch.Tensor):
        max_intensity = 255 if target_im.dtype == torch.uint8 else 1.0
        max_intensity = torch.max(target_im).item() if peak == 'max' else max_intensity
        psnr = 20 * math.log10(max_intensity) - 10 * torch.log10(compute_mse(reconstructed_im, target_im) + eps)

    else:
        raise RuntimeError(
            'Unsupported object type'
        )
    return psnr



import cv2
import matplotlib.pyplot as plt
import pylab

if __name__=="__main__":
    # forward_indices, inverse_indices=axes_to_last_order((1,2,3,4,5),(3,2))
    # forward_indices, inverse_indices=axes_to_last_order(5,(3,2)) 
    # forward_indices, inverse_indices=axes_to_last_order(5,(4,2)) 
    # print('forward_indices:',forward_indices)
    # print('inverse_indices:',inverse_indices)

    # image_size = [256, 256] #将图像转化为512*512大小的尺寸 
    # imag1 = cv2.resize(imag1, image_size, interpolation=cv2.INTER_CUBIC)
    # #检验psnr计算函数
    img_label = cv2.imread("/home/liuchun/Desktop/parallel_02/train_imgs/wantfig1/1label.png")
    img_label = cv2.cvtColor(img_label, cv2.COLOR_BGR2GRAY)
    # print('img_label.dtype:',img_label.dtype)
    # print("img_label.shape: {}".format(img_label.shape))
    # plt.subplot(121)
    # plt.imshow(img_label)

    img_out = cv2.imread("/home/liuchun/Desktop/parallel_02/train_imgs/wantfig1/1.png")
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
    print("img_out.shape: {}".format(img_out.shape))
    cv2.imshow("img_out", img_out)    # 显示图像
    cv2.waitKey()               # 默认为0，无限等待
    cv2.destroyAllWindows()      # 释放所有窗口
    # plt.subplot(122)
    # plt.imshow(img_out)
    # plt.show()
    # pylab.show()

    # res1 = compute_psnr_q(img_label, img_out)
    # print("res1:", res1)
    # res2 = compute_psnr(img_label, img_out)
    # print("res2:", res2)