# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import os
import scipy.io as io
import argparse
import logging
import random
import shutil
import time
import numpy as np
from tqdm import tqdm
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
# Custom
from net import ParallelNetwork as Network
from IXI_dataset import IXIData as Dataset
from IXI_dataset import build_loader
from mri_tools import rA, rAtA, ifft2,fft2  #
# from utils import psnr_slice, ssim_slice,pseudo2real,compute_psnr,compute_ssim,compute_psnr_q
from utils import pseudo2real,compute_ssim,compute_psnr_q,save_images,get_mask
from dataprocess import complex2pseudo,pseudo2complex,imsshow,image2kspace,kspace2image

from loss import cal_loss
import matplotlib.pyplot as plt
# System / Python
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#add
parser = argparse.ArgumentParser()
parser.add_argument('--strain','-strain', type=str, default=None, help='file_name_train')
parser.add_argument('--stest','-stest', type=str, default=None, help='file_name_test')
parser.add_argument('--overlap','-overlap', type=int, default=8, help='file_name_test')

parser.add_argument('--exp-name', type=str, default='self-supervised MRI reconstruction', help='name of experiment')
# parameters related to distributed training
parser.add_argument('--init-method', default=f'tcp://localhost:{np.random.randint(1000,2000)}', help='initialization method')
parser.add_argument('--nodes', type=int, default=1, help='number of nodes')
parser.add_argument('--gpus', type=int, default=torch.cuda.device_count(), help='number of gpus per node')
parser.add_argument('--world-size', type=int, default=None, help='world_size = nodes * gpus')
# parameters related to model
parser.add_argument('--use-init-weights', '-uit', type=bool, default=True, help='whether initialize model weights with defined types')
parser.add_argument('--init-type', type=str, default='xavier', help='type of initialize model weights')
parser.add_argument('--gain', type=float, default=1.0, help='gain in the initialization of model weights')
parser.add_argument('--num-layers', type=int, default=9, help='number of iterations')
# learning rate, batch size, and etc
parser.add_argument('--seed', type=int, default=30, help='random seed number')
parser.add_argument('--lr', '-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--batch-size', type=int, default=1, help='batch size of single gpu') #batch size=4 former
parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
parser.add_argument('--warmup-epochs', type=int, default=10, help='number of warmup epochs')
parser.add_argument('--num-epochs', type=int, default=500, help='maximum number of epochs')
# parameters related to data and masks
parser.add_argument('--train-path', type=str, default='/home/liuchun/Desktop/0_experiment/data/PD_train_852(71).npz', help='path of training data')
parser.add_argument('--val-path', type=str, default='/home/liuchun/Desktop/0_experiment/data/PD_val_252(21).npz', help='path of validation data')
parser.add_argument('--test-path', type=str, default='/home/liuchun/Desktop/0_experiment/data/PD_test_252(21).npz', help='path of test data')
# parser.add_argument('--u-mask-path','-ump', type=str, default='/home/liuchun/Desktop/ovlm_parallel_02/mask/undersampling_mask/mask_4.00x_acs24.mat', help='undersampling mask')
parser.add_argument('--u-mask-path','-ump', type=str, default='vd', help='undersampling mask')
parser.add_argument('--s-mask-up-path', '-smup',type=str, default='/home/liuchun/Desktop/ovlm_parallel_02/mask/selecting_mask/mask_2.00x_acs16.mat', help='selection mask in up network')
parser.add_argument('--s-mask-down-path','-smdp', type=str, default='/home/liuchun/Desktop/ovlm_parallel_02/mask/selecting_mask/mask_2.50x_acs16.mat', help='selection mask in down network')
parser.add_argument('--method','-method', type=str, default='baseline', help='choose baseline or loupe')
# parser.add_argument('--train-sample-rate', '-trsr', type=float, default=0.06, help='sampling rate of training data')
# parser.add_argument('--val-sample-rate', '-vsr', type=float, default=0.02, help='sampling rate of validation data')
# parser.add_argument('--test-sample-rate', '-tesr', type=float, default=0.02, help='sampling rate of test data')
# save path
# parser.add_argument('--model-save-path', '-mpath', type=str, default='./checkpoints/', help='save path of trained model')
#/home/liuchun/Desktop/0_experiment/model_save/
# parser.add_argument('--model-save-path', '-mpath', type=str, default='model_paramters', help='save path of trained model')
parser.add_argument('--model-save-path', '-mpath', type=str, default='/home/liuchun/Desktop/0_experiment/model_save/m00', help='save path of trained model')
parser.add_argument('--loss-curve-path', '-lpath', type=str, default='loss_log', help='save path of loss curve in tensorboard')
# others
parser.add_argument('--mode', '-m', type=str, default='train', help='whether training or test model, value should be set to train or test')
parser.add_argument('--pretrained', '-pt', type=bool, default=True, help='whether load checkpoint')
# parser.add_argument('--pretrained', '-pt', type=bool, default=False, help='whether load checkpoint')


def create_logger():
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s:\t%(message)s')
    stream_formatter = logging.Formatter('%(levelname)s:\t%(message)s')

    file_handler = logging.FileHandler(filename='logger.txt', mode='a+', encoding='utf-8')
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def init_weights(net, init_type='xavier', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('Initialization method {} is not implemented.'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


class EarlyStopping:
    def __init__(self, patience=50, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, metrics, loss=True):
        score = -metrics if loss else metrics
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def forward(mode, rank, model, dataloader, criterion, optimizer, log, args ):

    assert mode in ['train', 'val', 'test']
    loss, psnr, ssim = 0.0, 0.0, 0.0
    t = tqdm(dataloader, desc=mode + 'ing', total=int(len(dataloader))) if rank == 0 else dataloader
    for iter_num, data_batch in enumerate(t):
        label = data_batch[0].to(rank, non_blocking=True)
        mask_under = data_batch[1].to(rank, non_blocking=True)
        mask_up = data_batch[2].to(rank, non_blocking=True)
        mask_down = data_batch[3].to(rank, non_blocking=True)
        mask_under=mask_under.permute(0,3,1,2)
        mask_up=mask_up.permute(0,3,1,2)
        mask_down=mask_down.permute(0,3,1,2)
        option=True
        method=args.method
        gt_kspace = complex2pseudo(image2kspace(pseudo2complex(label)))
        print('method:',method)

        if(method=="baseline"):
            mask_net_up=mask_up
            mask_net_down=mask_down
            if mode=='test':
                mask_net_up=mask_net_down=mask_under
            net_img_up = rAtA(label, mask_net_up)
            net_img_down = rAtA(label, mask_net_down)
            output_up,  output_down = model(net_img_up.contiguous(), mask_net_up, net_img_down.contiguous(), mask_net_down,mode)

        if(method=="loupe"):
            output_up,  output_down,mask_net_up,mask_net_down  = model(mask_under.contiguous(), label.contiguous(),args.overlap,option,mode)
            net_img_up=complex2pseudo(kspace2image(pseudo2complex(gt_kspace*mask_net_up)))
            net_img_down=complex2pseudo(kspace2image(pseudo2complex(gt_kspace*mask_net_down)))
       
        out_mean=(output_up+output_down)/2.0
        under_img = rAtA(label, mask_under)
        under_kspace = rA(label, mask_under)

        # 保存生成图像
        if(args.strain!=None):
            save_images(under_img,net_img_up,net_img_down,\
                        output_up,output_down,mask_under,\
                            mask_net_up,mask_net_down,out_mean,\
                                label,iter_num,args.strain,mode)
        elif(args.stest!=None):
           save_images(under_img,net_img_up,net_img_down,\
                        output_up,output_down,mask_under,\
                            mask_net_up,mask_net_down,out_mean,\
                                label,iter_num,args.stest,mode)
    
        output_up, output_down = output_up .contiguous(), output_down .contiguous()
        output_up_kspace = fft2(output_up)
        output_down_kspace = fft2(output_down)
        label_kspace=fft2(label)
        #计算损失
        sssim_up=compute_ssim(pseudo2real(label),pseudo2real(output_up))
        sssim_down=compute_ssim(pseudo2real(label),pseudo2real(output_down))
 
        diff_otherf = (output_up_kspace - output_down_kspace) * (1 - mask_under)
        recon_loss_up = criterion(output_up_kspace * mask_under, under_kspace)
        recon_loss_down = criterion(output_down_kspace * mask_under, under_kspace)
        diff_loss = criterion(diff_otherf, torch.zeros_like(diff_otherf))
        lammda= nn.Parameter(torch.tensor(0.01,dtype=torch.float32))
        batch_loss = recon_loss_up + recon_loss_down + lammda * diff_loss   #+ 0.01 * constr_loss_up + 0.01 * constr_loss_down

        if mode == 'train':
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            a=pseudo2real(label)
            b=pseudo2real(out_mean) #参照parcel都对均值进行约束
            psnr += compute_psnr_q(a, b)            
            ssim += compute_ssim(a, b)
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(name)
                    print(param.device)
        else:
            a=pseudo2real(label)
            b=pseudo2real(out_mean)
            psnr += compute_psnr_q(a, b)            
            ssim += compute_ssim(a, b)
        loss += batch_loss.item()
        sssim_up+=sssim_up
        sssim_down+=sssim_down
    loss /= len(dataloader)
    sssim_up/=len(dataloader)
    sssim_down/=len(dataloader)
    up_ssim_log = sssim_up 
    down_ssim_log = sssim_down 
    log.append(loss)
    print('up_ssim_log:',up_ssim_log)
    print('down_ssim_log:',down_ssim_log)
    print('loss:',loss)
    if mode == 'train':
        curr_lr = optimizer.param_groups[0]['lr']
        log.append(curr_lr)
    else:
        psnr /= len(dataloader)
        ssim /= len(dataloader)
        log.append(psnr)
        log.append(ssim)
    return log,up_ssim_log,down_ssim_log


def solvers(rank, ngpus_per_node, args):
    if rank == 0:
        logger = create_logger()
        logger.info('Running distributed data parallel on {} gpus.'.format(args.world_size))
    
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, world_size=args.world_size, rank=rank)
    # set initial value
    start_epoch = 0
    best_ssim = 0.0
    # model
    model = Network(method=args.method, rank=rank)
    # whether load checkpoint
    if args.pretrained or args.mode == 'test':
        # path_of_model='/home/liuchun/Desktop/0_experiment/model_save/'+args.model_save_path
        path_of_model= args.model_save_path #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        model_path = os.path.join(path_of_model, 'best_checkpoint.pth.tar')

        assert os.path.isfile(model_path)
        checkpoint = torch.load(model_path, map_location='cuda:{}'.format(rank))
        start_epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        args.lr = lr
        best_ssim = checkpoint['best_ssim']
        model.load_state_dict(checkpoint['model'])
        if rank == 0:
            logger.info('Load checkpoint at epoch {}.'.format(start_epoch))
            logger.info('Current learning rate is {}.'.format(lr))
            logger.info('Current best ssim in train phase is {}.'.format(best_ssim))
            logger.info('The model is loaded.')
    elif args.use_init_weights:
        init_weights(model, init_type=args.init_type, gain=args.gain)
        if rank == 0:
            logger.info('Initialize model with {}.'.format(args.init_type))
    model = model.to(rank)

    # criterion, optimizer, learning rate scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if not args.pretrained:
        warm_up = lambda epoch: epoch / args.warmup_epochs if epoch <= args.warmup_epochs else 1
        scheduler_wu = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_up)
    scheduler_re = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.3, patience=20)
    early_stopping = EarlyStopping(patience=50, delta=1e-5)
    
    #数据集部分
    '''
    根据传入参数选择使用的mask
    vd:变密度采集
    c：笛卡尔采集
    r：随机采集
    默认使用baseline的mask
    '''
    u_mask_path,s_mask_up_path,s_mask_down_path=get_mask(args.u_mask_path)
    
    dataset_train = Dataset(data_path=args.train_path,u_mask_path=u_mask_path,s_mask_up_path=s_mask_up_path,s_mask_down_path=s_mask_down_path)
    dataset_val = Dataset(data_path=args.val_path,u_mask_path=u_mask_path,s_mask_up_path=s_mask_up_path,s_mask_down_path=s_mask_down_path)
    dataset_test = Dataset(data_path=args.test_path,u_mask_path=u_mask_path,s_mask_up_path=s_mask_up_path,s_mask_down_path=s_mask_down_path)

    train_loader =build_loader(dataset_train,args.batch_size,is_shuffle=True)
    val_loader =build_loader(dataset_val,args.batch_size,is_shuffle=False)
    test_loader =build_loader(dataset_test,args.batch_size,is_shuffle=False)
    
    # test step
    if args.mode == 'test':
        model.eval()
        with torch.no_grad():
            test_log = []
            start_time = time.time()
            test_log,_,_ = forward('test', rank, model, test_loader, criterion, optimizer, test_log, args )
            test_time = time.time() - start_time
        # test information
        test_loss = test_log[0]
        test_psnr = test_log[1]
        test_ssim = test_log[2]
        if rank == 0:
            logger.info('time:{:.5f}s\ttest_loss:{:.7f}\ttest_psnr:{:.5f}\ttest_ssim:{:.5f}'.format(test_time, test_loss, test_psnr, test_ssim))
        return

    if rank == 0:
        path_loss='/home/liuchun/Desktop/0_experiment/loss_save/'
        path_loss=path_loss+args.loss_curve_path
        writer = SummaryWriter(path_loss)
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        # train_sampler.set_epoch(epoch)
        train_log = [epoch]
        epoch_start_time = time.time()
        model.train()
        train_log,ssim_up_look,ssim_down_look = forward('train', rank, model, train_loader, criterion, optimizer, train_log, args )
        model.eval()
        with torch.no_grad():
            train_log,ssim_up_look,ssim_down_look = forward('val', rank, model, val_loader, criterion, optimizer, train_log, args )
        epoch_time = time.time() - epoch_start_time
        # train information
        epoch = train_log[0]
        train_loss = train_log[1]
        lr = train_log[2]
        val_loss = train_log[3]
        val_psnr = train_log[4]
        val_ssim = train_log[5]

        is_best = val_ssim > best_ssim
        best_ssim = max(val_ssim, best_ssim)
        if rank == 0:
            logger.info('epoch:{:<8d}time:{:.5f}s\tlr:{:.8f}\ttrain_loss:{:.7f}\tval_loss:{:.7f}\tval_psnr:{:.5f}\t'
                        'val_ssim:{:.5f}'.format(epoch, epoch_time, lr, train_loss, val_loss, val_psnr, val_ssim))
            writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
            writer.add_scalars('lossupdown', {'loss_up': ssim_up_look, 'loss_down': ssim_down_look}, epoch)
            # save checkpoint
            checkpoint = {
                'epoch': epoch,
                'lr': lr,
                'best_ssim': best_ssim,
                # 'model': model.module.state_dict()
                'model': model.state_dict()
            }
            path_of_model='/home/liuchun/Desktop/0_experiment/model_save/'+args.model_save_path
            if not os.path.exists(path_of_model):
                os.makedirs(path_of_model)
            model_path = os.path.join(path_of_model, 'checkpoint.pth.tar')
            best_model_path = os.path.join(path_of_model, 'best_checkpoint.pth.tar')
            torch.save(checkpoint, model_path)
            if is_best:
                shutil.copy(model_path, best_model_path)
        # scheduler
        if epoch <= args.warmup_epochs and not args.pretrained:
            scheduler_wu.step()
        scheduler_re.step(val_ssim)
        early_stopping(val_ssim, loss=False)
        if early_stopping.early_stop:
            if rank == 0:
                logger.info('The experiment is early stop!')
            break
    if rank == 0:
        writer.close()
    return


def main():
    args = parser.parse_args()
    args.world_size = args.nodes * args.gpus
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.multiprocessing.spawn(solvers, nprocs=args.gpus, args=(args.gpus, args))

if __name__ == '__main__':
    main()
