from torch.utils.tensorboard import SummaryWriter
import os, utils_IXI, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TriNet import TriNet, CONFIGS

import time
import datetime

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():
    batch_size = 1
    atlas_dir = r"D:\dataset\IXI-dataset\IXI_data(TransMorph)\atlas.pkl"
    train_dir = r"D:\dataset\IXI-dataset\IXI_data(TransMorph)\Train/"
    val_dir = r"D:\dataset\IXI-dataset\IXI_data(TransMorph)\Val/"
    weights = [1, 1]  # loss weights
    save_dir = 'TriNet(16, 16)_ixi_ncc_{}_diffusion_{}_normal_/'.format(weights[0], weights[1])
    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    sys.stdout = Logger('logs/' + save_dir)
    lr = 0.0002  # learning rate
    epoch_start = 0
    max_epoch = 100  # max traning epoch
    cont_training = False  # if continue training

    TARGET_IMG_SIZE = (160, 192, 224) 

    '''
    Initialize model
    '''
    config = CONFIGS['TriNet']
    config.inshape = TARGET_IMG_SIZE
    model = TriNet(config)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils_IXI.register_model(config.inshape, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils_IXI.register_model(config.inshape, 'bilinear')
    reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 60  
        model_dir = 'experiments/' + save_dir  
        model_files = natsorted(glob.glob(model_dir + '*.pth.tar'))
        if model_files:
            latest_model_path = model_files[-1]
            checkpoint = torch.load(latest_model_path, weights_only=False)
            epoch_start = checkpoint['epoch']
            # best_dsc = checkpoint["best_dsc"]
            best_model_state = checkpoint['state_dict']
            model.load_state_dict(best_model_state)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print(f'Model: {os.path.basename(latest_model_path)} loaded! Starting from epoch {epoch_start}')

            updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
        else:
            print(f"No models_IXI found in {model_dir} to continue training from.")
            updated_lr = lr
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

    val_composed = transforms.Compose([trans.Seg_norm(),  # rearrange segmentation label to 1 to 46
                                       trans.NumpyType((np.float32, np.int16))])
    train_set = datasets.IXIBrainDataset(glob.glob(train_dir + '*.pkl'), atlas_dir, transforms=train_composed)
    val_set = datasets.IXIBrainInferDataset(glob.glob(val_dir + '*.pkl'), atlas_dir, transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion_ncc = losses.NCC_vxm()
    # criterion_dsc = losses.DiceLoss()
    criterion_reg = losses.Grad3d(penalty='l2')
    best_dsc = 0
    writer = SummaryWriter(log_dir='logs/' + save_dir)

    print(f"Starting training with target image size: {TARGET_IMG_SIZE}")
    print(f"Model configured with img_size: {config.inshape}")
    print(f"Spatial transformer configured with img_size: {reg_model.spatial_trans.grid.shape[2:]}")  

    for epoch in range(epoch_start, max_epoch):
        print(f'\nEpoch {epoch}/{max_epoch - 1}')
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils_IXI.AverageMeter()
        idx = 0
        epoch_time_start = time.time()
        time_per_print = 0

        for data in train_loader:
            iter_time_start = time.time()
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            try:
                data = [t.cuda() for t in data]
            except Exception as e:
                print(f"Error moving data to CUDA: {e}")
                print(f"Data shapes: {[t.shape for t in data]}")
                continue

            x = data[0]
            y = data[1]

            flow, output = model(y, x)
            loss_ncc = criterion_ncc(output, y) * weights[0]
            loss_reg = criterion_reg(flow) * weights[1]
            loss = loss_ncc + loss_reg
            loss_all.update(loss.item(), y.numel())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del output, flow 

            iter_time_end = time.time()
            delta = iter_time_end - iter_time_start
            time_per_print += delta
            print(
                f'Iter {idx}/{len(train_loader)} Loss: {loss.item():.4f} (NCC: {loss_ncc.item():.6f}, Reg: {loss_reg.item():.6f}) Time: {time_per_print:.2f}s')
            time_per_print = 0
            # if idx % 20 == 0 or idx == 1: 
            #     print(f'Iter {idx}/{len(train_loader)} Loss: {loss.item():.4f} (NCC: {loss_ncc.item():.6f}, Reg: {loss_reg.item():.6f}) Time: {time_per_print:.2f}s')
            #     time_per_print = 0

            torch.cuda.empty_cache() 

        epoch_time_end = time.time()
        epoch_duration = epoch_time_end - epoch_time_start
        print(f'Epoch {epoch} finished. Average Loss: {loss_all.avg:.4f}. Duration: {epoch_duration:.2f}s')
        writer.add_scalar('Loss/train', loss_all.avg, epoch)

        '''
        Validation
        '''
        print('Validation Starts')
        eval_dsc = utils_IXI.AverageMeter()
        x_1 = None
        y_1 = None
        def_out_1 = None
        output_1 = None
        x_seg_1 = None
        y_seg_1 = None
        def_grid_1 = None
        with torch.no_grad():
            val_idx = 0
            for data_val in val_loader:
                val_idx += 1
                model.eval()
                try:
                    data_val = [t.cuda() for t in data_val]
                except Exception as e:
                    print(f"Error moving validation data to CUDA: {e}")
                    continue

                x = data_val[0]
                y = data_val[1]
                x_seg = data_val[2]
                y_seg = data_val[3]

                grid_img = mk_grid_img(8, 1, config.inshape) 
                flow, output = model(y, x) 
                def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
                def_grid = reg_model_bilin([grid_img.float(), flow.cuda()])  

                dsc = utils_IXI.dice_val_VOI(def_out.long(), y_seg.long()) 
                eval_dsc.update(dsc.item(), x.size(0))
                if val_idx % 10 == 0:  
                    print(f'Validation Iter {val_idx}/{len(val_loader)}, Current Avg Dice: {eval_dsc.avg:.4f}')
                x_1 = x
                y_1 = y
                def_out_1 = def_out
                output_1 = output
                x_seg_1 = x_seg
                y_seg_1 = y_seg
                def_grid_1 = def_grid
                del flow  
                torch.cuda.empty_cache()

        print(f'Validation finished. Average Dice: {eval_dsc.avg:.4f}')
        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)

        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/' + save_dir, filename=f'dsc_{eval_dsc.avg:.4f}.pth.tar') 


        epochs_left = max_epoch - (epoch + 1)
        estimated_total_time = epoch_duration * epochs_left
        time_result_str = str(datetime.timedelta(seconds=int(estimated_total_time)))
        print(f"Estimated time remaining: {time_result_str}")


        plt.switch_backend('agg')
        xvol_fig = comput_fig(x_1)
        tarvol_fig = comput_fig(y_1)
        predvol_fig = comput_fig(output_1)
        pred_fig = comput_fig_label(def_out_1)  
        grid_fig = comput_fig(def_grid_1)
        x_fig = comput_fig_label(x_seg_1)
        tar_fig = comput_fig_label(y_seg_1)
        writer.add_figure('input_vol', xvol_fig, epoch)
        plt.close(xvol_fig)
        writer.add_figure('gt_vol', tarvol_fig, epoch)
        plt.close(tarvol_fig)
        writer.add_figure('pred_vol', predvol_fig, epoch)
        plt.close(predvol_fig)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('input', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)
        loss_all.reset()
        eval_dsc.reset()  
        del def_out, def_grid, grid_img, x_seg, y_seg, x, y

    writer.close()


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):  
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[0], grid_step):
        grid_img[j + line_thickness - 1, :, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i + line_thickness - 1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def save_checkpoint(state, save_dir='models_IXI', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir + filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, :, 88:104, :]
    fig = plt.figure(figsize=(12, 12), dpi=240)
    for i in range(img.shape[1]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[:, i, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def comput_fig_label(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=240)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    torch.manual_seed(0)
    main()