import glob
import os
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TriNet import TriNet, CONFIGS
import torch.nn as nn
import utils_IXI
import time
from thop import profile

def main():
    atlas_dir = 'root'
    test_dir = 'root'
    model_idx = -1
    weights = [1, 1]
    model_folder = 'TriNet_last_IXI_ncc_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_dir = ""
    if 'Val' in test_dir:
        csv_name = model_folder[:-1]+'_Val'
    else:
        csv_name = model_folder[:-1]
    dict = utils_IXI.process_label()
    if not os.path.exists('Quantitative_Results/'):
        os.makedirs('Quantitative_Results/')
    if os.path.exists('Quantitative_Results/'+csv_name+'.csv'):
        os.remove('Quantitative_Results/'+csv_name+'.csv')
    csv_writter(model_folder[:-1], 'Quantitative_Results/' + csv_name)
    line = ''
    for i in range(46):
        line = line + ',' + dict[i]
    csv_writter(line +','+'non_jec', 'Quantitative_Results/' + csv_name)

    config = CONFIGS["TriNet"]
    model = TriNet(config)
    SIZE = (160, 192, 224)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx], weights_only=False)['state_dict']
    # best_model = torch.load(model_dir)['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    # print('Best model: {}'.format(model_dir))

    model.load_state_dict(best_model)
    model.cuda()
    
    
    reg_model = utils_IXI.register_model(SIZE, 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([trans.Seg_norm_IXI(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    eval_dsc_def = utils_IXI.AverageMeter()
    eval_dsc_raw = utils_IXI.AverageMeter()
    eval_det = utils_IXI.AverageMeter()
    eval_hd95 = utils_IXI.AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            flow, x_def = model(x, y)

            def_out = reg_model([y_seg.cuda().float(), flow.cuda()])
            tar = x.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils_IXI.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            line = utils_IXI.dice_val_substruct(def_out.long(), x_seg.long(), stdy_idx)
            line = line +','+str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            csv_writter(line, 'Quantitative_Results/' + csv_name)
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), y.size(0))
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(tar.shape)))
            hd95 = utils_IXI.hd95_val_substruct(def_out.long(), x_seg.long(), stdy_idx)
            print('hd95: {}'.format(hd95))
            dsc_trans = utils_IXI.dice_val_VOI(def_out.long(), x_seg.long())
            dsc_raw = utils_IXI.dice_val_VOI(y_seg.long(), x_seg.long())
            # dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 46)
            # dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), y.size(0))
            eval_dsc_raw.update(dsc_raw.item(), y.size(0))
            eval_hd95.update(hd95.item(), y.size(0))

            stdy_idx += 1

        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))
        print("hd95:{}, std: {}".format(eval_hd95.avg, eval_hd95.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

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
    main()
