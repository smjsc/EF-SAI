from __future__ import print_function
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import torch
import os
import cv2 as cv
import argparse
import numpy as np
from torch.utils.data import DataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # choose GPU
from codes.Networks.EF_SAI_Net import EF_SAI_Net
from codes.EF_Dataset import Dataset_EFNet
from prefetch_generator import BackgroundGenerator
def eval_bn(m):
    if type(m) == torch.nn.BatchNorm2d:
        m.eval()

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
##===================================================##
##********** Configure training settings ************##
##===================================================##
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./PreTrained/EF_SAI_Net.pth', help='directory used to store trained networks')
    parser.add_argument('--re_save_path', type=str, default='./Results/')
    parser.add_argument("--valGT", type=str, default="/home_ssd/LW/AIOEdata/sai_database/AugVal1015_pure_aug/Test/Aps/", help="validation aps path")
    parser.add_argument("--valEvent", type=str, default="/home_ssd/LW/AIOEdata/sai_database/AugVal1015_pure_aug/Test/Event/", help="validation event path")
    parser.add_argument("--valFrame", type=str, default="/home_ssd/LW/AIOEdata/sai_database/AugVal1015_pure_aug/Test/Frame/",help="validation frame path")
    parser.add_argument("--valEframe", type=str, default="/home_ssd/LW/AIOEdata/sai_database/AugVal1015_pure_aug/Test/Eframe/",
                        help="validation frame path")
    opts=parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ##===================================================##
    ##*************** Create dataloader *****************##
    ##===================================================##
    model_path = opts.model_path
    re_save_path = opts.re_save_path
    img_w = 256
    img_h = 256
    #valDataset = TrainSet_Hybrid_singleF(opts.valEvent, opts.valAPS, opts.valFrame, opts.valEframe, norm=False, aps_n = 0, if_exposure = False, exposure = 0.04)
    valDataset = Dataset_EFNet(opts.valGT, opts.valEvent, opts.valFrame, opts.valEframe, norm=False)
    valLoader = DataLoaderX(valDataset, batch_size=1, pin_memory=True, num_workers=0, shuffle=False)

    # ##===================================================##
    # ##****************** Create model *******************##
    # ##===================================================##
    print("Begin testing network ...")
    net = EF_SAI_Net()
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(opts.model_path))
    print("All keys matched!")
    net = net.eval()
    if (True):
        print("turn BN to eval mode ...")
        net.apply(eval_bn)
    if not (os.path.exists(opts.re_save_path)):
        os.mkdir(opts.re_save_path)
    if not (os.path.exists(opts.re_save_path + 'Re/')):
        os.mkdir(opts.re_save_path + 'Re/')
    if not (os.path.exists(opts.re_save_path + 'Gt/')):
        os.mkdir(opts.re_save_path + 'Gt/')
    # ##===================================================##
    # ##****************** Test model *******************##
    # ##===================================================##
    with torch.no_grad():
        for i, (event, frame, eframe, gt) in enumerate(valLoader):
            print('Processing ' + format(i,'d'))
            event = event.to(device)
            gt = gt.to(device)
            frame = frame.to(device)
            eframe = eframe.to(device)
            timeWin = event.shape[1]
            outputs = net(event, frame, eframe, timeWin)
            outputs = outputs.cpu().detach()
            outputs = np.array(outputs, dtype=np.float)
            outputs = np.squeeze(outputs)
            outputs = outputs * 255
            cv.imwrite(opts.re_save_path + 'Re/' + format(i, '04d') + '.png', outputs)
            gt = gt.cpu().detach()
            gt = np.array(gt, dtype=np.float)
            gt = np.squeeze(gt)
            gt = gt * 255
            cv.imwrite(opts.re_save_path + 'Gt/' + format(i, '04d') + '.png', gt)