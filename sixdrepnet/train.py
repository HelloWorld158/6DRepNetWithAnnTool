import time
import math
import re
import sys
import os,sys

curDir=os.path.dirname(os.path.abspath(__file__))
sys.path.append(curDir)
basDir=os.path.dirname(curDir)
sys.path.append(basDir)
import cvSDK.BasicUseFunc as basFunc
import argparse
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score
import numpy as np
from numpy.lib.function_base import _quantile_unchecked
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.utils import model_zoo
import torchvision
from torchvision import transforms
from transform import *
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
#matplotlib.use('TkAgg')

from model import SixDRepNet, SixDRepNet2,SixDRepNet3
import datasets
from loss import GeodesicLoss,GeodesicLossEx
from utils import *


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--num_epochs', dest='num_epochs',
        help='Maximum number of training epochs.',
        default=80, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=80, type=int)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.0002, type=float)
    parser.add_argument('--scheduler', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument(
        '--dataset', dest='dataset', help='Dataset type.',
        default='chenzhuo', type=str) #Pose_300W_LP
    parser.add_argument(
        '--data_dir', dest='data_dir', help='Directory path for data.',
        default='headDir/train', type=str)#BIWI_70_30_train.npz
    parser.add_argument(
        '--vdata_dir', dest='vdata_dir', help='Directory path for data.',
        default='headDir/val', type=str)
    parser.add_argument(
        '--filename_list', dest='filename_list',
        help='Path to text file containing relative paths for every example.',
        default='datasets/300W_LP/files.txt', type=str) #BIWI_70_30_train.npz #300W_LP/files.txt
    parser.add_argument(
        '--output_string', dest='output_string',
        help='String appended to output snapshots.', default='', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)

    args = parser.parse_args()
    return args

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)
def Validate(model,val_loader):
    total = 0
    yaw_error = pitch_error = roll_error = .0
    v1_err = v2_err = v3_err = .0
    model.eval()
    with torch.no_grad():
        for i, (images, gt_mat, _, file) in enumerate(val_loader):
            basFunc.Process(i,len(val_loader))
            images = torch.Tensor(images).cuda()
            R_pred = model(images)
            msk=gt_mat.abs().sum(-1).sum(-1)==0
            gt_mat=gt_mat[~msk]
            R_pred=R_pred[~msk]
            cont_labels=compute_euler_angles_from_rotation_matrices(gt_mat)
            total += cont_labels.size(0)
            gt_deg=cont_labels.float()*180/np.pi
            
            euler = compute_euler_angles_from_rotation_matrices(
                R_pred)*180/np.pi
            p_pred_deg = euler[:, 0].cpu()
            y_pred_deg = euler[:, 1].cpu()
            r_pred_deg = euler[:, 2].cpu()
            p_gt_deg= gt_deg[:, 0].float().cpu()
            y_gt_deg = gt_deg[:, 1].float().cpu()
            r_gt_deg = gt_deg[:, 2].float().cpu()

            #img=cv2.imread(file[0])
            #p,y,r=p_gt_deg[0],y_gt_deg[0],r_gt_deg[0]
            #img=draw_axis(img,y,p,r)
            #cv2.imwrite('temp.jpg',img)
            pitch_error += torch.sum(torch.min(torch.stack((torch.abs(p_gt_deg - p_pred_deg), torch.abs(p_pred_deg + 360 - p_gt_deg), torch.abs(
                p_pred_deg - 360 - p_gt_deg))), 0)[0])
            yaw_error += torch.sum(torch.min(torch.stack((torch.abs(y_gt_deg - y_pred_deg), torch.abs(y_pred_deg + 360 - y_gt_deg), torch.abs(
                y_pred_deg - 360 - y_gt_deg))), 0)[0])
            roll_error += torch.sum(torch.min(torch.stack((torch.abs(r_gt_deg - r_pred_deg), torch.abs(r_pred_deg + 360 - r_gt_deg), torch.abs(
                r_pred_deg - 360 - r_gt_deg))), 0)[0])



            # pitch_error += torch.sum(torch.min(torch.stack((torch.abs(p_gt_deg - p_pred_deg), torch.abs(p_pred_deg + 360 - p_gt_deg), torch.abs(
            #     p_pred_deg - 360 - p_gt_deg), torch.abs(p_pred_deg + 180 - p_gt_deg), torch.abs(p_pred_deg - 180 - p_gt_deg))), 0)[0])
            # yaw_error += torch.sum(torch.min(torch.stack((torch.abs(y_gt_deg - y_pred_deg), torch.abs(y_pred_deg + 360 - y_gt_deg), torch.abs(
            #     y_pred_deg - 360 - y_gt_deg), torch.abs(y_pred_deg + 180 - y_gt_deg), torch.abs(y_pred_deg - 180 - y_gt_deg))), 0)[0])
            # roll_error += torch.sum(torch.min(torch.stack((torch.abs(r_gt_deg - r_pred_deg), torch.abs(r_pred_deg + 360 - r_gt_deg), torch.abs(
            #     r_pred_deg - 360 - r_gt_deg), torch.abs(r_pred_deg + 180 - r_gt_deg), torch.abs(r_pred_deg - 180 - r_gt_deg))), 0)[0])
    mae=(yaw_error + pitch_error + roll_error) / (total * 3)
    print('Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f' % (
            yaw_error / total, pitch_error / total, roll_error / total,
            (yaw_error + pitch_error + roll_error) / (total * 3)))
    model.train()
    return mae
def Validate2(model,val_loader,thr=0.5):
    total = 0
    yaw_error = pitch_error = roll_error = .0
    v1_err = v2_err = v3_err = .0
    preds,gts=[],[]
    model.eval()
    with torch.no_grad():
        for i, (images, gt_mat, _, file) in enumerate(val_loader):
            basFunc.Process(i,len(val_loader))
            images = torch.Tensor(images).cuda()
            R_pred,pred = model.Inference(images)
            predout=pred.cpu()>thr
            msk=gt_mat.abs().sum(-1).sum(-1)==0
            gts.extend((~msk).cpu().numpy().tolist())
            preds.extend(predout.cpu().numpy().tolist())
            gt_mat=gt_mat[~msk]
            R_pred=R_pred[~msk]
            cont_labels=compute_euler_angles_from_rotation_matrices(gt_mat)
            total += cont_labels.size(0)
            gt_deg=cont_labels.float()*180/np.pi            
            euler = compute_euler_angles_from_rotation_matrices(
                R_pred)*180/np.pi
            p_pred_deg = euler[:, 0].cpu()
            y_pred_deg = euler[:, 1].cpu()
            r_pred_deg = euler[:, 2].cpu()
            p_gt_deg= gt_deg[:, 0].float().cpu()
            y_gt_deg = gt_deg[:, 1].float().cpu()
            r_gt_deg = gt_deg[:, 2].float().cpu()

            #img=cv2.imread(file[0])
            #p,y,r=p_gt_deg[0],y_gt_deg[0],r_gt_deg[0]
            #img=draw_axis(img,y,p,r)
            #cv2.imwrite('temp.jpg',img)
            pitch_error += torch.sum(torch.min(torch.stack((torch.abs(p_gt_deg - p_pred_deg), torch.abs(p_pred_deg + 360 - p_gt_deg), torch.abs(
                p_pred_deg - 360 - p_gt_deg))), 0)[0])
            yaw_error += torch.sum(torch.min(torch.stack((torch.abs(y_gt_deg - y_pred_deg), torch.abs(y_pred_deg + 360 - y_gt_deg), torch.abs(
                y_pred_deg - 360 - y_gt_deg))), 0)[0])
            roll_error += torch.sum(torch.min(torch.stack((torch.abs(r_gt_deg - r_pred_deg), torch.abs(r_pred_deg + 360 - r_gt_deg), torch.abs(
                r_pred_deg - 360 - r_gt_deg))), 0)[0])



            # pitch_error += torch.sum(torch.min(torch.stack((torch.abs(p_gt_deg - p_pred_deg), torch.abs(p_pred_deg + 360 - p_gt_deg), torch.abs(
            #     p_pred_deg - 360 - p_gt_deg), torch.abs(p_pred_deg + 180 - p_gt_deg), torch.abs(p_pred_deg - 180 - p_gt_deg))), 0)[0])
            # yaw_error += torch.sum(torch.min(torch.stack((torch.abs(y_gt_deg - y_pred_deg), torch.abs(y_pred_deg + 360 - y_gt_deg), torch.abs(
            #     y_pred_deg - 360 - y_gt_deg), torch.abs(y_pred_deg + 180 - y_gt_deg), torch.abs(y_pred_deg - 180 - y_gt_deg))), 0)[0])
            # roll_error += torch.sum(torch.min(torch.stack((torch.abs(r_gt_deg - r_pred_deg), torch.abs(r_pred_deg + 360 - r_gt_deg), torch.abs(
            #     r_pred_deg - 360 - r_gt_deg), torch.abs(r_pred_deg + 180 - r_gt_deg), torch.abs(r_pred_deg - 180 - r_gt_deg))), 0)[0])
    mae=(yaw_error + pitch_error + roll_error) / (total * 3)
    print('Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f' % (
            yaw_error / total, pitch_error / total, roll_error / total,
            (yaw_error + pitch_error + roll_error) / (total * 3)))
    preds=np.array(preds,np.int32)
    gts=np.array(gts,np.int32)
    prec=precision_score(gts,preds,average='macro')
    acc=accuracy_score(gts,preds)
    recall=recall_score(gts,preds,average='macro')
    f1score=f1_score(gts,preds,average='macro')
    print('prec',prec,'acc',acc,'recall',recall,'f1score',f1score)
    mae+=50*(1.0-f1score)
    print('mergescore:',mae)
    model.train()
    return mae

def writekeys(dct,filename):
    fp=open(filename,'w')
    for kk in dct.keys():
        fp.write(kk+'\n')
    if not hasattr(dct,'_metadata'):
        return
    for kk,vv in dct._metadata.items():
        fp.write(kk+':'+str(vv)+'\n')    
    fp.close()
def main():
    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
    b_scheduler = args.scheduler

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    summary_name = '{}_{}_bs{}'.format(
        'SixDRepNet', int(time.time()), args.batch_size)

    if not os.path.exists('output/snapshots/{}'.format(summary_name)):
        os.makedirs('output/snapshots/{}'.format(summary_name))

    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                        backbone_file='/data/6DRepNet-master/models/RepVGG-B1g2-train.pth',
                        deploy=False,
                        pretrained=True)
 
    if not args.snapshot == '':
        saved_state_dict = torch.load(args.snapshot)
        writekeys(saved_state_dict,'orimodel.txt')
        writekeys(model.state_dict(),'newmodel.txt')
        model.load_state_dict(saved_state_dict)

    print('Loading data.')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    # transformations = transforms.Compose([transforms.RandomResizedCrop(size=224,scale=(0.8,1)),
    #                                       transforms.ToTensor(),
    #                                       normalize])
    transformations=transforms.Compose([
        RotateImage(),
        ScaleImage(),
        EnhanceCVX(),
        GrayCV(),
        letterBox(224),
        InputToTensor(),
        Normlize()
    ])
    val_transform=transforms.Compose([
        letterBox(224),
        InputToTensor(),
        Normlize()
    ])

    pose_dataset = datasets.getDataset(
        args.dataset, args.data_dir, args.filename_list, transformations)
    numworks=10
    train_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=numworks)
    val_dataset=datasets.getDataset(args.dataset,args.vdata_dir,args.filename_list,val_transform)
    vtrain_loader=torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False,num_workers=numworks)

    model.cuda(gpu)
    crit = GeodesicLossEx().cuda(gpu) #torch.nn.MSELoss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)


    #milestones = np.arange(num_epochs)
    milestones = [10, 20]
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #    optimizer, milestones=milestones, gamma=0.5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,num_epochs,1e-6)
    mae=Validate(model,vtrain_loader)
    #mae=-1
    print('Starting training.')
    allcnt=0
    minmae=mae
    for epoch in range(num_epochs):
        loss_sum = .0
        iter = 0
        if epoch>=num_epochs-2:
            break
        for i, (images, gt_mat, _, _) in enumerate(train_loader):
            iter += 1
            allcnt+=1
            #continue
            images = torch.Tensor(images).cuda(gpu)
            
            # Forward pass
            pred_mat = model(images)
            weight=torch.zeros([gt_mat.shape[0]])
            msk=gt_mat.abs().sum(-1).sum(-1)==0
            weight[~msk]=1
            # Calc loss
            loss = crit(gt_mat.cuda(gpu), pred_mat,weight.cuda(gpu))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

            if (i+1) % 100 == 0: #100
                print('Epoch [%d/%d], lr:[%f] Iter [%d/%d] Loss: '
                      '%.6f' % (
                          epoch+1,
                          num_epochs,
                          optimizer.param_groups[0]['lr'],
                          i+1,
                          len(pose_dataset)//batch_size,
                          loss.item(),
                      ))
            if allcnt%450==0: #1000
                mae=Validate(model,vtrain_loader)
                print('Taking snapshot...',
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                    }, 'output/snapshots/' + summary_name + '/' + args.output_string +
                        '_epoch_' + str(epoch+1) + '.tar')
                    )
                if(minmae<0 or mae<minmae):
                    minmae=mae
                    print("-------------------this is best-----------------------")
                    torch.save({'model_state_dict': model.state_dict()},'output/best.pth')
            #break
        
        scheduler.step()
def main2():
    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
    b_scheduler = args.scheduler

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    summary_name = '{}_{}_bs{}'.format(
        'SixDRepNet', int(time.time()), args.batch_size)

    if not os.path.exists('output/snapshots/{}'.format(summary_name)):
        os.makedirs('output/snapshots/{}'.format(summary_name))

    model = SixDRepNet3(backbone_name='RepVGG-B1g2',
                        backbone_file='/data/6DRepNet-master/models/RepVGG-B1g2-train.pth',
                        deploy=False,
                        pretrained=True)
 
    if not args.snapshot == '':
        saved_state_dict = torch.load(args.snapshot)
        writekeys(saved_state_dict,'orimodel.txt')
        writekeys(model.state_dict(),'newmodel.txt')
        model.load_state_dict(saved_state_dict)

    print('Loading data.')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    # transformations = transforms.Compose([transforms.RandomResizedCrop(size=224,scale=(0.8,1)),
    #                                       transforms.ToTensor(),
    #                                       normalize])
    transformations=transforms.Compose([
        RotateImage(),
        ScaleImage(),
        EnhanceCVX(),
        GrayCV(),
        letterBox(224),
        InputToTensor(),
        Normlize()
    ])
    val_transform=transforms.Compose([
        letterBox(224),
        InputToTensor(),
        Normlize()
    ])

    pose_dataset = datasets.getDataset(
        args.dataset, args.data_dir, args.filename_list, transformations)
    numworks=10
    train_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=numworks)
    val_dataset=datasets.getDataset(args.dataset,args.vdata_dir,args.filename_list,val_transform)
    vtrain_loader=torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False,num_workers=numworks)

    model.cuda(gpu)
    crit = GeodesicLossEx().cuda(gpu) #torch.nn.MSELoss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)


    #milestones = np.arange(num_epochs)
    milestones = [10, 20]
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #    optimizer, milestones=milestones, gamma=0.5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,num_epochs,1e-6)
    mae=Validate2(model,vtrain_loader)
    #mae=-1
    print('Starting training.')
    allcnt=0
    minmae=mae
    celoss=nn.BCEWithLogitsLoss()
    for epoch in range(num_epochs):
        loss_sum = .0
        iter = 0
        if epoch>=num_epochs-2:
            break
        for i, (images, gt_mat, _, _) in enumerate(train_loader):
            iter += 1
            allcnt+=1
            #continue
            images = torch.Tensor(images).cuda(gpu)
            
            # Forward pass
            pred_mat,predcls = model(images)

            weight=torch.zeros([gt_mat.shape[0]])
            msk=gt_mat.abs().sum(-1).sum(-1)==0
            weight[~msk]=1
            gt=(~msk).float()
            lossce=celoss(predcls,gt.cuda(gpu))
            # Calc loss
            lossmae = crit(gt_mat.cuda(gpu), pred_mat,weight.cuda(gpu))
            loss=lossmae*0.5+lossce*0.5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

            if (i+1) % 1 == 0: #100
                print('Epoch [%d/%d], lr:[%f] Iter [%d/%d] Loss: '
                      '%.6f %.6f %.6f' % (
                          epoch+1,
                          num_epochs,
                          optimizer.param_groups[0]['lr'],
                          i+1,
                          len(pose_dataset)//batch_size,
                          loss.item(),lossmae.item(),lossce.item()
                      ))
            if allcnt%20==0: #450
                mae=Validate2(model,vtrain_loader)
                print('Taking snapshot...',
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                    }, 'output/snapshots/' + summary_name + '/' + args.output_string +
                        '_epoch_' + str(epoch+1) + '.tar')
                    )
                if(minmae<0 or mae<minmae):
                    minmae=mae
                    print("-------------------this is best-----------------------")
                    torch.save({'model_state_dict': model.state_dict()},'output/best.pth')
            #break
        
        scheduler.step()
if __name__ == '__main__':
    #main()
    main2()