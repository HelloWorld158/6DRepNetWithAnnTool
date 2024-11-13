from typing import Any
from enhaunce import *
import cvSDK.BasicUseFunc
import cvSDK.BasicPicDeal as baspic
import random
from torchvision import transforms
from utils import *
import albumentations as alb
import numpy as np
class letterBox:
    def __init__(self,imgsize=224):
        self.imgsize=imgsize
    def __call__(self, img,gt=None):
        img=baspic.GenerateExpandImageData(img,self.imgsize,self.imgsize)
        if(gt is not None):
            return img,gt
        return img
class EnhanceCVX:
    def __call__(self,img,gt=None):
        if random.uniform(0,1)<0.6:
            img=EnhanunceCV(img)
        if(gt is not None):
            return img,gt
        return img
class GrayCV:
    def __call__(self,img,gt=None):
        if random.uniform(0,1)<0.2:
            img=Gray(img)
        if(gt is not None):
            return img,gt
        return img
class RotateImage:
    def __call__(self,img,gt=None):
        if(random.uniform(0,1)<0.4):  
            return img,gt
        outelar=compute_euler_angles_from_rotation_matricesnpy(gt[None,...])[0]
        pitch,yaw,roll=outelar.tolist()
        rot=get_R(-pitch,yaw,-roll)
        rott=np.transpose(rot,[1,0])
        img,ngt,angle=RotateCV(img,rott)
        ngtt=np.transpose(ngt,[1,0])
        outelar=compute_euler_angles_from_rotation_matricesnpy(ngtt[None,...])[0]
        npitch,nyaw,nroll=outelar.tolist()
        npitch=-npitch
        nroll=-nroll
        finalgt=get_R(npitch,nyaw,nroll)
        #debugfile="temp.jpg"
        #img=draw_axis(img,nyaw*180/np.pi,npitch*180/np.pi,nroll*180/np.pi,None,None)
        #cv2.imwrite(debugfile,img)
        return img,finalgt
class ScaleImage:
    def __init__(self,scales=[0.8,1.2]): 
        self.scales=scales
        
    def __call__(self,img,gt=None):
        if(random.uniform(0,1)<0.3):
            scale=random.uniform(self.scales[0],self.scales[1])
            mat=cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),0,scale)
            img=cv2.warpAffine(img,mat,(img.shape[1],img.shape[0]))
        if(gt is not None):
            return img,gt
        return img
class InputToTensor:
    def __init__(self) -> None:
        self.trans=transforms.ToTensor()
    def __call__(self,img,gt=None):
        img=torch.from_numpy(img)
        img=img.permute([2,0,1]).float()
        if(gt is not None):
            return img,gt
        return img
class Normlize:
    def __init__(self) -> None:
        self.normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    def __call__(self,img,gt=None):
        img=self.normalize(img)
        if(gt is not None):
            return img,gt
        return img