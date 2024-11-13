import os,sys
import albumentations as alb
import cv2
import numpy as np
import random
from scipy.spatial.transform import Rotation
# def EnhanunceCV(img:np.ndarray)->np.ndarray:
#     aug=alb.Compose([
#         alb.OneOf([
#             alb.MotionBlur((9,19),p=1.0),
#             alb.Defocus(alias_blur=(0.3,0.6),p=1),
#             alb.CLAHE(p=1)
#         ],p=0.5),
#         alb.ISONoise(p=0.6),
#         alb.RGBShift(p=0.6),
#         alb.Emboss(p=0.5),
#         alb.Downscale(0.1,0.3,p=1.0)
#     ])
#     img=aug(image=img)['image']
#     return img
def EnhanunceCV(img:np.ndarray)->np.ndarray:
    if(random.uniform(0,1)>=0.5):
        aug=alb.Compose([
            alb.OneOf([
                alb.MotionBlur((9,19),p=1.0),
                alb.Defocus(alias_blur=(0.3,0.6),p=1),
                alb.CLAHE(p=1)
            ],p=0.5),
            alb.ISONoise(p=0.6),
            alb.RGBShift(p=0.6),
            alb.Emboss(p=0.5),
            alb.Downscale(0.1,0.3,p=1.0)
        ])
    else:
        aug=alb.Compose([
            alb.Downscale(0.04,0.06,p=1.0)
        ])
    img=aug(image=img)['image']
    return img
def EnhanunceCVDown(img:np.ndarray):
    aug=alb.Compose([
            alb.Downscale(0.04,0.06,p=1.0)
        ])
    img=aug(image=img)['image']
    return img
def EnhanunceCVless(img:np.ndarray)->np.ndarray:
    aug=alb.Compose([
        alb.OneOf([
            alb.MotionBlur((5,9),p=1.0),
            alb.Defocus(alias_blur=(0.2,0.4),p=1),
            alb.CLAHE(p=1)
        ],p=0.5),
        alb.ISONoise(p=0.6),
        alb.RGBShift(p=0.6),
        alb.Emboss(p=0.5),
        alb.Downscale(0.1,0.4,p=1.0)
    ])
    img=aug(image=img)['image']
    return img
def Gray(img:np.ndarray,ratio=0.2):
    if(random.uniform(0,1)>=ratio): return img
    nimg=img.mean(-1)
    nimg=nimg[...,None]
    nimg=np.repeat(nimg,3,-1)
    nimg=nimg.astype(np.uint8)
    return nimg
def RotateCV(img:np.ndarray,gt,centers=None):
    if(random.uniform(0,1)>=0.5):
        angle=random.uniform(30,180)
    else:
        angle=random.uniform(-180,-30)
    #angle=120
    if(centers is not None):
        cenx,ceny=centers
    else:
        cenx,ceny=img.shape[1]/2,img.shape[0]/2
    #vecs=np.array([[1,0,0],[0,1,0]],np.float32)
    # outvecs=gt@vecs #3x2
    matrix=cv2.getRotationMatrix2D((cenx,ceny),angle,1)
    rot=matrix[:2,:2]
    nrot=np.eye(3,dtype=np.float32)
    nrot[:2,:2]=rot
    #noutvecs=outvecs.transpose([1,0])
    ngt=nrot@gt
    img=cv2.warpAffine(img,matrix,(img.shape[1],img.shape[0]))
    return img,ngt,angle


