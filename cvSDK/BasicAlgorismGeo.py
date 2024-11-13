import BasicUseFunc as basFunc
import numpy as np
import os
import sys
import glob
import struct
import random
import cv2
#输入maskNum[N,N],每一个id就是一个数字转换成[N,N,m]
def MaskNumConvert(maskNum,minDex=None,maxDex=None):
    msklst=[]
    maxs=np.max(maskNum)
    mins=np.min(maskNum)
    if(minDex is None):
        minDex=mins+1
    if(maxDex is None):
        maxDex=maxs+1
    if(minDex>=maxDex):
        return None,minDex,maxDex
    for i in range(minDex,maxDex):
        mask=maskNum==i
        mask=np.expand_dims(mask,len(mask.shape))
        msklst.append(mask)
    return np.concatenate(msklst,-1),minDex,maxDex
#注意mask是[N,M]
def MaskConvert(mask):
    msk=np.expand_dims(mask,len(mask.shape))
    return msk
##注意mask的shape是[3,3,1]
def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = []
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
            boxes.append([x1,y1, x2,y2])
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        
    return boxes
def ioubox(box1, box2):
    '''
    两个框（二维）的 iou 计算
    
    注意：边框以左上为原点
    
    box:[top, left, bottom, right]
    '''
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h<0 or in_w<0 else in_h*in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    if(union==0): return -1
    iou = inter / union
    return iou
#[N,1,4]与[1,M,4]之间的每一个iou俩俩计算问题ltrb
def ComputeGroupIOU(aboxes,bboxes):
    aboxes=aboxes.reshape([-1,1,4])
    bboxes=bboxes.reshape([1,-1,4])
    whaboxes=(aboxes[...,3]-aboxes[...,1])*(aboxes[...,2]-aboxes[...,0])
    whbboxes=(bboxes[...,3]-bboxes[...,1])*(bboxes[...,2]-bboxes[...,0])
    areaboxes=whaboxes+whbboxes
    aboxes=np.repeat(aboxes,bboxes.shape[1],axis=1)
    bboxes=np.repeat(bboxes,aboxes.shape[0],axis=0)
    mboxes=np.max([aboxes[...,:2],bboxes[...,:2]],axis=0)
    xboxes=np.min([aboxes[...,2:],bboxes[...,2:]],axis=0)
    dboxes=xboxes-mboxes
    dboxes=np.where(dboxes>0,dboxes,0)
    inter=dboxes[...,0]*dboxes[...,1]
    iou=inter/(areaboxes-inter)
    return iou  
def iouheightwidth(hwbox1, hwbox2):
    box1=hwbox1.copy()
    box2=hwbox2.copy()
    box1[0]-=box1[2]/2
    box1[1]-=box1[3]/2
    box2[0]-=box2[2]/2
    box2[1]-=box2[3]/2
    box1[2]+=box1[0]
    box1[3]+=box1[1]
    box2[2]+=box2[0]
    box2[3]+=box2[1]
    iou=ioubox(box1,box2)
    return iou
def LtRbToxywh(ltrb):
    x=float(ltrb[0]+ltrb[2])/2.0
    y=float(ltrb[1]+ltrb[3])/2.0
    w=float(ltrb[2]-ltrb[0])
    h=float(ltrb[3]-ltrb[1])
    return [x,y,w,h]
def XywhToLtRb(xywh):
    x=xywh[0]
    y=xywh[1]
    w=xywh[2]
    h=xywh[3]
    mw=float(w)/2.0
    mh=float(h)/2.0
    l=x-mw
    r=x+mw
    t=y-mh
    b=y+mh
    return [l,t,r,b]
#像素平均交并比
def compute_mean_iou(pred, label,lstMatch):
    I = np.zeros(len(lstMatch))
    U = np.zeros(len(lstMatch))
    index=0
    for p,l in lstMatch:
        pred_i = pred == p  #“＝＝”号比“＝”的优先级高
        label_i = label == l

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))  #与 交
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))  #或 并
        index+=1
    mean_iou = np.mean(I / U)
    return mean_iou
 #误报像素比率统计
def compute_out_labelpix(pred,label,lstMatch):
    I = np.zeros(len(lstMatch))
    U = np.zeros(len(lstMatch))
    index=0
    for p,l in lstMatch:
        pred_i = pred == p  #“＝＝”号比“＝”的优先级高
        label_i = label == l
        pred_i[label_i]=False
        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))  #与 交
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))  #或 并
        index+=1
    falsepix = np.mean(I / U)
    return falsepix
def PointCheckRegion(left_top,right_top,right_bottom,left_bottom,point):
    left_top_x, left_top_y = left_top
    right_top_x, right_top_y = right_top
    right_bottom_x, right_bottom_y = right_bottom
    left_bottom_x, left_bottom_y = left_bottom
    x,y = point
    a = (left_top_x - left_bottom_x) * (y - left_bottom_y) - (left_top_y - left_bottom_y) * (x - left_bottom_x)
    b = (right_top_x - left_top_x) * (y - left_top_y) - (right_top_y - left_top_y) * (x - left_top_x)
    c = (right_bottom_x - right_top_x) * (y - right_top_y) - (right_bottom_y - right_top_y) * (x - right_top_x)
    d = (left_bottom_x - right_bottom_x) * (y - right_bottom_y) - (left_bottom_y - right_bottom_y) * (x - right_bottom_x)
    if (a > 0 and b > 0 and c > 0 and d > 0) or (a < 0 and b < 0 and c < 0 and d < 0):
        return True
    else:
        return False
def PointCheckInRegionRectangle(center,left,top,right,bottom):
    leftop=[left,top]
    rightop=[right,top]
    leftbottom=[left,bottom]
    rightbottom=[right,bottom]
    return PointCheckRegion(leftop,rightop,rightbottom,leftbottom,center)
def PointCheckInRegionXYWH(center,x,y,w,h):
    m=XywhToLtRb([x,y,w,h])
    return PointCheckInRegionRectangle(center,m[0],m[1],m[2],m[3])
def BoxCheckInRegionRectangle(box,boxRec):
    l,t,r,b=box
    pslst=[[l,t],[l,b],[r,t],[r,b]]
    for pt in pslst:
        bFlag=PointCheckInRegionRectangle(pt,boxRec[0],boxRec[1],boxRec[2],boxRec[3])
        if(not bFlag): return False
    return True
def BoxCheckInRegionRectangleT(box,boxRec):
    l,t,r,b=box
    pslst=[[l,t],[l,b],[r,t],[r,b]]
    for pt in pslst:
        bFlag=PointCheckInRegionRectangle(pt,boxRec[0],boxRec[1],boxRec[2],boxRec[3])
        if( bFlag): return True
    return False
def ClassCheck(pridct,real):
    num=min(len(pridct),len(real))
    TP=0
    FN=None
    TN=None
    FP=0
    for i in range(num):
        if(pridct[i]==real[i]):
            TP+=1
        if(pridct[i]!=real[i]):
            FP+=1
    return PrecisonTFNP.MakeDict(TP,FN,TN,FP)
def CheckTrueFalse(data,trfs,realTF):
    if(realTF):
        return data==trfs
    else:
        return (data!=0)==trfs
def TwoClassCheck(pridict,real,realTF=False):
    num=min(len(pridict),len(real))
    TP=0
    FN=0
    TN=0
    FP=0
    for i in range(num):
        if(pridict[i]==real[i] and CheckTrueFalse(real[i],True,realTF)):
            TP+=1
        if(pridict[i]==real[i] and CheckTrueFalse(real[i],False,realTF)):
            TN+=1
        if(pridict[i]!=real[i] and CheckTrueFalse(real[i],True,realTF)):
            FP+=1
        if(pridict[i]!=real[i] and CheckTrueFalse(real[i],False,realTF)):
            FN+=1
    return PrecisonTFNP.MakeDict(TP,FN,TN,FP)

class PrecisonTFNP(object):
    def __init__(self,CheckFunc=None):
        self.TP=0
        self.FN=0
        self.TN=0
        self.FP=0
        self.tpValid=False
        self.fnValid=False
        self.tnValid=False
        self.fpValid=False
        if(not CheckFunc):
            self.checkFunc=TwoClassCheck
        else:
            self.checkFunc=CheckFunc
    def MakeDict(TP,FN,TN,FP):
        return {'TP':TP,'FN':FN,'TN':TN,'FP':FP}
    def AddTFNPDirect(self,TP,FN,TN,FP):
        if(TP):
            self.tpValid=True
            self.TP+=TP
        if(FN):
            self.fnValid=True
            self.FN+=FN
        if(TN):
            self.tnValid=True
            self.TN+=TN
        if(FP):
            self.fpValid=True
            self.FP+=FP
    def AddTFNP(self,pridct,real,checkFunc=None):
        dct=dict()
        if(checkFunc):
            dct=checkFunc(pridct,real)
        else:
            dct=self.checkFunc(pridct,real)
        if(dct['TP']):
            self.tpValid=True
            self.TP+=dct['TP']
        if(dct['FN']):
            self.fnValid=True
            self.FN+=dct['FN']
        if(dct['TN']):
            self.tnValid=True
            self.TN+=dct['TN']
        if(dct['FP']):
            self.fpValid=True
            self.FP+=dct['FP']
    def OutPutTFNP(self,flush=True):
        acc=None
        if(self.fnValid and self.fpValid and self.tnValid and self.tpValid):
            acc=float(self.TP+self.TN)/float(self.TP+self.TN+self.FP+self.FN)
        pec=None
        if(self.tpValid and self.fpValid):
            prec=float(self.TP)/float(self.TP+self.FP)
        recall=None
        if(self.tpValid and self.fnValid):
            recall=float(self.TP)/float(self.TP+self.FN)
        if(flush):
            print('Accuracy:',acc,'Precision:',prec,'Recall:',recall)
        return {'acc':acc,'prec':prec,'recall':recall}
def GetOverlapRectangle(boxA,boxB):
    overlap=None
    iou=ioubox(boxA,boxB)
    if(iou>0):
        overlap=(max(boxA[0],boxB[0]),max(boxA[1],boxB[1]),min(boxA[2],boxB[2]),min(boxA[3],boxB[3]))
    return overlap
def GetOverlapBoxCenter(bxcA,bxcB):
    boxA=XywhToLtRb(bxcA)
    boxB=XywhToLtRb(bxcB)
    boxC=GetOverlapRectangle(boxA,boxB)
    if(boxC):
        boxC=LtRbToxywh(boxC)
    return boxC
def GetDistance(ptsA,ptsB):
    npA=np.array(ptsA,dtype=np.float32)
    npB=np.array(ptsB,dtype=np.float32)
    return np.linalg.norm(npA-npB)
#这个函数仅支持ltrb box
def RandomResizeArea(boxes,namelst,img,choosebx,minWidth=-1,minHeight=-1,maxWidth=-1,maxHeight=-1,iouThres=0.5):
    curbox=boxes[choosebx]
    [x,y,cw,ch]=LtRbToxywh(curbox)
    l,t=curbox[0],curbox[1]
    if(maxWidth<0):maxWidth=img.shape[1]
    if(maxHeight<0):maxHeight=img.shape[0]
    if(minWidth<0):minWidth=int(cw+1)
    minWidth=max(minWidth,int(cw+1))
    if(minHeight<0):minHeight=int(ch+1)
    minHeight=max(minHeight,int(ch+1))
    maxWidth=max(minWidth+20,maxWidth)
    maxHeight=max(minHeight+20,maxHeight)
    w=random.randint(minWidth,maxWidth)
    h=random.randint(minHeight,maxHeight)
    rw,rh=random.randint(0,w-cw),random.randint(0,h-ch)
    nl,nt=[min(img.shape[1]-1,max(0,l-rw)),min(img.shape[0]-1,max(0,t-rh))]
    nbox=[nl,nt,min(nl+w,img.shape[1]-1),min(nt+h,img.shape[0]-1)]
    finalBox=[]
    finalnames=[]
    for i in range(len(boxes)):
        b=boxes[i]
        if(BoxCheckInRegionRectangleT(b,nbox)):
            c=[max(b[0],nbox[0]),max(b[1],nbox[1]),min(b[2],nbox[2]),min(b[3],nbox[3])]
            iou=ioubox(b,c)
            if(iou>iouThres):
                c=[c[0]-nbox[0],c[1]-nbox[1],c[2]-nbox[0],c[3]-nbox[1]]
                finalBox.append(c)
                finalnames.append(namelst[i])
    return img[int(nbox[1]):int(nbox[3]),int(nbox[0]):int(nbox[2]),:],finalBox,finalnames

def BoxOverlapBoxCheck(boxlstA,boxlstB,iouThres):
    matchlst=[]
    flaglstA=np.zeros([len(boxlstA)],np.bool)
    flaglstB=np.zeros([len(boxlstB)],np.bool)
    for i in range(len(boxlstA)):
        maxiou=-1
        match=[]
        for k in range(len(boxlstA)):
            if(flaglstA[k]):continue
            for j in range(len(boxlstB)):
                if(flaglstB[j]):continue
                iou=iouheightwidth(boxlstA[k],boxlstB[j])
                if(iou>maxiou):
                    maxiou=iou
                    match=[k,j]
        if(maxiou>iouThres):
            matchlst.append(match)
            flaglstA[match[0]]=True
            flaglstB[match[1]]=True
    return matchlst
def CenterInPoly(point, region):
    """
    point:(x1,y1) 待判断的点
    region: [(x1,y1), (x2,y2),...] 多边形区域
    点在区域内部返回True，否则返回False
    """
    # 在检测区域内 1.0内部 0.0边缘 -1.0在区域外    
    in_bool = cv2.pointPolygonTest(np.array(region,np.float32).reshape(-1,2), (int(point[0]), int(point[1])), False)
    if in_bool == 1.0:
        return True
    else:
        return False
if __name__=='__main__':
    pridict=[0,0,1,1,0,1,0,1]
    real=[1,0,1,0,1,1,0]
    pt=PrecisonTFNP()
    pt.AddTFNP(pridict,real,ClassCheck)
    pt.AddTFNPDirect(None,2,None,3)
    s=pt.OutPutTFNP(True)
    #保留小数位数 
    #方法1：
    print("%.2f" % 0.13333)
    #方法2
    print("{:.2f}".format(0.13333))
    #方法3
    round(0.13333, 2)
    print('ok')
    spred=np.zeros([2,3],np.int32)
    spred=np.ma.make_mask(spred)
    spred=np.logical_and(spred,spred)
    
    g=np.zeros([30,30],np.int32)
    s=g!=0
    s[5:9,6:17]=True
    s=np.reshape(s,[s.shape[0],s.shape[1],1])
    m=extract_bboxes(s)
    print(s)
