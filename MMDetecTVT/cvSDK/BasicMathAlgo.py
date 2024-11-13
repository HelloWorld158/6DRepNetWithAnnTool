import BasicUseFunc as basFunc
import numpy as np
import os
import sys
import glob
import struct
def NCHWtoNHWC(nchw):
    nchw=np.array(nchw,dtype=np.float32)
    nhwc=np.transpose(nchw,(0,2,3,1))
    return nhwc
def NHWCtoNCHW(nhwc):
    nhwc=np.array(nhwc,dtype=np.float32)
    nchw=np.transpose(nhwc,(0,3,1,2))
    return nchw
#probReal, 产生事件的概率
'''
基于的原理是贝叶斯
C:事件产生
B:检测事件
P(C|B)=(P(B|C)P(C))/(P(B|C)P(C)+P(B|-C)P(-C))
其中P(B|C)是tp/(tp+fn)召回率
P(B|-C)是fp/(fp+tn) 事件不发生时检测的错误概率
'''
def GetRealWorldPrecFromCount(tp,fp,tn,fn,probReal):
    d=probReal
    dv=1.0-d
    return (tp/(tp+fn))*d/((tp/(tp+fn))*d+(fp/(fp+tn))*dv)
def GetRealWorldPrecFromPAR(prec,recall,acc,probReal):
    d=probReal
    dv=1.0-d
    a=recall
    b=prec
    c=acc
    return (a*d)/(a*d+((1-b)*(1-c)*a*dv)/(a+b*c-2*a*b+1e-20))
def GetRealWorldPrecFromNegsFullP(negs,full,prec,probReal):
    d=probReal
    ngs=negs/full
    return 1/(1+(1/prec-1)*(1/ngs-1)*(1/d-1))
#precReal期望获得的现场概率，下面这个函数结果是模型必须要达到的精确率
def GetModelPrecFromRealPrec(negs,full,precReal,probReal):
    neg=negs
    d=probReal
    dv=1.0-d
    res3=precReal
    return (full-neg)*dv*res3/(neg*(d-res3)+full*res3*dv)
'''
matches:(N,H,W,2,2)
A:(N,1,1,2,3,3)
Rt:(N,1,1,2,4,4)
or
matches:(H,W,2,2)
A:(1,1,2,3,3)
Rt:(1,1,2,4,4)
or
matches:(N,2,2)
A:(1,2,3,3)
Rt:(1,2,4,4)
'''
def ReconstructWithoutDistort(matches:np.ndarray,A:np.ndarray,Rt:np.ndarray):
    input=matches[...,None]
    if Rt.shape[-2]==4:
        Rt=Rt[...,:3,:]
    m=A@Rt
    bef=m[...,:2,:3]
    aft=m[...,[2,2],:3]

    vec=bef-aft*input  #(...,2,2,3)
    b0=matches[...,0]*m[...,2,3]-m[...,0,3]
    b1=matches[...,1]*m[...,2,3]-m[...,1,3]
    b=np.stack([b0,b1],-1)#(...,2,2)
    shape=vec.shape[:-3]
    vec=vec.reshape(list(shape)+[4,3])
    shape=b.shape[:-2]
    b=b.reshape(list(shape)+[4,1])
    shape=[i for i in range(len(vec.shape))]
    nshape=shape[:-2]+[shape[-1]]+[shape[-2]]
    tvec=np.transpose(vec,nshape)
    nvec=tvec@vec
    nb=tvec@b
    xyz=np.linalg.inv(nvec)@nb
    msk=np.isnan(xyz)
    imsk=np.isinf(xyz)
    allmsk=np.logical_or(msk,imsk)
    xyz[allmsk]=0
    return xyz
'''
xyz:H,W,N,3
A:H,W,N,3,3
Rt:H,W,N,3,4
'''
def ReprojectPoint(xyz:np.ndarray,A:np.ndarray,Rt:np.ndarray):
    m=A@Rt
    shape=list(xyz.shape[:-1])
    nshape=shape+[1]
    ones=np.ones(nshape,np.float32)
    nxyz=np.concatenate([xyz,ones],-1)[...,None]
    res=m@nxyz
    res=res.squeeze(-1)
    xy=res[...,:2]/res[...,2][...,None]
    return xy
    
if __name__=='__main__':
    a=[[[[1,2,3],[3,4,2]],[[5,6,1],[7,8,4]],[[9,10,2],[11,12,1]],[[5,6,1],[7,8,4]]],[[[5,2,2],[3,4,3]]
    ,[[5,6,1],[7,8,4]],[[9,10,5],[11,12,6]],[[5,6,1],[7,8,4]]]]
    b=NHWCtoNCHW(a)
    c=NCHWtoNHWC(b)
    pass
