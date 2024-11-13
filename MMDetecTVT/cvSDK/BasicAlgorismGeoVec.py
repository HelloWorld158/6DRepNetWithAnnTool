import BasicUseFunc as basFunc
import numpy as np
import os
import sys
import glob
import struct
def DistanceSeg(sf,center,linePtsA,linePtsB):
    ptsA=np.array(linePtsA,dtype=np.float32)
    ptsB=np.array(linePtsB,dtype=np.float32)
    cen=np.array(center,dtype=np.float32)
    AB=ptsB-ptsA
    cenA=cen-ptsA
    d=np.linalg.norm(AB)
    t=np.dot(cenA,AB/d)
    if(d>0):
        t/=d
    if(t<0): t=0
    elif(t>1): t=1
    l=ptsA+t*AB-cen
    return np.linalg.norm(l)
def NP(a):
    return np.array(a,dtype=np.float32)
def NormVec(vec):
    vec=NP(vec)
    return vec/np.linalg.norm(vec)
def NormVecPts(ptsA,ptsB):
    AB=NP(ptsB)-NP(ptsA)
    return NormVec(AB)
def NormalizeVec(vec):
    return np.linalg.norm(NP(vec))
def CosVec(vecA,vecB):
    return (NP(vecA).dot(NP(vecB)))/(NormalizeVec(vecA)*NormalizeVec(vecB))
if __name__=='__main__':
    pass