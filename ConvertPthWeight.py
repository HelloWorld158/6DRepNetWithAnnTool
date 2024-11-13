import os,sys
import torch
from mmengine.runner.checkpoint import _load_checkpoint

def ChangeDict(dct):
    ndct={}
    return ndct
file ='FineTune/tracktor_reid_r50_iter25245-a452f51f.pth'
outfile='FineTune/convert.pth'
dct=_load_checkpoint(file,'cpu')
ndct=ChangeDict(dct)
torch.save(ndct,outfile)