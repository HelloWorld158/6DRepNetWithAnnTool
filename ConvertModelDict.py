import os,sys
import torch
import cvSDK.BasicUseFunc as basFunc
import cvSDK.DataIO as dio
from collections import OrderedDict
from copy import deepcopy



'''
orimodel.pth:其他代码训练好的权重
oridct:使用writekeys这个函数保存的orimodel.pth中权重和metadata,并写到orimodel.txt里面
writekeys(torch.load(orimodel,'cpu')['state_dict'],'orimodel.txt')
newdct:使用writekeys这个函数保存的MModel里面的state_dict如下所示
writekeys(model.state_dict(),'newmodel.txt')
保存好后,对比这俩文件在metadata前(带有':'前几行)保证行号对应,并且保证权重名称内容对应,如不对应请修改对应行号
"depth_net.encoder.encoder.conv1.weight"->"depthNet.backbone.conv1.weight"
"depth_net.encoder.encoder.bn1.weight"->"depthNet.backbone.bn1.weight"
"depth_net.encoder.encoder.bn1.bias"->"depthNet.backbone.bn1.bias"
然后使用python ConvertModelDict.py
'''
def writekeys(dct,filename):
    fp=open(filename,'w')
    for kk in dct.keys():
        fp.write(kk+'\n')
    if not hasattr(dct,'_metadata'):
        return
    for kk,vv in dct._metadata.items():
        fp.write(kk+':'+str(vv)+'\n')    
    fp.close()
orimodel='/data/6DRepNet-master/models/RepVGG-B1g2-train.pth'
oridct='/data/6DRepNet-master/newmodel.txt'
newdct='/data/6DRepNet-master/orimodel.txt'
save_state_dict=False
'''
save_state_dict将模型用dict方式存储到pytorch权重中
其中:
权重:state_dict
其他信息:meta
可以从orimodel中转出来
ConvertKey这个函数仅在save_state_dict=True的时候生效
'''
def ConvertKey(orimodel,resdct):
    '''
    orimeta=orimodel['meta']
    orimeta['mmdet_version']='3.0.0'
    orimeta['mmcv_version']='2.0.0'
    classes=deepcopy(orimeta['CLASSES'])
    del orimeta['CLASSES']
    orimeta['dataset_meta']={'classes':classes}
    resdct['meta']=orimeta
    '''
    return resdct




d,name,ftr=basFunc.GetfileDirNamefilter(orimodel)
newmodel=os.path.join(d,'newmodel.pth')
resdct=os.path.join(d,'convert.json')
def opendctxt(file):
    fp=open(file,'r')
    txts=fp.readlines()
    fp.close()
    modelkeys=[]
    metakeys=OrderedDict()
    for i in range(len(txts)):
        txts[i]=txts[i][:-1]
        dex=txts[i].find(':')
        if dex>=0:
            kk,vv=txts[i][:dex],txts[i][dex+1:]
            metakeys[kk]=eval(vv)
        else:
            modelkeys.append(txts[i])
    return modelkeys,metakeys
orikeys,orimeta=opendctxt(oridct)
newkeys,newmeta=opendctxt(newdct)
orimodelpth=torch.load(orimodel,'cpu')
modeldct=orimodelpth
nmodeldct=deepcopy(modeldct)
nmodeldct.clear()
if hasattr(nmodeldct,'_metadata'): nmodeldct._metadata.clear()
wdct={}
assert(len(orikeys)==len(newkeys))
for i in range(len(newkeys)):
    orikey=orikeys[i]
    newkey=newkeys[i]
    wdct[orikey]=newkey
    value=modeldct[orikey]
    nmodeldct[newkey]=value
if save_state_dict:
    res={'state_dict':nmodeldct}
    res=ConvertKey(orimodelpth,res)
else:
    res=nmodeldct
dio.writejsondictFormatFile(wdct,resdct)
torch.save(res,newmodel)
