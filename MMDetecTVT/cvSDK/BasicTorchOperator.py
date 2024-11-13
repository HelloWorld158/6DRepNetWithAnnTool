import BasicUseFunc as basFunc
import torch as tch
import torch.nn as nn
import torch.nn.functional as fn
import torchvision as tchv
import numpy as np
import random
import os,sys
def GetCudaMemFromDevice(device:str='0'):
    dev=tch.device('cuda:'+device)
    props = tch.cuda.get_device_properties(dev)
    total_memory = props.total_memory / (1024 ** 3)
    return total_memory
def GetCudaDeviceCnt():
    return tch.cuda.device_count()
def SetTorchSeed(seed=1024):
    if(seed<0):
        tch.backends.cudnn.benchmark = True
        tch.backends.cudnn.deterministic = False
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tch.manual_seed(seed)
    tch.cuda.manual_seed(seed)
    tch.cuda.manual_seed_all(seed)
    tch.backends.cudnn.benchmark = False
    tch.backends.cudnn.deterministic = True
    if hasattr(tch,'use_deterministic_algorithms'):
        import inspect
        signature = inspect.signature(tch.use_deterministic_algorithms)
        flag=False
        for parameter in signature.parameters.values():
            if(parameter.name=='warn_only'):
                flag=True
                break
        if not flag:return
        tch.use_deterministic_algorithms(True,warn_only=True)

#gpuid -1 自动适配一个GPU
#gpuid -n 自动适配n个GPU
#gpuid >=0 客户指定GPU
def ClientGvGPUs(gpuids):
    gpustr=''
    count=0
    lst=[]
    for id in gpuids:
        gpustr+=str(id)
        if(count!=len(gpuids)-1):
            gpustr+=','
        lst.append(count)
        count+=1
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES']=gpustr
        print('process:',os.getpid(),'os.environ:'
        ,os.environ['CUDA_VISIBLE_DEVICES'])
    else:
        print('warning CUDA_VISIBLE_DEVICES allready set device:'
        ,os.environ['CUDA_VISIBLE_DEVICES'],'process:',os.getpid())
    return lst
def SetGPUConfig(config):
    gpuflag=False
    gpulst,gpuCount=basFunc.GetAvailableGPUsList()
    if(gpulst==None or gpuCount==0):
        print('No GPU Device Avaliabel,find Device:',gpuCount)
        exit()
    if(len(config.gpuid)!=0):
        gpuflag=True
        clst=[]
        if(len(config.gpuid)==1):
            if(config.gpuid[0]<0):
                for i in range(min(len(gpulst),abs(config.gpuid[0]))):
                    clst.append(gpulst[i])
            if(config.gpuid[0]>=0):
                clst=config.gpuid
        else:
            clst=config.gpuid
        lst=ClientGvGPUs(clst)
    return lst,gpuflag
def LoadModelWeight(model,mdlpath):
    if(mdlpath and os.path.exists(mdlpath)):
        print('Weight Load Path:',mdlpath)
        model.load_state_dict(tch.load(mdlpath))
def SaveModelWeight(model,mdlpath,config,gvDataToCuda=True):
    if(config.gpuflag):
        tch.save(model.cpu().module.state_dict(),mdlpath)
        if(gvDataToCuda):
            model.cuda()
    else:
        tch.save(model.cpu().state_dict(),mdlpath)
def SaveModelWeightEx(model,mdlpath,config,gvDataToCuda=True,bstFlag=False):
    dir,name,ftr=basFunc.GetfileDirNamefilter(mdlpath)
    epcpath=os.path.join(dir,name+str(config.epcho)+ftr)
    SaveModelWeight(model,epcpath,config,False)
    if(bstFlag):
        bstpath=os.path.join(dir,'bst_'+name+ftr)
        SaveModelWeight(model,bstpath,config,False)
    SaveModelWeight(model,mdlpath,config,gvDataToCuda)
#输入的loss必须是转换为CPU版本后的
def ConvertTorchLoss(loss):
    loss=[loss.cpu().detach().numpy().tolist()].copy()
    return loss[0]
def DBGTensor(tensor):
    tensor=[tensor.cpu().detach().numpy().tolist()].copy()
    return tensor[0]
def DataLoadProcess(step,count):
    print(step,'/',count,end=' ')
def DataLoadProcessEnd():
    print('          ',end='\r',flush=True)
def GetMinLoss(loss,minLoss):
    if(minLoss<0 or loss<minLoss):
        minLoss=loss
        return minLoss,True
    return minLoss,False
#    backbonePath=os.path.join(os.getcwd(),'resnet34-333f7ec4.pth')
    #    ylctmdl.backbone.load_state_dict(tch.load(backbonePath))
    #backboneParams=list(map(id,model.yolact.backbone.parameters()))
    #otherParams=filter(lambda p:id(p) not in backboneParams,model.parameters())
    #params=[{'params':otherParams},
    #        {'params':model.yolact.backbone.parameters(),'lr':cfg.backBoneLr,'momentum':cfg.backBoneMR}]
    #params=model.parameters()
    #optimizer=tch.optim.SGD(params,lr=cfg.learningrate,
     #                       momentum=cfg.momument,weight_decay=cfg.weightdecay)
     #网络层里面修改
     #sf.backbone=tchv.models.resnet34(pretrained=False)
def GetModuleLayersFromNames(net,names):
    layers=nn.ModuleList([])#注意初始化模型时必须用modulelist，但是运行的tensor可以直接压入list里面
    layersNames=[]
    for i in range(len(names)):
        bFlag=False
        curName=None
        curLayer=None
        for name,layer in net._modules.items():                    
            if(name==names[i]):
                bFlag=True
                curName=name
                curLayer=layer
                break
        if(not bFlag): continue
        layers.append(curLayer)
        layersNames.append(curName)
    return layers,layersNames
def GetArrangeModuleLayersFromDexs(net,dexlist):
    layers=nn.ModuleList([])
    for i in range(len(dexlist)):
        dex=dexlist[i]
        if(not dex or len(dex)==0):continue        
        netlst=list(net.children())[dex[0]:dex[1]]
        for j in range(len(netlst)):
            layers.append(netlst[j])
    return layers
def TrainOnlyRequireGradParams(model):
    return filter(lambda p:p.requires_grad,model.parameters())
def CheckParamTrainState(model):
    for name,param in model.named_parameters():
        print('name:',name,',trainstate:',param.requires_grad)
def FreezeLoopWithName(name,param,module):
    param.requires_grad=False
def UnFreezeLoopWithName(name,param,module):
    param.requires_grad=True
#names是字符串的list
def LoopParamWithName(model,names,loopfunc):
    module=None
    if(isinstance(model,nn.parallel.DataParallel)):
        module=model.module
    if(isinstance(model,nn.Module)):
        module=model
    if(not module):
        print('Not Support Model Type,the input model must be nn.parallel.DataParallel or nn.Module')
        exit()
    for name,param in module.named_parameters():
        for i in range(len(names)):
            n=names[i]
            if(name.find(n)!=-1):
                loopfunc(name,param,module)
                break
    return
def FreezeLoopWithDexs(param,module):
    param.requires_grad_(False)
def UnFreezeLoopWithDexs(param,module):
    param.requires_grad_(True)
def CheckDexList(dexlst):
    if(len(dexlst)!=2):return False
    if(isinstance(dexlst[0],int) and isinstance(dexlst[1],list)):
        return True
    return False
def RecursionDex(dexlst,childrenlst,loopfunc,module):
    for i in range(len(dexlst)):
        dex=dexlst[i]
        if(isinstance(dex,list) and len(dex)==2  and isinstance(dex[0],int) and isinstance(dex[1],list)):
            if(CheckDexList(dex[1])):
                RecursionDex([dex[1]],list(childrenlst[dex[0]].children()),loopfunc,module)
            else:
                RecursionDex(dex[1],list(childrenlst[dex[0]].children()),loopfunc,module)
        if(isinstance(dex,int)):
            loopfunc(childrenlst[dex],module)
    return
#dexlst示例[[0   ,       [0,1]  ],3]
            #第一个child  子child  可以不带child
#外面必须是list，要想使用子child，这个元素必须是list，并且该list第一个元素必须是数字,第二个元素必须是列表
#LoopParamWithDexs(m,[
#                        [0,
#                            [0,
#                                [0,1]
#                            ]
#                        ],
#                        2
#                    ],FreezeLoopWithDexs)
def LoopParamWithDexs(model,dexlst,loopfunc):
    module=None
    if(isinstance(model,nn.parallel.DataParallel)):
        module=model.module
    if(isinstance(model,nn.Module)):
        module=model
    if(not module):
        print('Not Support Model Type,the input model must be nn.parallel.DataParallel or nn.Module')
        exit()
    for i in range(len(dexlst)):
        RecursionDex([dexlst[i]],list(module.children()),loopfunc,module)
    return
def CompareCaculateTensor(tensor,file,fDir=None,cmpThr=0.00001):
    if(fDir is not None):
        file=os.path.join(fDir,file)
    out=tensor
    if(os.path.exists(file)):
        cmp=np.load(file)
        delta=np.abs(cmp-out)
        msk=delta>cmpThr
        arr=np.where(msk)
        print('cmpdiff:',cmp[arr])
        print('outdiff:',out[arr])
    np.save(file,out)
def GenerateModuleDct(model,exclude=['module']):
    module2id={}
    id2module={}
    for name,module in model.named_modules():
        if len(name)==0 or name in exclude:continue
        module2id[name]=[id(module),module]
        id2module[id(module)]=name
    return module2id,id2module
def ShowModulelstToFile(file,model):
    fp=open(file,'w')
    print('out model file:',file)
    for name,module in model.named_modules():
        fp.write(name+':'+str(id(module))+'\n')
    fp.close()
    return
'''
获取图像质量函数
img:H,W,C or N,H,W,C
图像范围[0,255]
'''
def TorchImageQuality(img:np.ndarray):
    curimg=tch.from_numpy(img)
    if len(curimg.shape)==3:
        curimg=curimg[None,...]
    curimg=curimg.permute(0,3,1,2).float()
    curimg/=255.0
    import piq
    res=piq.brisque(curimg)
    res=float(res.cpu().numpy())
    return res
def NumpyEqual(arr0,arr1,checksame=True, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    shape0=np.array(arr0.shape,np.int32)
    shape1=np.array(arr1.shape,np.int32)
    shapes=np.array_equal(shape0,shape1)
    if not shapes:
        if not checksame:
            return False
        return False,False
    if not checksame:
        return True
    flag= np.allclose(arr0,arr1,rtol,atol,equal_nan)
    return True,flag
def TensorEqual(arr0,arr1,checksame=True,rtol=1.e-5, atol=1.e-8, equal_nan=False):
    shape0=np.array(arr0.shape,np.int32)
    shape1=np.array(arr1.shape,np.int32)
    shapes=np.array_equal(shape0,shape1)
    if not shapes:
        if not checksame:
            return False
        return False,False
    if not checksame:
        return True
    flag=tch.allclose(arr0,arr1,rtol,atol,equal_nan)
    return True,flag    
if __name__ =='__main__':
    pthPath='E:\\backbone\\pytorch版本\\resnet50-19c8e357.pth'
    model=tchv.models.resnet50(pretrained=False)
    LoadModelWeight(model,pthPath)
    pass