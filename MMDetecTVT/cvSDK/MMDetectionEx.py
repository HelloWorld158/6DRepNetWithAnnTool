import BasicUseFunc as basFunc
import os,sys
import DataIO as dio
import time
import shutil
import cvExtendModel.ExtendMMDet
import BasicMMCocoDataSet
import MMDetectionTransFormEx
import torch
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.optim import OptimWrapper
from mmengine.registry import OPTIM_WRAPPERS
import BasicTorchOperator as basTch
import numpy as np
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
from mmdet.registry import RUNNERS
from mmengine.registry import DefaultScope
from mmengine.runner.runner import Runner
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union
import numpy as np
import torch
import torch.nn as nn
from mmcv.ops import RoIPool
from mmcv.transforms import Compose
from mmengine.config import Config
from mmengine.runner import load_checkpoint

from mmdet.evaluation import get_classes
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import get_test_pipeline_cfg
from mmengine.dist import is_distributed,collect_results,get_world_size,get_rank
'''
custom_hooks = [dict(type='DealModelPostProcess')]
'''
@HOOKS.register_module()
class DealModelPostProcess(Hook):
    priority='LOWEST'
    def __init__(self,sepDex=20,keys=[['loss'],['coco/bbox_mAP']],keysType=['epoch','step']
        ,by_epoch=True,title='MMDetectCurve',outfile='MMDetectResult',interval=1,
        best_score_key='best_score',evalkey='coco/bbox_mAP'):
        self.interval=interval        
        self.keysType=keysType
        self.keys=keys
        self.title=title
        self.outfile=outfile
        self.sepDex=sepDex
        self.time=time.time()
        self.by_epoch=by_epoch
        self.bestkey=best_score_key
        self.evalkey=evalkey
        self.bestpth=''
        self.distributed=is_distributed()
        self.syncdata=['syndata']
    def GetDeltaTime(self,curtimes,maxtimes):
        deltaTime=time.time()-self.time
        leftime=(deltaTime/curtimes)*(maxtimes-curtimes)
        leftime/=3600.0
        leftime=round(leftime,3)        
        return leftime
    def before_train(self, runner):
        self.loggerhook=None
        self.ckpthook=None
        for hook in runner.hooks:
            if hasattr(hook,'json_log_path'):
                self.loggerhook=hook
                continue
            if hasattr(hook,'filename_tmpl'):
                self.ckpthook=hook
                continue
        logdir=runner.log_dir
        self.workDir=os.path.dirname(logdir)
        self.logfile=os.path.join(logdir,'vis_data',self.loggerhook.json_log_path)
        return
    def SaveDct(self,runner):
        if hasattr(self.ckpthook,'best_ckpt_path') and self.bestpth!=self.ckpthook.best_ckpt_path\
            and self.ckpthook.best_ckpt_path is not None:
            if not os.path.exists(self.ckpthook.best_ckpt_path):
                print('====The best weight is in last workdir,but now not found it====')
                self.bestpth=self.ckpthook.best_ckpt_path
                print(f'====New Best PthFile:{self.bestpth}====')
                return
            shutil.copy(self.ckpthook.best_ckpt_path,os.path.join(self.workDir,'best.pth'))
            self.bestpth=self.ckpthook.best_ckpt_path
            print(f'====New Best PthFile:{self.bestpth}====')
        if self.bestkey not in runner.message_hub.runtime_info:
            return
        best_score = runner.message_hub.get_info(self.bestkey)
        dctlst=[{self.evalkey:best_score}]
        dctlst.append({'End':'end'})
        jsfile=os.path.join(self.workDir,'resultEval.json')
        dio.writejsondiclstFilelines(dctlst,jsfile)
        return
    def plot_curve(self,log_dicts,deltaTime):        
        for j, metric in enumerate(self.keysType):
            fig=plt.figure(figsize=(20,20),dpi=80) 
            xs = []
            curkeys=self.keys[j]
            ys = {key:[] for key in curkeys}
            for i, log_dict in enumerate(log_dicts):
                conflag=True
                for _,key in enumerate(curkeys):
                    if key not in log_dict.keys():
                        conflag=False
                        break
                if metric not in log_dict.keys():
                    conflag=False
                if not conflag:continue
                if log_dict[metric] not in xs:
                    xs.append(log_dict[metric])
                    dex=None
                else:
                    dex=xs.index(log_dict[metric])
                for _,key in enumerate(curkeys):
                    if dex is None:
                        ys[key].append(log_dict[key])
                    else:
                        ys[key][dex]=log_dict[key]
            if(len(xs)==0):continue
            xs=np.array(xs,np.int32)
            yMax,yMin=None,None
            for kk,vv in ys.items():
                ys[kk]=np.array(vv,np.float32)
                if yMax is None:
                    yMax=ys[kk].max()
                    yMin=ys[kk].min()
                else:
                    yMax=max(yMax,ys[kk].max())
                    yMin=min(yMin,ys[kk].min())
            spDex=min(len(xs),self.sepDex)
            x_major_locator=MultipleLocator(int((np.max(xs)-np.min(xs))/spDex+1))
            ax = plt.gca()
            ax.xaxis.set_major_locator(x_major_locator)
            if(yMax==yMin):
                yMax=yMin+0.1
                yMin-=0.01
            y_major_locator=MultipleLocator((yMax-yMin)/spDex)
            ax.yaxis.set_major_locator(y_major_locator)
            plt.ylim(yMin,yMax)
            plt.xlim(np.min(xs)-1,np.max(xs)+1)  
            plt.xlabel(metric)
            for kk,vv in ys.items():
                lab=kk
                plt.plot(
                    xs, vv, label=lab, linewidth=2)
                plt.legend()
            plt.title(self.title+' '+str(deltaTime)+' h')
            plt.savefig(os.path.join(self.workDir,self.outfile+'Train_'+str(j)+'.jpg'))
            plt.cla()
            plt.close(fig)
        return
    def after_val_epoch(self,runner,**kwargs):
        if not self.by_epoch:
            return
        if not self.every_n_epochs(runner,self.interval):
            return
        if not os.path.exists(self.logfile):
            return
        if self.distributed:
            count=get_world_size()
            curank=get_rank() 
            data=collect_results(self.syncdata,count,device='cpu') 
            if curank!=0:return  
            assert(len(data)==count)
        deltaime=self.GetDeltaTime(runner.iter,runner.max_iters)
        log_dicts=dio.getjsdatlstlindct(self.logfile)
        self.plot_curve(log_dicts,deltaime)
        self.SaveDct(runner)
        return
'''
optim_wrapper = dict(
    type='OptimNoneWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(norm_type=2,max_norm=35))
'''
@OPTIM_WRAPPERS.register_module(force=True)
class OptimNoneWrapper(OptimWrapper):
    def step(self, **kwargs) -> None:
        return
    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        self._inner_count += 1
'''
custom_hooks = [dict(type='GradLookHook',interval=1)]
'''
@HOOKS.register_module(force=True)
class GradLookHook(Hook):
    def __init__(self,interval,gradMode='SimpleOutGrad',outMode='SimpleOutGradOutDex'
        ,**kwargs):
        super().__init__()
        self.interval=interval
        self.outgrad={}
        self.cnt=0
        self.gradMode=gradMode
        self.kwargs=kwargs
        self.outMode=outMode
    def before_train_epoch(self,runner):
        basTch.ShowModulelstToFile('showmodel.txt',runner.model)
        self.module2id,self.id2module=basTch.GenerateModuleDct(runner.model)        
        return
    def before_train_iter(self, runner,batch_idx,data_batch=None):
        self.cnt+=1
        if self.cnt%self.interval!=0:return
        return
    def SimpleOutGrad(self,name,param,dctlst,runner):
        mn=param.detach().cpu().numpy()
        mn=np.abs(mn)
        st=float(mn.std())
        mn=float(mn.mean())
        val=[mn,st] if param.requires_grad else [0,0]
        dctlst[name]=val
    def SimpleOutGradOutEx(self,dctlst,runner):
        dio.writejsondictFormatFile(dctlst,'supportools/gradmeanstd.json')
    def SimpleOutGradOutDex(self,dctlst,runner):
        dio.writejsondictFormatFile(dctlst,'supportools/gradmeanstd_'+str(self.cnt)+'.json')
    def LowerSimpleOutGrad(self,name,param,dctlst,runner):
        mn=param.detach().cpu().numpy()
        mn=np.abs(mn)
        st=float(mn.std())
        mn=float(mn.mean())
        val=[mn,st] if param.requires_grad else [0,0]
        if mn>self.kwargs.get('minthr',1e-3):return
        dctlst[name]=val
    def after_train_iter(self, runner,outputs,batch_idx,data_batch=None):        
        if self.cnt%self.interval!=0:return
        dctlst={}
        cmd='(name,param,dctlst,runner)'
        for name,param in runner.model.named_parameters():
            exec('self.'+self.gradMode+cmd)
        cmd='(dctlst,runner)'
        exec('self.'+self.outMode+cmd)        
        return
'''
冻结网络
# 使用优化器可以冻结网络, 但BN层还无法冻结
custom_hooks = [dict(type='FreezeBatchNormHook')]
'''
@HOOKS.register_module()
class FreezeBatchNormHook(Hook): 
    def before_train_iter(self,runner,batch_idx,data_batch=None):
        for m in runner.model.modules():
            # trick: eval have effect on BatchNorm only
            if isinstance(m, _BatchNorm):
                m.eval()
'''
'''
@RUNNERS.register_module()
class RunnerSplitGPU(Runner):
    def dump_config(self) -> None:
        return 
