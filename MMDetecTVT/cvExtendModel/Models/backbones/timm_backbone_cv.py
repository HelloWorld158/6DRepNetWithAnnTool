# Copyright (c) OpenMMLab. All rights reserved.
try:
    import timm
except ImportError:
    timm = None
import torch
import torch.nn as nn
from itertools import chain
from torch.nn.modules.batchnorm import _BatchNorm
from tabnanny import check
from unittest import result
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from torch.utils.checkpoint import checkpoint
from timm.layers.create_act import get_act_layer
from timm.layers.create_norm_act import _NORM_ACT_MAP
import torch.nn as nn
import os
'''
请自己到timm的执行目录下寻找权重地址,下载到FineTune下面
,这个backbone不管下载权重,谢谢理解
请使用supportools/LookupTimmNetWork.py查看Timm网络,
建议再MMDetectionConfig.py里面禁用NumClassCheckHook
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='TIMMBackboneCV',
        model_name='efficientnetv2_rw_s',
        outfeats=['timm_model.blocks.1',
            'timm_model.blocks.2','timm_model.blocks.3','timm_model.blocks.4',
            'timm_model.blocks.5'],
        norm_layer='synbatchnormcv',
        exedct=dict(before_file="norm.py"),
        checkpoint_path='FineTune/efficientnet_v2s_ra2_288-a6477665.pth'),
    neck=dict(
        type='FPN_CARAFE',
        in_channels=[48,64,128,160,272],
        out_channels=256,
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=classNum,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=focaloss,
        loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=2.0,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.5,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
norm.py:
import torch.nn as nn
def SynBnFunc(num_features,eps=0.001,momentum=0.003,affine= True, track_running_stats = True, 
        process_group= None, device=None, dtype=None,apply_act=True, act_layer=nn.ReLU, inplace=True, drop_layer=None):
    return SynBatchNormAct2d(num_features,eps=0.001,momentum=0.003,affine= True, track_running_stats = True, 
        process_group= None, device=None, dtype=None,apply_act=True, act_layer=nn.ReLU, inplace=True, drop_layer=None)
self.kwargs['norm_layer']=SynBnFunc
'''
class SynBatchNormAct2d(nn.SyncBatchNorm):
    def __init__(self, num_features, eps= 0.00001, momentum= 0.1, affine= True, track_running_stats = True, 
        process_group= None, device=None, dtype=None,apply_act=True, act_layer=nn.ReLU, inplace=True, drop_layer=None):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        act_layer = get_act_layer(act_layer)  # string -> nn.Module
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()
    def forward(self,input):
        bnres=super().forward(input)
        x=self.drop(bnres)
        x=self.act(x)
        return x
_NORM_ACT_MAP['synbatchnormcv']=SynBatchNormAct2d
@MODELS.register_module(force=True)
class TIMMBackboneCV(BaseModule):
    """Wrapper to use backbones from timm library. More details can be found in
    `timm <https://github.com/rwightman/pytorch-image-models>`_ .

    Args:
        model_name (str): Name of timm model to instantiate.
        pretrained (bool): Load pretrained weights if True.
        checkpoint_path (str): Path of checkpoint to load after
            model is initialized.
        in_channels (int): Number of input image channels. Default: 3.
        init_cfg (dict, optional): Initialization config dict
        **kwargs: Other timm & model specific arguments.
    """

    def __init__(
        self,
        model_name,
        pretrained=False,
        features_only=True,
        checkpoint_path='',
        norm_eval=True,
        frozen_layers=None,
        in_channels=3,
        useOutasFeats=False,
        exedct=None,
        outfeats=None,
        init_cfg=None,
        **kwargs,
    ):
        self.useOutasFeats=useOutasFeats
        self.norm_eval=norm_eval
        self.frozen_layers=frozen_layers
        self.exedct=exedct
        self.kwargs=kwargs
        self.RunExeDct('before_')
        if timm is None:
            raise RuntimeError('timm is not installed')
        super(TIMMBackboneCV, self).__init__(init_cfg)
        self.timm_model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            features_only=features_only,
            in_chans=in_channels,
            checkpoint_path=checkpoint_path,
            **self.kwargs,
        )
        if features_only:
            self.timm_model.global_pool = None
            self.timm_model.fc = None
            self.timm_model.classifier = None
        self.GenerateModuleDct()
        # reset classifier        
        self.outfeats={}
        self.outModule=outfeats
        def forward_hook(module, input, output):
            #print(module.__module__, len(input), len(output))
            #assert(len(output)==1)    
            if self.id2module[id(module)] not in self.outfeats.keys():
                self.outfeats[self.id2module[id(module)]] = [output]
            else:
                self.outfeats[self.id2module[id(module)]].append(output)
            return None
        if outfeats is None:
            self.OutModule()
        else:
            self.RegisterHook(outfeats,forward_hook)
        # Hack to use pretrained weights from timm
        if pretrained or checkpoint_path:
            self._is_init = True
        self._freeze_stages()
        self.RunExeDct('after_')
    def _freeze_stages(self):
        if self.frozen_layers is None:return
        for name,module in self.named_modules():
            if name not in self.frozen_layers:continue
            module.eval()
            for param in module.parameters():
                param.requires_grad=False
    def train(self, mode = True):
        super(TIMMBackboneCV,self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
        return 
    def RunExeDct(self,name,feats=None):
        if self.exedct is None:return None
        flag=False
        for kk in self.exedct:
            if kk.find(name)>=0:
                flag=True
        if not flag:return feats
        reskeys=self.exedct.get('reskeys',[])
        result=[]
        for kk,vv in self.exedct.items():
            if kk.find(name+'file')>=0:
                fp=open(self.exedct[kk],'r')
                txts=fp.read()
                fp.close()
                loc=locals()
                exec(txts)            
                break
            elif kk.find(name+'func')>=0:
                loc=locals()
                cmd=self.exedct[name+'func']
                exec(cmd)
                break
        for key in reskeys:
            result.append(loc[key])
        return result    
    def ClearHook(self):
        self.outfeats={}
    def OutModule(self):
        outfile=os.path.join(os.getcwd(),'timmbackbone.txt')
        print('out module file:',outfile)
        fp=open(outfile,'w')
        for name,module in self.named_modules():
            fp.write(name+':'+str(id(module))+'\n')
        fp.close()
    def GenerateModuleDct(self):
        self.module2id={}
        self.id2module={}
        for name,module in self.named_modules():
            if len(name)==0 or name =='module':continue
            self.module2id[name]=[id(module),module]
            self.id2module[id(module)]=name
    def RegisterHook(self,layernames,func):
        if self.useOutasFeats:return
        for name,module in self.named_modules():
            if name in layernames:
                module.register_forward_hook(func)
    def forwardef(self, x):
        if self.outModule is None:
            features = self.timm_model(x)
            features=[features]
        else:
            self.timm_model(x)
            features=[]
            for i,out in enumerate(self.outModule):
                feat=self.outfeats[out]
                assert(len(feat)==1)
                features.append(feat[0])
                print(out,feat[0].shape)
        return tuple(features)
    def CheckPointSeq(self,functions,x,every=1,flatten=1,skip_last=False,
        preserve_rng_state=True):
        def run_function(start, end, functions):
            def forward(_x):
                for j in range(start, end + 1):
                    _x = functions[j](_x)
                return _x
            return forward

        if isinstance(functions, torch.nn.Sequential):
            functions = functions.children()
        if flatten:
            functions = chain.from_iterable(functions)
        if not isinstance(functions, (tuple, list)):
            functions = tuple(functions)

        num_checkpointed = len(functions)
        if skip_last:
            num_checkpointed -= 1
        end = -1
        for start in range(0, num_checkpointed, every):
            end = min(start + every - 1, num_checkpointed - 1)
            x = checkpoint(run_function(start, end, functions), x, preserve_rng_state=preserve_rng_state)
        if skip_last:
            return run_function(end + 1, len(functions) - 1, functions)(x)
        return x
    def forward(self,x):
        self.ClearHook()
        out=self.timm_model(x)
        if not self.useOutasFeats:
            features=[]
            for i,out in enumerate(self.outModule):
                feat=self.outfeats[out]
                assert(len(feat)==1)
                features.append(feat[0])
        else:
            features=out
        if self.exedct is not None:
            features=self.RunExeDct('forward_',features)
        return tuple(features)
