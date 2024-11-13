import os,sys
from mmdet.models import detectors, losses
from mmdet.models.detectors import RetinaNet
from mmdet.registry import MODELS
import torch.nn as nn
import torch
from mmengine.model.weight_init import normal_init
from cvExtendModel.bricks.sppmodule import SPPModule
from mmdet.models.backbones.resnet import Bottleneck
import numpy as np
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.utils import unpack_gt_instances

'''
注意这个模型结构需要配合'EhlXMLCocoClassifyDataSet','EhlJsonCocoClassifyDataSet'俩个Dataset
请勿使用其他Dataset,现阶段这个模块仅支持单阶段retinaNet，其他单阶段网络恕不支持。
这个模型不能直接使用InferMMDet.py,需要修改
def ConvertGeneralBox(result):
    res=[]
    arr=[]
    namelst=[]
    for i,r in enumerate(result):
        if(len(r.shape)!=1):
            for m in r:
                res.append([model.CLASSES[i],float(m[4]),list(m[:4])])
        else:
            namelst.append(model.CLASSES[i])
            arr.append(r[...,4])
    arr=np.array(arr,np.float32)
    dex=np.argmax(arr)
    nres=[]
    for i in range(len(result)):
        if(len(result[i].shape)==1):continue
        nres.append(result[i])
    return res,nres,namelst[dex],arr[dex]
res,nres,clss,score=ConvertGeneralBox(result)
import BasicDrawing as basDraw
添加最后一行代码：
model.show_result(file,nres,score_thr=0,show=False,out_file=debugfile)
basDraw.DrawImageTextFile(debugfile,debugfile,[[10,10]],[clss+':'+str(round(score,3))],color='#000000',font=basDraw.GetCurFont())
到InferMMDet.py里面
请使用如下配置：
convDct=dict(
        type='ConvModuleEx',
        act_cfg=None
    )
dct=dict(SLEGAR=dict(oneof=classes))
model = dict(
    type='RetinaNetWithTail',
    classDct=dct,
    classes=classes,
    pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        .....
'''
convDct=dict(
        type='ConvModule',
        act_cfg=dict(type='ReLU', inplace=True)
    )
class TailModule(nn.Module):
    def __init__(self,thr,loss_cls,clsChannel,classDct,classes
            ,hiddenChannel=[12,16],boxIOU=0.6,backboneMode=True,
            conv_cfg=convDct,spplevel=4,bottleneckcfg=None
            ,bnChannel=[2048,1024,512,64]):
        super().__init__()
        self.thr=thr
        self.net=nn.Sequential()
        self.classDct=classDct
        self.classes=classes
        self.buildClassDex()
        self.boxIOU=boxIOU
        self.backboneMode=backboneMode
        outchannel=len(classDct)+1
        if backboneMode:
            self.spp=SPPModule(spplevel,in_channels=bnChannel[-1],out_channels=bnChannel[-1],conv_cfg=conv_cfg)
            self.bottleneck=nn.Sequential()
            lastLayer=bnChannel[0]
            for i in range(1,len(bnChannel)):
                stride=1
                pad=0
                if(i==1):
                    stride=2
                    pad=0
                self.bottleneck.add_module('tailconv'+str(i),nn.Conv2d(lastLayer,bnChannel[i],1,stride,pad))
                self.bottleneck.add_module('bottleneck'+str(i),Bottleneck(bnChannel[i],
                int(bnChannel[i]/4),stride=1,conv_cfg=bottleneckcfg))
                lastLayer=bnChannel[i]            
            lastLayer=self.spp.GetFeatureNum()*lastLayer
            self.fulinear=nn.Linear(lastLayer,clsChannel)
            lastLayer=clsChannel+clsChannel
        else:
            lastLayer=clsChannel
        for i in range(len(hiddenChannel)):
            self.net.add_module('tail'+str(i),nn.Linear(lastLayer,hiddenChannel[i]))
            self.net.add_module('relu'+str(i),nn.ReLU())
            lastLayer=hiddenChannel[i]
        self.net.add_module('tail',nn.Linear(lastLayer,outchannel))
        self.loss=MODELS.build(loss_cls)        
        return
    def buildClassDex(self):
        classDct=dict(self.classDct)
        keys=list(classDct.keys())
        self.gtdct={}
        for kk,vv in classDct.items():
            key=keys.index(kk)
            vv=dict(vv)
            dct={}
            for k,v in vv.items():
                lst=[]
                for m in v:
                   lst.append(self.classes.index(m))
                dct[k]=lst
            self.gtdct[key]=dct
    def GetCurClass(self,gtlabel):
        lst=[]
        res=gtlabel.detach().cpu().numpy().tolist()
        curKey=len(self.classDct)
        for kk,vv in self.gtdct.items():
            if('include' in vv):
                bFlag=True
                for v in vv['include']:
                    if v not in res:bFlag=False
                if(bFlag):
                    return kk
                continue
            if('oneof' in vv):
                for v in vv['oneof']:
                    if v in res:
                        return kk
                continue
            if('exclude' in vv):
                bFlag=False
                for v in vv['exclude']:
                    if v in res:
                        bFlag=True
                        break
                if(not bFlag):
                    return kk
                continue
        return curKey
    def assignGt(self,gtlabels):
        res=[]
        for i in range(len(gtlabels)):
            res.append(self.GetCurClass(gtlabels[i]))
        res=gtlabels[0].new_tensor(res)
        return res
    def init_weights(self):
        for module in self.net._modules.values():
            normal_init(module)
        return
    def forward_single(self,result,curDex,x=None):
        coo=None
        det_bboxes, det_labels=result
        if(det_bboxes.shape[0]!=0):
            detx=det_labels.new_tensor([i for i in range(det_labels.shape[0])])
            dety=det_labels
            detIndices=torch.cat([dety.unsqueeze(0),detx.unsqueeze(0)],0)
            coo=torch.sparse_coo_tensor(detIndices,det_bboxes[...,4],[len(self.classes),det_labels.shape[0]]).to_dense()
            flag=coo==0
            coo[flag]=-1
            coo=coo.permute([1,0])
            coo,_=torch.max(coo,0)
        else:
            coo=x[0].new_tensor([-1 for i in self.classes])
        #coo.requires_grad=False
        return coo
    def forward(self,x,results):
        coos=[]
        for i,r in enumerate(results):
            coos.append(self.forward_single(r,i,x).unsqueeze(0))
        coos=torch.cat(coos,0)
        if(self.backboneMode):
            feat=self.bottleneck(x[-1])
            feat=self.spp(feat)
            res=self.fulinear(feat)
            coos=torch.cat([coos,res],1)
        res=self.net(coos)
        return res
    def forward_train(self,x, results,gt_labels):
        res=self.forward(x,results)
        label=self.assignGt(gt_labels)
        ratio=1.0
        losses=self.loss(res,label)*ratio
        return losses
#outclass是新类别的数量，不用考虑除新类别的其他类别
#言外之意新类别如果是渣土车类别，不用考虑非渣土车类别，下面会自动构造非渣土车类别通道

@MODELS.register_module()
class RetinaNetWithTail(RetinaNet):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""
    def __init__(self,
                 classDct,
                 classes,
                 thr=0.5,
                 bboxIou=0.6,
                 hiddenchannel=[12,16],
                 backboneMode=True,
                 conv_cfg=convDct,spplevel=4,bottleneckcfg=None
                ,bnChannel=[2048,1024,512,64],
                 loss_cls=dict(type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0,
                     reduction='mean'),
                 *args,
                 **kwargs):       
        super(RetinaNetWithTail, self).__init__(*args,**kwargs)
        self.tailModule=TailModule(thr,loss_cls,self.bbox_head.num_classes
            ,classDct,classes,hiddenchannel,bboxIou,backboneMode,conv_cfg,
            spplevel,bottleneckcfg,bnChannel)        
        self.finalcls=list(classDct)+['NoneClass']
        return
    def init_weights(self):
        super().init_weights()
        self.tailModule.init_weights()
        return 
    def GetResults(self,results):
        bbox_list=[]
        for res in results:
            det_label,det_scores,det_bboxes=res.labels,res.scores,res.bboxes
            det_scores=det_scores[...,None]
            det_bboxes=torch.cat([det_bboxes,det_scores],-1)
            bbox_list.append([det_bboxes,det_label])
        return bbox_list
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        b = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(b)
        else:
            x=b
        losses={}
        closses = losses = self.bbox_head.loss(x, batch_data_samples)
        losses.update(closses)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=False)
        bbox_list=self.GetResults(results_list)
        batch_gt_instances,_,_=unpack_gt_instances(batch_data_samples)
        gt_labels=[]
        for inst in batch_gt_instances:
            gt_labels.append(inst.labels)
        losses['tailloss']=self.tailModule.forward_train(b,bbox_list,gt_labels)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        b = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(b)
        results_list = self.bbox_head.predict(x,batch_data_samples,rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        bbox_list =self.GetResults(results_list)
        res=self.tailModule(b,bbox_list)
        res=torch.nn.Softmax(-1)(res)
        for i in range(len(batch_data_samples)):
            batch_data_samples[i].tailcls=res[i]
        return batch_data_samples
    