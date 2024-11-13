from mmdet.registry import MODELS
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from mmcv.cnn.bricks import build_plugin_layer
from torch.nn.modules import conv
@MODELS.register_module(force=True)
class SPPModule(nn.Module):
    def __init__(self,num_levels,pooltype='maxpool',conv_cfg=None,in_channels=-1,out_channels=-1):
        #pooltype='avgpool'
        self.num_levels=num_levels
        self.conv_cfg=conv_cfg
        super(SPPModule,self).__init__()
        if conv_cfg is not None:
            self.convlst=nn.ModuleList([])
            for i in range(num_levels):
                self.convlst.append(build_plugin_layer(conv_cfg,in_channels=in_channels,
                 out_channels=out_channels,kernel_size=1)[1])
        if pooltype=='maxpool':
            self.pool=F.max_pool2d
            return
        if pooltype=='avgpool':
            self.pool=F.avg_pool2d
            return
        print('pool not support')
        return
    def GetFeatureNum(self):
        cnt=0
        for i in range(self.num_levels):
            cnt+=(i+1)**2
        return cnt
    def forward(self, x):
        num, c, h, w = x.size() # num:样本数量 c:通道数 h:高 w:宽
        for i in range(self.num_levels):
            level = i+1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))

            # 选择池化方式 
            tensor = self.pool(x, kernel_size=kernel_size, stride=stride, padding=pooling)            
            if self.conv_cfg is not None:
                tensor = self.convlst[i](tensor)
            tensor=tensor.view(num, -1)
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten
        
        
        