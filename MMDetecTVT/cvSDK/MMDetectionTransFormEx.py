import os,sys
import BasicUseFunc as basFunc
import numpy as np
from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
import numpy as np
@TRANSFORMS.register_module()
class RectJustified(BaseTransform):
    def transform(self,results):
        if 'gt_bboxes' in results:
            results['gt_bboxes'].clip_(results['img_shape'])
        return results  
@TRANSFORMS.register_module()
class CheckAlbuFormat(BaseTransform):
    def transform(self,results):
        img = results['img']
        if img.dtype!=np.dtype('uint8'):
            print('albumentation only support uint8,please change LoadImageFromFile\'s to_float32=False')
            assert(img.dtype==np.dtype('uint8'))
        return results