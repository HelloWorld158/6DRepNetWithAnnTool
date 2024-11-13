from albumentations.core.composition import BboxParams
import mmcv
import numpy as np
from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms.transforms import Resize
from mmcv.image.geometric import rescale_size
from mmcv.image.geometric import imresize,_scale_size
try:
    import albumentations as albu
    from albumentations import Compose
except ImportError:
    albu = None
    Compose = None
'''
dict(type='RandomChoiceResize',
    scales=imageScale,
    resize_type = 'ResizeEx',
    baseReduc=None,
    allscales=imageScale,
    keep_ratio=True),
注意不要配合RandomSize
'''
@TRANSFORMS.register_module()
class ResizeEx(Resize):
    def __init__(self,baseReduc,allscales,**kwargs):
        super(ResizeEx,self).__init__(**kwargs)
        self.baseReduc=baseReduc
        self.allscales=allscales
        if self.baseReduc is None and self.allscales is None:
            raise('baseReduc and allscales must not be None at same time')
    def GetCmpSize(self,size):
        s=max(size[0],size[1])
        for i,m in enumerate(self.allscales):
            if s in m:
                return i
        '''
        print('Error')
        print('s:',s)
        print('scales:',self.img_scale)
        exit(0)
        '''
    def imresize(self,img,
             size,
             return_scale=False,
             interpolation='bilinear',
             out=None,
             backend=None):
        if self.baseReduc is not None:
            newsize=[(size[0]+self.baseReduc-1)//self.baseReduc*self.baseReduc,
                (size[1]+self.baseReduc-1)//self.baseReduc*self.baseReduc]    
        else:
            newsize=self.allscales[self.GetCmpSize(size)]
            newsize=list(newsize)
        img=mmcv.image.imresize(img,size,False,interpolation,out,backend)
        nimg=np.zeros([newsize[1],newsize[0],3],dtype=np.uint8)
        nimg[:size[1],:size[0]]=img
        return nimg,size
    def imrescale(self,img,
              scale,
              return_scale=False,
              interpolation='bilinear',
              backend=None):
        """Resize image while keeping the aspect ratio.

        Args:
            img (ndarray): The input image.
            scale (float | tuple[int]): The scaling factor or maximum size.
                If it is a float number, then the image will be rescaled by this
                factor, else if it is a tuple of 2 integers, then the image will
                be rescaled as large as possible within the scale.
            return_scale (bool): Whether to return the scaling factor besides the
                rescaled image.
            interpolation (str): Same as :func:`resize`.
            backend (str | None): Same as :func:`resize`.

        Returns:
            ndarray: The rescaled image.
        """
        h, w = img.shape[:2]
        new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
        rescaled_img,size = self.imresize(
            img, new_size, interpolation=interpolation, backend=backend)
        return rescaled_img,size
    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""

        if results.get('img', None) is not None:
            if self.keep_ratio:
                img,size = self.imrescale(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results['img'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img,size = self.imresize(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            new_h, new_w = size[1],size[0]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['scale_factor'] = (w_scale, h_scale)
            results['keep_ratio'] = self.keep_ratio
    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        if results.get('gt_seg_map', None) is not None:
            if self.keep_ratio:
                gt_seg,_ = self.imrescale(
                    results['gt_seg_map'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results['gt_seg_map'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_seg_map'] = gt_seg

@TRANSFORMS.register_module()
class ResizeAlongAxis(Resize):
    def rescale_size(self,old_size, scale, return_scale=False):
        """Calculate the new size to be rescaled to.

        Args:
            old_size (tuple[int]): The old size (w, h) of image.
            scale (float | tuple[int]): The scaling factor or maximum size.
                If it is a float number, then the image will be rescaled by this
                factor, else if it is a tuple of 2 integers, then the image will
                be rescaled as large as possible within the scale.
            return_scale (bool): Whether to return the scaling factor besides the
                rescaled image size.

        Returns:
            tuple[int]: The new rescaled image size.
        """
        w, h = old_size
        if isinstance(scale, (float, int)):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
            scale=(scale,scale)
        if isinstance(scale, tuple):
            nw,nh=scale
            nratio=nh/nw
            oratio=h/w
            if oratio>=nratio:
                sc=h
                di=nh
            else:
                sc=w
                di=nw
            ratio=di/sc
            scale_factor=(ratio,ratio)
        else:
            raise TypeError(
                f'Scale must be a number or tuple of int, but got {type(scale)}')

        new_size = _scale_size((w, h), scale_factor)

        if return_scale:
            return new_size, scale_factor
        else:
            return new_size
    def imrescale(self,img,
              scale,
              return_scale=False,
              interpolation='bilinear',
              backend=None):
        h, w = img.shape[:2]
        new_size, scale_factor = self.rescale_size((w, h), scale, return_scale=True)
        rescaled_img = imresize(
            img, new_size, interpolation=interpolation, backend=backend)
        if return_scale:
            return rescaled_img, scale_factor
        else:
            return rescaled_img
    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if results.get('img', None) is not None:
            if self.keep_ratio:
                img, scale_factor = self.imrescale(
                    results['img'], results['scale'], 
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results['img'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results['img'], results['scale'], return_scale=True)
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['scale_factor'] = (w_scale, h_scale)
            results['keep_ratio'] = self.keep_ratio 
    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        if results.get('gt_seg_map', None) is not None:
            if self.keep_ratio:
                gt_seg = self.imrescale(
                    results['gt_seg_map'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results['gt_seg_map'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_seg_map'] = gt_seg
'''
这个PIPELINE将会模拟从大分辨率变为小分辨率物体，不能模拟从小分辨率物体变成大分辨率物体
(建议重新编写pipeline,或者加入其它神经网络进行小图像超分)
dict(type='ScaleImageRatio',outCnt=2000,validratio=-0.74)
将训练输出的recordScale填写到validratio即可
'''
'''
@TRANSFORMS.register_module()
class ScaleImageRatio(object):
    def __init__(self,ratio=0.05,validratio=0.2,minArea=4.0,outCnt=1000,usestd=False):
        super().__init__()
        assert(albu)
        assert(Compose)
        self.ratio=ratio
        self.minarea=minArea
        self.usestd=usestd
        self.trainCnt=0
        self.recordScale=0
        self.validratio=validratio
        self.outCnt=outCnt
        self.count=0
    def __NoGTCall__(self,results):
        img=results['img']
        dct={'image':img}
        aug=Compose([albu.ShiftScaleRotate(shift_limit=0,scale_limit=(self.validratio,self.validratio),rotate_limit=0,p=1,border_mode=cv2.BORDER_CONSTANT,value=0)])
        res=aug(**dct)
        results['img']=res['image']
        tempimg=np.array(res['image'],np.uint8)
        return results
    def __call__(self, results):
        if 'gt_bboxes' not in results:
            return self.__NoGTCall__(results)
        self.count+=1
        img=results['img']
        gtboxes=results['gt_bboxes']        
        if(len(gtboxes)==0):return results
        wh=np.array([img.shape[1],img.shape[0]],np.float32)[None,...]
        ratios=(gtboxes[...,[2,3]]-gtboxes[...,[0,1]])/wh
        tratio=ratios.mean()
        stdratio=ratios.std()/3/self.ratio
        scaleratio=tratio/self.ratio
        minratio=1.0/(scaleratio-stdratio)
        maxratio=1.0/(scaleratio+stdratio)
        dct={'image':img,'bboxes':gtboxes}
        dct['label']=results['gt_labels']
        if not self.usestd:
            minratio=1.0/scaleratio
            maxratio=1.0/scaleratio
        self.trainCnt+=1
        self.recordScale=(1.0/scaleratio+self.recordScale*(self.trainCnt-1))/self.trainCnt
        if self.count%self.outCnt==0:
            print('recordScale:',self.recordScale-1.0)
        maxratio+=-1.0
        minratio+=-1.0
        aug=Compose([albu.ShiftScaleRotate(shift_limit=0,scale_limit=(maxratio,minratio),rotate_limit=0,p=1,border_mode=cv2.BORDER_CONSTANT,value=0)],
                bbox_params=BboxParams('pascal_voc',label_fields=['label'],min_area=self.minarea))
        res=aug(**dct)
        results['img']=res['image']
        results['gt_bboxes']=np.array(res['bboxes'],np.float32)
        return results
'''