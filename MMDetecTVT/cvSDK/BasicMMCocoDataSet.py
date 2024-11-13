import os
import BasicUseFunc as basFunc
import DataIO as dio
import EhualuInterFace as ehl
import numpy as np
from mmdet.registry import DATASETS
import matplotlib.image as matimg
import BasicDrawing as basDraw
import BasicSegmentFunc as basSeg
from mmdet.datasets.coco import CocoDataset
import cv2

@DATASETS.register_module()
class EhlXMLCocoDataSet(CocoDataset):
    def __init__(self,dir,pipeline,filter='*.jpg',
                 filterEmptyGT=False,palette=None,
                 classes=None,test_mode=False,debugSep=0,debugSaveCnt=20,debugDir=None
                 ,showStatic=False,refresh=False,debugrgb=True,**kwargs):
        #支持Ehualu的数据集标注,dir是文件夹位置存储图片文件及对应图片文件的标注文件，pipeline是给图片进行数据变换同mmdetection 
        #默认的datasets里面的pipeline,ann_file这个必须是None，不许更改，必须要带，img_prefix 同ann_file,filter 是图片的后缀名,filterEmptyGT 
        #这个参数如果是True数据集中将会过滤空标注的图片，这些没有标注文件的图片既不会加载也不会参与训练
        #classes 同默认mmdetection的datasets里面classes这个参数，test_mode同mmdetection的datasets里面test_mode参数,debugSep 隔多少次训练输出
        #到debugDir中的图片及其标注内容,debugSaveCnt debugDir里面每个进程最多有多少张图片，debugDir，保存中间训练输出图片的位置，注意debugDir
        #每次训练不会清空上一次训练产生的图片,showStatic 显示训练数据的统计信息包含参与训练的图片总数，以及每个类别拥有的标注框数量,refresh 第一
        #次训练时会在train valid文件夹里面产生cocofile.json,如果这个参数置位为True，每次训练会重新产生cocofile.json
        self.DataDir = dir
        self.filter=filter
        self.METAINFO={}
        self.METAINFO['classes']=classes
        if palette is not None:
            self.METAINFO['palette']=palette
        else:
            self.METAINFO['palette']=[tuple(basSeg.GetColor(i)) for i in range(len(classes))]
        self.refresh=refresh
        self.filterEmptyGT=filterEmptyGT
        self.ann_file=self.GenerateCocoJson(self.METAINFO)  
        super(EhlXMLCocoDataSet,self).__init__(ann_file=self.ann_file,data_prefix=dict(img='',img_path=''),
            metainfo=self.METAINFO,data_root=dir,pipeline=pipeline,test_mode=test_mode
            ,**kwargs)
        if(showStatic):self.ShowStatic()
        self.debugSep=debugSep
        if(debugSep>0 and debugDir is None):
            curDir=os.path.abspath(os.path.dirname(basFunc.GetCurrentFileDir(__file__)))
            debugDir=os.path.join(curDir,'debugDir')
        self.debugDir=debugDir        
        self.debugSaveCnt=debugSaveCnt
        self.debugCnt=0
        self.usrcnt=0
        self.debugrgb=debugrgb
        if(debugDir):
            basFunc.MakeExistRetDir(debugDir)
    def ShowStatic(self):
        print('==================Static Data===================')
        print('dataInfo Num:',len(self),self.DataDir)
        poscnt=0
        negcnt=0
        lst=[0 for i in range(len(self.metainfo['classes']))]
        for i in range(len(self)):
            ann=self.prepare_data(i)['data_samples'].gt_instances.labels
            for label in ann:
                lst[int(label)]+=1
            if len(ann)==0:
                negcnt+=1
            else:
                poscnt+=1
        print('negNum:',negcnt,'posNum:',poscnt)
        for i in range(len(self.metainfo['classes'])):
            print(self.metainfo['classes'][i]+':'+str(lst[i]))
        print('================================================')
    def GenerateCocoJson(self,metainfo):        
        jsfile=os.path.join(self.DataDir,'cocofile.json')
        if(os.path.exists(jsfile) and not self.refresh):
            return jsfile
        self.ConvertCocoDataSet(jsfile,metainfo)
        return jsfile        
    def ConvertOwnData(self,jsfile,classes):
        jsfile+='.xml'
        if not os.path.exists(jsfile):
            return None
        return ehl.ConvertOWNXML(jsfile,classes)
    def ConvertCocoDataSet(self,jsonfile,metainfo):
        bmpfiles=basFunc.getdatas(self.DataDir,self.filter)
        from BasicCocoFunc import GiveCocoAnnotation,GiveCocoImage,GetFinalCOCODct,GiveCategories
        count=0
        classes=[]
        clsanns=metainfo['classes']
        anns=[]
        licns=[]
        files=[]
        fileid=0
        boxid=0
        for file in bmpfiles:
            d,name,ftr=basFunc.GetfileDirNamefilter(file)
            jsfile=os.path.join(d,name)
            count+=1
            basFunc.Process(count,len(bmpfiles))
            tdata=self.ConvertOwnData(jsfile,clsanns)            
            if(tdata is not None):                      
                fileid+=1                
                segflag=False
                if len(tdata)==5:
                    namelst,clsidlst,boxes,w,h=tdata
                else:
                    assert(len(tdata)==6)
                    namelst,clsidlst,boxes,seges,w,h=tdata
                    segflag=True
                dctfile,dctlsn=GiveCocoImage(file,h,w,fileid)
                files.append(dctfile)
                licns.append(dctlsn)
                for i in range(len(namelst)):
                    c=namelst[i]
                    boxm=boxes[i]
                    box=[boxm[0],boxm[1],boxm[2]-boxm[0],boxm[3]-boxm[1]]
                    if not segflag:
                        segs=[]
                    else:
                        segs=seges[i]
                        segs=np.array(segs,np.float64).reshape([1,-1]).tolist()
                    boxid+=1
                    dctbox=GiveCocoAnnotation(boxid,box,c,fileid,segs)
                    if(c not in clsanns):continue
                    anns.append(dctbox)
            else:
                if(self.filterEmptyGT):continue
                img=dio.GetOriginImageData(file)
                h,w=img.shape[:2]
                fileid+=1
                dctfile,dctlsn=GiveCocoImage(file,h,w,fileid)
                files.append(dctfile)
                licns.append(dctlsn)
        print('\nannotation')
        for i in range(len(anns)):
            anns[i]['category_id']=clsanns.index(anns[i]['category_id'])
            basFunc.Process(i+1,len(anns))
        for i in range(len(clsanns)):
            classes.append(GiveCategories(i,clsanns[i]))
        GetFinalCOCODct(jsonfile,classes,anns,licns,files)
        return
    def BlendMasksToImageFile(self,file,outfile,masks,palettes=None):
        image=dio.GetOriginImageData(file)
        image=np.array(image,np.float32)
        for j in range(masks.shape[-1]):
            smaks=masks[...,j]==1
            if palettes is None:
                clr=basSeg.GetColor(j)
            else:
                clr=palettes[j]
            image[smaks]=(np.array(clr,np.float32)+image[smaks])/2
            image=np.where(image>255.0,255.0,image)
        image=image.astype(np.uint8)
        matimg.imsave(outfile,image)
    def _DebugMaskImage(self,data,newimg):
        if 'masks' in data['data_samples'].gt_instances:
            maskes=data['data_samples'].gt_instances.masks.masks.copy()
            maskes=maskes.transpose([1,2,0])
            self.BlendMasksToImageFile(newimg,newimg,maskes)
        '''
        if 'gt_sem_seg' in data:
            maskes=data['data_samples'].gt_sem_seg.copy()
            maskes=maskes.transpose([1,2,0])
            self.BlendMasksToImageFile(newimg,newimg,maskes)
        '''
        return
    def _DebugImage(self,data):
        if(self.debugDir is None):return
        self.debugCnt+=1
        if(self.debugCnt%self.debugSep!=0):
            return
        img=data['inputs'].data.detach().cpu().numpy()
        img=img.transpose([1,2,0])
        iMax=np.max(img)
        iMin=np.min(img)
        img=(img-iMin)/(iMax-iMin)
        bboxes=data['data_samples'].gt_instances.bboxes.tensor.detach().cpu().numpy().tolist()
        labeles=data['data_samples'].gt_instances.labels.detach().cpu().numpy().tolist()
        clstxts=[]
        for i in range(len(labeles)):
            clstxts.append(self.metainfo['classes'][labeles[i]])
        processId=os.getpid()
        self.usrcnt+=1
        self.usrcnt%=self.debugSaveCnt
        newimg=os.path.join(self.debugDir,str(processId)+'_'+str(self.usrcnt)+'.jpg')
        img=img*255.0
        img=img.astype(np.uint8)
        if self.debugrgb:
            mimg=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        else:
            mimg=img
        matimg.imsave(newimg,mimg)
        self._DebugMaskImage(data,newimg)
        basDraw.DrawImageCopyFileRectangles(newimg,newimg,bboxes,namelst=clstxts)        
        return
    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """
        data=super().__getitem__(idx)
        if self.test_mode:
            return data
        self._DebugImage(data)
        return data
@DATASETS.register_module()
class EhlJsonCocoDataSet(EhlXMLCocoDataSet):
    def ConvertOwnData(self, jsfile, classes):
        jsfile+='.json'
        if not os.path.exists(jsfile):
            return None
        return ehl.ConvertOWNJson(jsfile,classes)
#只能根据seg产生box
@DATASETS.register_module()
class EhlJsonSegDataSet(EhlJsonCocoDataSet):
    def __init__(self,**kwargs):
        #kwargs['seg_prefix']=''
        super().__init__(**kwargs)
    def ConvertOwnData(self, jsfile, classes):
        jsfile+='.json'
        if not os.path.exists(jsfile):
            return None
        ptlst,dxlst,indexs,w,h=ehl.ConvertOwnSegJson(jsfile,classes)      
        boxlst=[] 
        nptlst=[]
        for pts in ptlst:
            pts=[np.array(p,np.float32) for p in pts]
            pts=np.concatenate(pts,0)
            box=[float(pts[...,0].min()),float(pts[...,1].min()),float(pts[...,0].max()),float(pts[...,1].max())]
            boxlst.append(box)
            pts=pts.reshape([pts.shape[0],-1])
            nptlst.append(pts.tolist())        
        return  dxlst,indexs,boxlst,nptlst,w,h