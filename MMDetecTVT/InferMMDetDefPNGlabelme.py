import os,sys
curDir=os.path.abspath(os.path.dirname(__file__))
originDir=os.getcwd()
os.chdir(curDir)
sys.path.append(os.path.join(curDir,'cvSDK'))
sdkDir=os.path.join(curDir,'cvSDK')
import cvSDK.BasicUseFunc as basFunc
#---------------不要轻易更改GPU的位置，否则会出现报错的现象------------------------#
gpulst,_=basFunc.GetAvailableGPUsList()
print('FinalGPUse:',gpulst[0])
os.environ['CUDA_VISIBLE_DEVICES']=str(gpulst[0])
import cvSDK.DataIO as dio
import cvSDK.BasicConfig as basCfg
def GetDefaultDict():
    return basCfg.GetDefaultDict('train')
cfgdct=dio.InitJsonConfig(GetDefaultDict,os.path.join(curDir,'trainConfig.json'))
config=dio.Config(cfgdct)
basCfg.ChangeMMDetectionDir(config,cfgdct,os.path.join(curDir,'trainConfig.json'))
import mmdet as det
import mmdet.utils as detutils
detutils.register_all_modules(False)
import cvSDK.MMDetectionEx as mmdetEx
import cvSDK.BasicMMDet as basDet
import mmyolo as yolo
import mmyolo.utils as yoloutils
yoloutils.register_all_modules(False)
import cvSDK.BasicMMCocoYoloDataSet
import cv2
import torch
import base64
import time
import numpy as np
import cvSDK.BasicObjectDetection as basObj
from mmdet.registry import VISUALIZERS
import mmcv
from mmdet.apis.inference import init_detector,inference_detector
from mmengine import Config,DefaultScope
torch.backends.cudnn.deterministic = True
print('UseDetFile:',det.__file__)
usefile=config.Infer_UseFile
modelurl=config.modelurl
mmdetDir=os.path.dirname(os.path.abspath(det.__file__))
mmdetDir=os.path.dirname(mmdetDir)
#modelurl="https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201-9352eb0d.pth"
#编写这个detfile时注意留出第一行就可以了，剩下的这个程序会自动完成
detfile=config.configpyfile
detfile=os.path.join(curDir,detfile)
if os.path.exists(detfile): 
    print('================exist detfile config===================')
else:
    with open(detfile,'w'):
        pass
#detfile=os.path.join(os.getcwd(),'maskrcnndet.py')
#这个python文件会把所有结果保存到valRes文件夹中
#要注意detfile编写，detfile内容是能够替换掉原先设置的内容详见maskrcnndet.py/cascadercnndet.py
#detfile的内容类似配置文件，和普通的python代码要区分开，注意最好不要在这个detfile内使用python的相关语法
weightFile=config.Infer_WeightFile
def WriteLabelmeJson(namelst,box,w,h,file,jsfile):
    dct={}
    dct['imageWidth']=w
    dct['imageHeight']=h
    dct['shapes']=[]
    for i in range(len(box)):
        b=box[i]
        c=namelst[i]
        cdct={}
        cdct['label']=c
        cdct['points']=[[int(b[0]),int(b[1])],[int(b[2]),int(b[3])]]
        cdct['shape_type']='rectangle'
        cdct['group_id']=None
        cdct['flags']={}
        dct['shapes'].append(cdct)
    dct['version']="5.0.1"
    dct['flags']={}
    _,name,ftr=basFunc.GetfileDirNamefilter(file)
    dct['imagePath']=name+ftr
    fp=open(file,mode='rb')
    buffer=fp.read()
    fp.close()
    dct['imageData']=base64.b64encode(buffer).decode('utf-8')
    dio.writejsondictFormatFile(dct,jsfile)
changeflag=False
detDir=os.path.join(mmdetDir,'configs')
pthfile,pyfile,detDir=basDet.InitPthandConfig(modelurl,detDir,config.modelpyfile)
if(os.path.exists(weightFile)):
    pthfile=weightFile
with open(detfile,'r') as fp:
    txts=fp.readlines()
if len(txts)<10:
    basDet.ChangeDetectFile(pyfile,detfile)
    changeflag=True
print('Use Final PthFile:',pthfile,changeflag)
mmCfgEx=Config.fromfile(detfile)
if changeflag:mmCfgEx.dump(detfile)
model=init_detector(detfile,pthfile,device='cuda:0')
defaultInstName=next(iter(reversed(DefaultScope._instance_dict)))
sumtime=0
count=0
os.chdir(originDir)
def DetectImage(img):
    DefaultScope._instance_dict.move_to_end(defaultInstName,last=True)
    result=inference_detector(model,img)
    return result
def DetectImageFile(imgfile):
    DefaultScope._instance_dict.move_to_end(defaultInstName,last=True)
    result=inference_detector(model,imgfile)
    return result
def ConvertGeneralBox(result):
    res,classcore,boxes=[],[],[]
    cures=result.numpy()
    for i in range(cures.pred_instances.labels.shape[0]):
        label=cures.pred_instances.labels[i]
        box=cures.pred_instances.bboxes[i].tolist()
        score=cures.pred_instances.scores[i]
        classcore.append(model.dataset_meta['classes'][label]+':'+str(round(float(score),2)))
        boxes.append(box)
        res.append([model.dataset_meta['classes'][label],float(score),box])
    return res,classcore,boxes
def GetMasksLabels(result):
    res=result.numpy()
    labels=res.pred_instances.labels
    masks=res.pred_instances.masks
    return labels,masks
def GetModelClass():
    return model.dataset_meta['classes']
if __name__=='__main__':
    delcnt=0
    curDir=os.getcwd()
    [jsDir]=basFunc.GenerateEmtyDir(['jsonDir'],curDir)
    [valResPos,valResNeg]=basFunc.GenerateEmtyDir(['valResPos','valResNeg'])
    [valTestData]=basFunc.GetCurDirNames(['valTestData'],curDir=os.path.dirname(curDir))
    files=basFunc.getdatas(valTestData,"*.jpeg")
    visualizer=VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    print('-----------------------------------------------------')
    print(GetModelClass())
    print('-----------------------------------------------------')
    for file in files:
        count+=1
        start=time.time()
        d,name,ftr=basFunc.GetfileDirNamefilter(file)
        if(usefile):
            result=DetectImageFile(file)
            img=cv2.imread(file)
        else:
            #img=dio.GetOriginImageData(file)
            img=cv2.imread(file)
            result=DetectImage(img)
        sumtime+=time.time()-start
        basFunc.Process(count,len(files))
        res,_,_=ConvertGeneralBox(result)        
        valRes=valResNeg
        flag=False
        if(len(res)!=0):
            flag=True
            valRes=valResPos
        debugfile=os.path.join(valRes,name+ftr)
        if(flag):
            c,b,s=basObj.GetGeneralBoxToCSB(res)
            oname=name
            jsfile=os.path.join(jsDir,oname+'.json')
            WriteLabelmeJson(c, b, img.shape[1], img.shape[0], file, jsfile)
        if model.with_mask:
            pngfile=os.path.join(jsDir,oname+'.png')
            pklfile=os.path.join(jsDir,oname+'.pkl')
            pngdata=np.zeros([img.shape[0],img.shape[1]],np.uint8)
            labels,masks=GetMasksLabels(result)
            for i in range(labels.shape[0]):
                label=labels[i]
                msk=masks[i]
                pngdata[msk]=label+1
            #import matplotlib.image as matimg
            #matimg.imsave(pngfile,pngdata)
            cv2.imwrite(pngfile,pngdata)
            dio.SaveVariableToPKL(pklfile,result)
        imgcvt=mmcv.imconvert(img,'bgr','rgb')
        visualizer.add_datasample(
            'result',
            imgcvt,
            data_sample=result,
            draw_gt=False,
            show=False,
            wait_time=0,
            out_file=debugfile,
            pred_score_thr=0.0)
    print('One Frame Consume Time:',sumtime/len(files))
        
