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
from mmengine import DefaultScope
torch.backends.cudnn.deterministic = True
print('UseDetFile:',det.__file__)
usefile=config.Infer_UseFile
weightFile=config.Infer_WeightFile
weightFile=basCfg.GetDefaultBestPth(curDir,config,weightFile)
if(os.path.exists(weightFile)):
    pthfile=weightFile
else:
    raise FileNotFoundError('weightFile not found,please Check infer_weight or workDir/best.pth')
print('Use Final PthFile:',pthfile)
detfile=config.configpyfile
model=init_detector(detfile,pthfile,device='cuda:0')
defaultInstName=next(iter(reversed(DefaultScope._instance_dict)))
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
if __name__=='__main__':
    curDir=os.getcwd()
    [valRes,jsonDir]=basFunc.GenerateEmtyDir(['valRes','jsonDir'])
    [valTestData]=basFunc.GetCurDirNames(['valTestData'],curDir=os.path.dirname(curDir))
    files=basFunc.getdatas(valTestData)
    sumtime=0
    count=0
    visualizer=VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    for file in files:
        count+=1
        start=time.time()
        if(usefile):
            result=DetectImageFile(file)
            result=[result]
            img=cv2.imread(file)
        else:
            #img=dio.GetOriginImageData(file)
            img=cv2.imread(file)#注意img是bgr
            result=DetectImage(img)
        sumtime+=time.time()-start
        basFunc.Process(count,len(files))
        res,_,_=ConvertGeneralBox(result)
        c,b,s=basObj.GetGeneralBoxToCSB(res)
        _,name,ftr=basFunc.GetfileDirNamefilter(file)
        debugfile=os.path.join(valRes,name+ftr)
        jsfile=os.path.join(jsonDir,name+'.json')
        WriteLabelmeJson(c,b,img.shape[1],img.shape[0],file,jsfile)        
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
    
