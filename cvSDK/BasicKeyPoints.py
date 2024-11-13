import os,sys
import BasicUseFunc as basFunc
import EhualuInterFace as ehl
import numpy as np
import cv2
import DataIO as dio
import BasicCocoFunc as bascoco
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.image as matimg
from typing import List
from copy import deepcopy
try:
    import mmpose
    upDir=os.path.dirname(os.path.dirname(mmpose.__file__))
    configDir=os.path.join(upDir,'configs/_base_/datasets')
    print('find mmpose config:',configDir)
    sys.path.append(configDir)
    mmposeflag=True
except:
    print('not found mmpose some function not use')
    mmposeflag=False
import importlib
'''
{
    'imagefile':'imagefile'
    0:{'keypoints':np.ndarray,'class':'person','bbox':np.ndarray,'box_score':float,'keypoints_scores':np.ndarray},
    1:{'keypoints':np.ndarray,'class':'person','bbox':np.ndarray,'box_score':float,'keypoints_scores':np.ndarray}
}
'''
def GetMMPoseDataSetInfo(pyfile:str):
    assert(mmposeflag)
    dataset=importlib.import_module(pyfile)
    return dataset.dataset_info
def ConvertMMCVFormat2labelme(imgfile:str,jsdata:dict,keyinfo,boxcls:str,scorethr=None)->dict:
    img=cv2.imread(imgfile)
    dct={}
    dct['imageWidth']=img.shape[1]
    dct['imageHeight']=img.shape[0]
    dct['shapes']=[]
    dct['version']='5.0.1'
    for i,data in enumerate(jsdata):
        b=data['bbox'][0]
        if b[0]==0 and b[1]==0 and b[2]==img.shape[1] and b[3]==img.shape[0]:
            continue
        c=boxcls
        cdct={}
        cdct['label']=c
        cdct['shape_type']='rectangle'
        cdct['group_id']=i
        cdct['flags']={}
        cdct['points']=[[int(b[0]),int(b[1])],[int(b[2]),int(b[3])]]
        dct['shapes'].append(cdct)
        for j in range(len(data['keypoints'])):
            cdct={}
            if scorethr is not None and data['keypoint_scores'][j]<scorethr:
                continue
            kpts=data['keypoints'][j]
            c=keyinfo[j]['name']
            cdct['label']=c
            cdct['shape_type']='point'
            cdct['group_id']=i
            cdct['description']=str(2)
            cdct['flags']={}
            cdct['points']=[[int(kpts[0]),int(kpts[1])]]
            dct['shapes'].append(cdct)
    dct=ehl.AddLabelmeTailImageData(imgfile,dct)
    return dct
def GetLabelmeRes(jsfile,keyinfo)->dict:
    jsdata=dio.getjsondata(jsfile)
    names=[]
    for i in range(len(keyinfo.keys())):
        vv=keyinfo[i]
        names.append(vv['name'])
    dct={'imagefile':jsdata['imagePath']}
    for s in jsdata['shapes']:
        if s['group_id'] not in dct:
            dct[s['group_id']]={'keypoints':np.full([len(names),2],-1,np.float32),
                                'keypoints_scores':np.full([len(names)],0,np.float32)}
        if s['shape_type']=='rectangle':
            dct[s['group_id']]['class']=s['label']
            dct[s['group_id']]['bbox']=np.array(s['points'],np.float32)
        else:
            if s['label'] not in names:
                continue
            index=names.index(s['label'])
            if 'description' in s.keys():
                dct[s['group_id']]['keypoints_scores'][index]=float(s['description'])
            else:
                dct[s['group_id']]['keypoints_scores'][index]=2.0
            dct[s['group_id']]['keypoints'][index]=np.array(s['points'][0],np.float32)
    return dct
def GetMMCVRes(jsdata:dict,keyinfo,imagefile:str,boxcls:str,w,h):
    names=[]
    for i in range(len(keyinfo.keys())):
        vv=keyinfo[i]
        names.append(vv['name'])
    dct={'imagefile':imagefile}
    for i,data in enumerate(jsdata):
        b=data['bbox'][0]
        if b[0]==0 and b[1]==0 and b[2]==w and b[3]==h:
            continue
        dct[i]={'keypoints':np.array(data['keypoints'],np.float32),
                'keypoints_scores':np.array(data['keypoint_scores'],np.float32)
                ,'box_score':data['bbox_score'],
                'class':boxcls,
                'bbox':np.array(data['bbox'],np.float32)}
    return dct
def ConvertOwnFormat2labelme(ownjsdata:dict,keyinfo,imgfile:str=None,scorethr=None):
    if imgfile is None:
        imgfile=ownjsdata.pop('imagefile')
    else:
        ownjsdata.pop('imagefile')
    img=cv2.imread(imgfile)
    dct={}
    dct['imageWidth']=img.shape[1]
    dct['imageHeight']=img.shape[0]
    dct['shapes']=[]
    dct['version']='5.0.1'    
    for kk,vv in ownjsdata.items():
        data=vv
        b=data['bbox'].flatten().tolist()
        c=vv['class']
        cdct={}
        cdct['label']=c
        cdct['shape_type']='rectangle'
        cdct['group_id']=kk
        cdct['flags']={}
        cdct['points']=[[int(b[0]),int(b[1])],[int(b[2]),int(b[3])]]
        dct['shapes'].append(cdct)
        for j in range(len(data['keypoints'])):
            cdct={}
            if scorethr is not None and data['keypoint_scores'][j]<scorethr:
                continue
            kpts=data['keypoints'][j].tolist()
            if(kpts[0]<0 or kpts[1]<0): continue
            c=keyinfo[j]['name']
            cdct['label']=c
            cdct['shape_type']='point'
            cdct['group_id']=kk
            cdct['flags']={}
            cdct['description']=str(2)
            cdct['points']=[[int(kpts[0]),int(kpts[1])]]
            dct['shapes'].append(cdct)
    dct=ehl.AddLabelmeTailImageData(imgfile,dct)
    return dct
class PicPoseType:
    def __init__(self):
        self.classes=[]
        self.boxes=[]
        self.w=0
        self.h=0
        self.file=''
        self.scores=[]
        self.poses=[]
    def fromlist(sf,classes,boxes,w,h,file,poses,scores=[]):
        sf.classes=classes
        sf.boxes=boxes
        sf.scores=scores
        sf.w=w
        sf.h=h
        sf.file=file
        sf.poses=poses
    def fromdict(sf,owndct,w,h):
        sf.w=w
        sf.h=h
        sf.file=owndct['imagefile']
        for kk,vv in owndct.items():
            try:
                id=int(kk)
            except:
                continue
            sf.classes.append(vv['class'])
            sf.boxes.append(vv['bbox'].flatten().tolist())
            if 'box_score' in vv.keys():
                sf.scores.append(vv['box_score'])
            pose=np.concatenate([vv['keypoints'],vv['keypoints_scores'][...,None]],-1)
            sf.poses.append(pose)
def GiveCategories(idex,name,dataset_info):
    dct={'id':idex,'name':name,'supercategory':name}
    keyinfo=dataset_info['keypoint_info']
    dct['keypoints']=[keyinfo[i]['name'] for i in range(len(keyinfo.keys()))]
    dct['skeleton']=[]
    skton=dataset_info['skeleton_info']
    for i in range(len(skton.keys())):
        link=skton[i]['link']
        dex0=dct['keypoints'].index(link[0])+1
        dex1=dct['keypoints'].index(link[1])+1
        dct['skeleton'].append([dex0,dex1])
    return dct
def GiveCocoAnnotation(idex,bbox,cateid,imageid,nposes,segs=[]):
    dct={'id':idex,'image_id':imageid,'segmentation':segs,'area':(bbox[2])*(bbox[3])
    ,'iscrowd':0,'bbox':bbox,'category_id':cateid}
    poses=deepcopy(nposes)
    posevis=poses[...,-1]
    msk=posevis==0
    poses[msk]=0
    dct['keypoints']=poses.flatten().astype(np.int32).tolist()    
    posevis=posevis.astype(np.int32)
    dct['num_keypoints']=int((posevis>0).sum())
    return dct
def GiveCocoAnnotationScore(idex,bbox,cateid,imageid,score,poses,segs=[]):
    dct={'id':idex,'image_id':imageid,'segmentation':segs,'area':(bbox[2])*(bbox[3])
    ,'iscrowd':0,'bbox':bbox,'category_id':cateid,'score':score}
    dct['keypoints']=poses.flatten().tolist()    
    return dct
def GenerateCOCOFile(cocofile,picDatas:List[PicPoseType],clses,dataset_info):
    count=0
    classes=[]
    clsanns=[]
    if(clses is not None):
        clsanns=clses
    anns=[]
    licns=[]
    files=[]
    fileid=0
    boxid=0
    for pic in  picDatas:
        fileid+=1
        dctfile,dctlsn=bascoco.GiveCocoImage(pic.file,pic.h,pic.w,fileid)
        files.append(dctfile)
        licns.append(dctlsn)
        for i in range(len(pic.classes)):
            c=pic.classes[i]
            boxm=pic.boxes[i]
            poses=pic.poses[i]
            box=[boxm[0],boxm[1],boxm[2]-boxm[0],boxm[3]-boxm[1]]
            boxid+=1
            if(len(pic.scores)!=0):
                dctbox=GiveCocoAnnotationScore(boxid,box,c,fileid,pic.scores[i],poses)
            else:
                dctbox=GiveCocoAnnotation(boxid,box,c,fileid,poses)
            if(clses is not None and c not in clsanns):continue
            if(clses is None and c not in clsanns):clsanns.append(c)
            anns.append(dctbox)
    for i in range(len(anns)):
        anns[i]['category_id']=clsanns.index(anns[i]['category_id'])+1
    for i in range(len(clsanns)):
        classes.append(GiveCategories(i+1,clsanns[i],dataset_info))
    bascoco.GetFinalCOCODct(cocofile,classes,anns,licns,files)
    return
class CocoEvalBoxPoseClsFile(bascoco.CocoEvalBoxClsFile):
    def addData(sf,data,dt:PicPoseType,gt:PicPoseType):
        mx=np.max(data)
        mn=np.min(data)
        img=(data-mn)/(mx-mn)
        sf.files.append(os.path.join(sf.tempDir,str(len(sf.files))+'.jpg'))
        dt.file=gt.file=os.path.join(sf.tempDir,str(len(sf.files))+'.jpg')
        matimg.imsave(sf.files[-1],img)
        sf.dtcoco.append(dt)
        sf.gtcoco.append(gt)
        return
    def addFile(sf,datafile,dt:PicPoseType,gt:PicPoseType):
        sf.files.append(datafile)
        sf.dtcoco.append(dt)
        sf.gtcoco.append(gt)
        return
    def eval(sf,methodtype,cocodt,cocogt,MaxSample):
        cocoEval=COCOeval(cocogt,cocodt,methodtype)
        if methodtype=='bbox':
            cocoEval.params.maxDets=MaxSample
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize() 
        sf.cocoEval=cocoEval
        box_coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
        key_coco_metric_names = {
                'keypoints_mAP': 0,
                'keypoints_mAP_50': 1,
                'keypoints_mAP_75': 2,
                'keypoints_mAP_m': 3,
                'keypoints_mAP_l': 4,
                'keypoints_AR@all': 5,
                'keypoints_AR@50': 6,
                'keypoints_AR@75': 7,
                'keypoints_AR_m@all': 8,
                'keypoints_AR_l@all': 9
            }
        if methodtype=='keypoints':
            coco_metric_names = key_coco_metric_names
        else:
            coco_metric_names = box_coco_metric_names
        ndct={}
        for kk,vv in coco_metric_names.items():
            ndct[vv]=kk
        dctkeys={vv:cocoEval.stats[kk] for kk,vv in ndct.items()}
        return dctkeys
    def Summarize(sf,datasetinfo,MaxSample=(100, 300, 1000),classes=None):
        sf.dtfile=os.path.join(sf.tempDir,'dtcocofile.json')
        sf.gtfile=os.path.join(sf.tempDir,'gtcocofile.json')
        GenerateCOCOFile(sf.dtfile,sf.dtcoco,classes,datasetinfo)
        GenerateCOCOFile(sf.gtfile,sf.gtcoco,classes,datasetinfo)
        cocodt=COCO(sf.dtfile)
        cocogt=COCO(sf.gtfile)
        dctbox=sf.eval('bbox',cocodt,cocogt,MaxSample)
        dctkeys=sf.eval('keypoints',cocodt,cocogt,MaxSample)
        return dctbox,dctkeys
def CocoEvalBoxPoseClsData(imgDatas,dtRes,gtlabel,classes,datasetinfo):
    ceval=CocoEvalBoxPoseClsFile()
    for i in range(len(imgDatas)):
        img=imgDatas[i]
        ceval.addData(imgDatas[i],dtRes[i],gtlabel[i])
    res=ceval.Summarize(datasetinfo,classes=classes)    
    return res
def CocoEvalBoxPoseClsFiles(imgfiles,dtRes,gtlabel,classes,datasetinfo):
    ceval=CocoEvalBoxPoseClsFile()
    for i in range(len(imgfiles)):
        img=dio.GetOriginImageData(imgfiles[i])
        ceval.addData(img,dtRes[i],gtlabel[i])
    res=ceval.Summarize(datasetinfo,classes=classes)    
    return res
'''
if __name__=='__main__':
    import shutil
    outDir=basFunc.MakeEmptyDir(args.output_dir)
    dataset=baskeypts.GetMMPoseDataSetInfo("coco")
    files=basFunc.GetFileDirLst(args.input_dir,filter='*.jpg').fileDirs
    files.extend(basFunc.GetFileDirLst(args.input_dir,filter='*.jpeg').fileDirs)
    i=0
    picdatas,gtdatas,appfile=[],[],[]
    for file,d,name,ftr in files:
        basFunc.Process(i,len(files))
        i+=1
        shutil.copy(file,outDir)
        jsfile=os.path.join(args.mmcv_output_dir,name+".json")
        jsdata=dio.getjsondata(jsfile)
        dct=baskeypts.ConvertMMCVFormat2labelme(file,jsdata,dataset['keypoint_info'],'person',0.45)
        dio.writejsondictFormatFile(dct,os.path.join(outDir,name+".json"))
        dctour=baskeypts.GetMMCVRes(jsdata,dataset['keypoint_info'],file,'person')
        gdct=baskeypts.GetLabelmeRes(os.path.join(outDir,name+".json"),dataset['keypoint_info'])
        #dct2=baskeypts.ConvertOwnFormat2labelme(ownres,dataset['keypoint_info'],file)
        #dio.writejsondictFormatFile(dct2,os.path.join(outDir,name+"_own.json"))
        img=dio.GetOriginImageData(file)
        posepic=baskeypts.PicPoseType()
        posepic.fromdict(dctour,img.shape[1],img.shape[0])
        gtpic=baskeypts.PicPoseType()
        gtpic.fromdict(gdct,img.shape[1],img.shape[0])
        picdatas.append(posepic)
        gtdatas.append(gtpic)
        appfile.append(file)
        if i>20:break
    baskeypts.CocoEvalBoxPoseClsFiles(appfile,picdatas,gtdatas,classes=['person'],datasetinfo=dataset)
'''