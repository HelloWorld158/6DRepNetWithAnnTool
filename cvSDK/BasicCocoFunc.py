import BasicUseFunc as basFunc
import os,sys
import DataIO as dio
#请安装whl,pycocotools控件谢谢。
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.image as matimg
import numpy as np
import BasicPicDeal as basPic
'''
注意这里面所有外部输入框必须是ltrb
'''
def GiveCocoImage(file,height,width,idex):
    dctImg={'coco_url':'','data_captured':'','file_name':file
    ,'flickr_url':'','height':height,'width':width,'license':5,'id':idex}
    dctLscs={'id':idex,'name':file,'url':''}
    return dctImg,dctLscs
def GiveCategories(idex,name):
    dct={'id':idex,'name':name,'supercategory':''}
    return dct
def GiveCocoAnnotation(idex,bbox,cateid,imageid,segs=[]):
    dct={'id':idex,'image_id':imageid,'segmentation':segs,'area':(bbox[2])*(bbox[3])
    ,'iscrowd':0,'bbox':bbox,'category_id':cateid}
    return dct
def GiveCocoAnnotationScore(idex,bbox,cateid,imageid,score,segs=[]):
    dct={'id':idex,'image_id':imageid,'segmentation':segs,'area':(bbox[2])*(bbox[3])
    ,'iscrowd':0,'bbox':bbox,'category_id':cateid,'score':score}
    return dct
def GetFinalCOCODct(file,classes,anns,licns,files):
    dct={'info':{'contributor':'chenzhuo','url':'chenzhuo','version':'1.0','year':2020}}
    dct['images']=files
    dct['licenses']=licns
    dct['annotations']=anns
    dct['categories']=classes
    dio.writejsondictFilelines(dct,file)
    return
def ConvertCrtJson(labelFile):
    jsonfile=labelFile
    data=dio.getjsondata(jsonfile)
    w = data['imageWidth']
    h = data['imageHeight']
    box=[]
    namelst=[]
    for obj in data['shapes']:
        cls = obj['label']
        b = (float(obj['points'][0][0]), float(obj['points'][0][1]), float(obj['points'][1][0]),
                    float(obj['points'][1][1]))
        c=list(b)
        b=(min(c[0],c[2]),min(c[1],c[3]),max(c[0],c[2]),max(c[1],c[3]))
        box.append(list(b))
        namelst.append(cls)
    return namelst,box,w,h
def ConvertCoCoDataSet(dir):
    jsonfile=os.path.join(dir,'cocofile.json')
    bmpfiles=basFunc.getdatas(dir)
    count=0
    classes=[]
    clsanns=[]
    anns=[]
    licns=[]
    files=[]
    fileid=0
    boxid=0
    for file in bmpfiles:
        d,name,ftr=basFunc.GetfileDirNamefilter(file)
        jsfile=os.path.join(d,name+'.json')
        count+=1
        basFunc.Process(count,len(bmpfiles))
        if(not os.path.exists(jsfile)):continue
        namelst,boxes,w,h=ConvertCrtJson(jsfile)        
        fileid+=1
        dctfile,dctlsn=GiveCocoImage(file,h,w,fileid)
        files.append(dctfile)
        licns.append(dctlsn)
        for i in range(len(namelst)):
            c=namelst[i]
            boxm=boxes[i]
            box=[boxm[0],boxm[1],boxm[2]-boxm[0],boxm[3]-boxm[1]]
            boxid+=1
            dctbox=GiveCocoAnnotation(boxid,box,c,fileid)
            if(c not in clsanns):
                clsanns.append(c)
            anns.append(dctbox)
    print('\nannotation')
    for i in range(len(anns)):
        anns[i]['category_id']=clsanns.index(anns[i]['category_id'])
        basFunc.Process(i+1,len(anns))
    for i in range(len(clsanns)):
        classes.append(GiveCategories(i,clsanns[i]))
    GetFinalCOCODct(jsonfile,classes,anns,licns,files)
class PicType(object):
    def __init__(sf,classes,boxes,w,h,file,scores=[]):
        sf.classes=classes
        sf.boxes=boxes
        sf.scores=scores
        sf.w=w
        sf.h=h
        sf.file=file
#这个函数仅支持ltrb
def GenerateCOCOFile(cocofile,picDatas,clses):
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
        dctfile,dctlsn=GiveCocoImage(pic.file,pic.h,pic.w,fileid)
        files.append(dctfile)
        licns.append(dctlsn)
        for i in range(len(pic.classes)):
            c=pic.classes[i]
            boxm=pic.boxes[i]
            box=[boxm[0],boxm[1],boxm[2]-boxm[0],boxm[3]-boxm[1]]
            boxid+=1
            if(len(pic.scores)!=0):
                dctbox=GiveCocoAnnotationScore(boxid,box,c,fileid,pic.scores[i])
            else:
                dctbox=GiveCocoAnnotation(boxid,box,c,fileid)
            if(clses is not None and c not in clsanns):continue
            if(clses is None and c not in clsanns):clsanns.append(c)
            anns.append(dctbox)
    for i in range(len(anns)):
        anns[i]['category_id']=clsanns.index(anns[i]['category_id'])
    for i in range(len(clsanns)):
        classes.append(GiveCategories(i,clsanns[i]))
    GetFinalCOCODct(cocofile,classes,anns,licns,files)
    return
class BoxClsData(object):
    def __init__(sf,classes,boxes,scores=[]):
        sf.classes=classes
        sf.boxes=boxes
        sf.scores=scores
#使用注意事项如果使用这个函数进行MAP评估一定要使用Summerize的classes类别,必须指定类别顺序,
#必须指定顺序
class CocoEvalBoxClsFile(object):
    def __init__(sf):
        sf.curDir=basFunc.GetCurrentFileDir(__file__)
        [sf.tempDir]=basFunc.GenerateEmtyDir(['tempDir'],sf.curDir)
        sf.files=[]
        sf.dtcoco=[]
        sf.gtcoco=[]
    def addData(sf,data,dt,gt):
        mx=np.max(data)
        mn=np.min(data)
        img=(data-mn)/(mx-mn)
        sf.files.append(os.path.join(sf.tempDir,str(len(sf.files))+'.jpg'))
        matimg.imsave(sf.files[-1],img)
        dtobj=PicType(dt.classes,dt.boxes,data.shape[1]
                ,data.shape[0],sf.files[-1],dt.scores)
        gtobj=PicType(gt.classes,gt.boxes,data.shape[1]
                ,data.shape[0],sf.files[-1])
        sf.dtcoco.append(dtobj)
        sf.gtcoco.append(gtobj)
        return
    def Summarize(sf,MaxSample=(100, 300, 1000),classes=None):
        sf.dtfile=os.path.join(sf.tempDir,'dtcocofile.json')
        sf.gtfile=os.path.join(sf.tempDir,'gtcocofile.json')
        GenerateCOCOFile(sf.dtfile,sf.dtcoco,classes)
        GenerateCOCOFile(sf.gtfile,sf.gtcoco,classes)
        cocodt=COCO(sf.dtfile)
        cocogt=COCO(sf.gtfile)
        cocoEval=COCOeval(cocogt,cocodt,'bbox')
        cocoEval.params.maxDets=MaxSample
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize() 
        sf.cocoEval=cocoEval
        coco_metric_names = {
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
        ndct={}
        for kk,vv in coco_metric_names.items():
            ndct[vv]=kk
        dct={vv:cocoEval.stats[kk] for kk,vv in ndct.items()}
        return dct
def CocoEvalBoxClsData(imgDatas,dtRes,gtlabel):
    ceval=CocoEvalBoxClsFile()
    for i in range(len(imgDatas)):
        img=imgDatas[i]
        ceval.addData(imgDatas[i],dtRes[i],gtlabel[i])
    res=ceval.Summarize()    
    return res
def CocoEvalBoxClsFiles(imgfiles,dtRes,gtlabel):
    ceval=CocoEvalBoxClsFile()
    for i in range(len(imgfiles)):
        img=dio.GetOriginImageData(imgfiles[i])
        ceval.addData(img,dtRes[i],gtlabel[i])
    res=ceval.Summarize()    
    return res

        
            

            

if __name__ == "__main__":
    cocoDir='/data/trainMMDet/debugMMDet/valid_coco'
    cocofile=os.path.join(cocoDir,'cocofile.json')
    jsdct=dio.getjsondata(cocofile)
    dct=jsdct.copy()
    dct['images']=[]
    dct['licenses']=[]
    for img in jsdct['images']:
        file=img['file_name']
        _,name,ftr=basFunc.GetfileDirNamefilter(file)
        newfile=os.path.join(cocoDir,name+ftr)
        img['file_name']=newfile
        dct['images'].append(img)
    for img in jsdct['licenses']:
        file=img['name']
        _,name,ftr=basFunc.GetfileDirNamefilter(file)
        newfile=os.path.join(cocoDir,name+ftr)
        img['name']=newfile
        dct['licenses'].append(img)
    dio.writejsondictFormatFile(dct,cocofile)
        

    
    
        
