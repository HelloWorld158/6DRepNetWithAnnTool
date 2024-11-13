import BasicUseFunc as basFunc
import os,sys
import shutil
import BasicDrawing as basDraw
import BasicAlgorismGeo as basGeo
import DataIO as dio
import xml.etree.ElementTree as ET
import json
import random
import EhualuInterFace as ehl
import matplotlib.image as matImage
def GetModeDir(mode,convertDir):
    if(mode=='VOC'):
        convertDir=os.path.join(convertDir,'voc')
    if(mode=='COCO'):
        convertDir=os.path.join(convertDir,'COCO')
    if(mode=='OBJECT365'):
        convertDir=os.path.join(convertDir,'Objects365')
    return convertDir
def GetSubDir(mode):
    if(mode=='COCO'):
        return['2014','2015','2017']
def GetFilterClass(mode):
    global configData
    if(mode not in configData):
        print('mode is not in configData')
        exit()
    data=configData[mode]
    clsFilter=[]
    nClasses=[]
    trainRand=[]
    validRand=[]
    for i in range(len(data)):
        d=data[i]
        s=list(d.keys())
        clss=s[0]
        clsFilter.append(clss)
        d=d[clss]
        nClasses.append(d['name'])
        trainRand.append(d['trainrand'])
        validRand.append(d['validrand'])
    return clsFilter,nClasses,trainRand,validRand




#新类名称,name原始类名称，trainrand训练集抽样几率，validrand验证集抽样几率
configData={
    'VOC':[{'bottle':{'name':'bottle','trainrand':1.0,'validrand':0.01}},
            {'chair':{'name':'chair','trainrand':0.3,'validrand':0.005}},
            {'sofa':{'name':'sofa','trainrand':1.0,'validrand':0.01}}],
    'OBJECT365':[{'bottle':{'name':'bottle','trainrand':1.0,'validrand':0.01}},
            {'chair':{'name':'chair','trainrand':0.3,'validrand':0.005}},
            {'sofa':{'name':'sofa','trainrand':1.0,'validrand':0.01}},
            {'barrel/bucket':{'name':'barrel','trainrand':1.0,'validrand':0.01}}],
    'COCO':[{'bicycle':{'name':'bicycle','trainrand':1.0,'validrand':1.0}}],
    'OWNXMLDATA':[{'carton':{'name':'carton','trainrand':1.0,'validrand':1.0}},
            {'desk_chair':{'name':'chair','trainrand':1.0,'validrand':1.0}}],
    'OWNJSONDATA':[{'carton':{'name':'carton','trainrand':1.0,'validrand':1.0}},
            {'desk_chair':{'name':'chair','trainrand':1.0,'validrand':1.0}}]
}
mode='COCO'#'Initial','VOC','OBJECT365','OWNXMLDATA','OWNJSONDATA','COCO'
clsFilter,nClasses,trainRand,validRand=GetFilterClass(mode)
fClasses=nClasses#['bottle','chair','sofa','barrel','carton']
convertDir='/notebooks/TrainData_COCOVOCOBJ'
convertDir=GetModeDir(mode,convertDir)
batch=64
subdivisions=64
anchorMasks=3
ResetMode=True
yoloname='yolov4'
debugTemp=False
boxMin=0.08
minMode=['VOC','OBJECT365','COCO']
bConvertOthData=False
bSepDataAug=False
SepRand=0.5
maskExtract=True










def SaveOtherData(file,masks,rescls):
    global maskExtract
    if(not maskExtract):return
    d,name,ftr=basFunc.GetfileDirNamefilter(file)
    jsfile=os.path.join(d,name+'.json')
    if(os.path.exists(jsfile)):
        dct=dio.getjsondata(jsfile)
    else:
        dct={}
    dct['masks']=masks
    dct['classes']=rescls
    dio.writejsondictFormatFile(dct,jsfile)
def ConvertOtherData(file,boxes,names,w,h,data=None):
    if(len(boxes)==0):return None
    d,name,ftr=basFunc.GetfileDirNamefilter(file)
    name=mode+'_'+name
    svfile=os.path.join(data,name+'.json')
    svimg=os.path.join(data,name+ftr)
    shutil.copy(file,svimg)
    ehl.WriteOwnJson(svfile,h,w,boxes,names)
    return svimg
def SeperateJPGFile(namelst,boxes,orimg,txtDir,fp,debugflag=True):
    global bSepDataAug,boxMin
    if(not bSepDataAug):return
    img=dio.GetOriginImageData(orimg)
    debugDir=os.path.join(os.getcwd(),'debugDir')
    minsize=300
    maxsize=600
    bMinBox=False
    if(mode in minMode):bMinBox=True
    global dct,configFile
    global fClasses
    global bConvertOthData,SepRand
    for i in range(len(boxes)):
        if(random.uniform(0,1)>=SepRand):continue
        mimg,bxs,nms=basGeo.RandomResizeArea(boxes,namelst,img,i,minsize,minsize,maxsize,maxsize)
        dct['count']+=1
        dio.writejsondictFormatFile(dct,configFile)
        newimg=os.path.join(txtDir,str(dct['count'])+'.jpg')  
        debugimg=os.path.join(debugDir,str(dct['count'])+'.jpg')      
        txtfile=os.path.join(txtDir,str(dct['count'])+'.txt')
        out_file=open(txtfile,'w')
        matImage.imsave(newimg,mimg)
        w=mimg.shape[1]
        h=mimg.shape[0]
        resbox=[]
        rescls=[]
        for j in range(len(bxs)):
            b=bxs[j]
            b=list(b)
            c=b
            b=basGeo.LtRbToxywh(b)
            for k in range(2):
                b[2*k]/=float(w)
                b[2*k+1]/=float(h)
            if(bMinBox and min(b[2],b[3])<boxMin):
                continue
            resbox.append(c)
            rescls.append(namelst[j])
            if(not bConvertOthData):
                out_file.write(str(fClasses.index(namelst[j])) + " " + " ".join([str(a) for a in b]) + '\n')
        out_file.close()
        if(bConvertOthData):
            os.remove(txtfile)
            ConvertOtherData(newimg,resbox,rescls,w,h,txtDir)
        fp.write(newimg+'\n')
        if(not debugflag):
            if(bConvertOthData):os.remove(newimg)
            continue
        basDraw.DrawImageCopyFileRectangles(debugimg,newimg,resbox,namelst=rescls)
        if(bConvertOthData):os.remove(newimg)
    return
def FindFirstAvalDex(lines,matchstr):
    txtdex,dex=basFunc.Findtxtlinelst(lines,matchstr)
    while(txtdex!=0):
        txtdex,dex=basFunc.Findtxtlinelst(lines,matchstr)
    return txtdex,dex
def ChangeYoloData(lines,tdex,ldex,classesCount):
    txtdex,dex=basFunc.Findtxtlinelst(lines,'filters',endPos=ldex,reverse=True)
    lines[dex]='filters='+str((classesCount+5)*anchorMasks)+'\n'
    #print(lines[dex])
    txtdex,dex=basFunc.Findtxtlinelst(lines,'classes',startPos=ldex)
    lines[dex]='classes='+str(classesCount)+'\n'
    return txtdex,dex
def ChangeYoloBox(lines,classesCount):
    txtdex,dex=basFunc.Findtxtlinelst(lines,'yolo')
    txtdex,dex=ChangeYoloData(lines,txtdex,dex,classesCount)
    while(txtdex>=0 or dex>=0):
        txtdex,dex=basFunc.Findtxtlinelst(lines,'yolo',dex)
        if(txtdex<0 or dex<0):
            break
        txtdex,dex=ChangeYoloData(lines,txtdex,dex,classesCount)
    return
def AnalyzeGenCfg(filepath,svPath,classesCount):
    #if(not AutoAnalyzeCfg):return
    fp=open(filepath,'r')
    lines=fp.readlines()
    fp.close()
    _,dex=FindFirstAvalDex(lines,'batch')
    lines[dex]='batch='+str(batch)+'\n'
    _,dex=FindFirstAvalDex(lines,'subdivisions')
    lines[dex]='subdivisions='+str(subdivisions)+'\n'
    ChangeYoloBox(lines,classesCount)
    #print(lines[960])
    fp=open(svPath,'w')
    #basFunc.AddTailStrlists(lines)
    fp.writelines(lines)
    fp.close()
    return
def GenerateCoDataFromNames(cocoNames,classesCount,cocofiles,trainDir,validDir):
    filepath=cocofiles
    curDir=os.getcwd()
    fp=open(filepath,'w')
    dio.WriteLine(fp,'classes= '+str(classesCount))
    dio.WriteLine(fp,'train  = '+trainDir)
    dio.WriteLine(fp,'valid  = '+validDir)
    dio.WriteLine(fp,'names  = '+cocoNames)
    backup=os.path.join(curDir,'backup')
    basFunc.MakeEmptyDir(backup)
    dio.WriteLine(fp,'backup  = '+backup)
    dio.WriteLine(fp,'eval = coco')
    fp.close()
def WriteCocoNames(coco):
    fp=open(coco,'w')
    global nClasses
    for i in range(len(nClasses)):
        if(i!=len(nClasses)-1):
            fp.write(nClasses[i]+'\n')
        else:
            fp.write(nClasses[i])
    fp.close()
#[extDir]=GenerateEmtyDir(['extDir'])
def GenerateEmtyDir(dirs,curDir=os.getcwd()):
    retdirs=[]
    for dir in dirs:
        ndir=os.path.join(curDir,dir)
        retdirs.append(ndir)
        basFunc.MakeEmptyDir(ndir)
    return retdirs
def GetCurDirNames(dirfiles,curDir=os.getcwd()):
    retNames=[]
    for df in dirfiles:
        ndf=os.path.join(curDir,df)
        retNames.append(ndf)
    return retNames
def FindDirectoryName(dir,name):
    dirlst=[]
    dirlst=basFunc.get_dirlist(dir,dirlst)
    for d in dirlst:
        bn=os.path.basename(d)
        if(bn==name):
            return d
    return None
def KeepFile(names,trainflag=True,classes=None):
    global trainRand,validRand,clsFilter
    if(not classes):classes=clsFilter
    maxRate=-1
    for name in names:
        dex=classes.index(name)
        if(trainflag):
            maxRate=max(trainRand[dex],maxRate)
        else:
            maxRate=max(validRand[dex],maxRate)
    return maxRate>random.uniform(0,1)
def CheckTrainFlag(dir):
    cdir=basFunc.DeletePathLastSplit(dir)
    name=os.path.basename(cdir)
    if(name=='train'):
        return True
    return False
def CheckRandomXML(root,trainflag):
    global clsFilter
    names=[]
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in clsFilter:continue
        names.append(cls)
    return KeepFile(names,trainflag)
def ConvertXML(dir,imgDir,trainDir,validDir,classes,txtfile,valfile,randRate):
    filelst=basFunc.getdatas(dir,'*.xml')
    count=0
    debugDir=os.path.join(os.getcwd(),'debugDir')
    basFunc.MakeExistRetDir(debugDir)
    global dct,bConvertOthData
    global configFile,trainRand,validRand
    global nClasses
    global fClasses
    global mode
    global minMode
    global boxMin
    bMinBox=False
    if(mode in minMode):bMinBox=True
    dctData={}
    for file in filelst:      
        dct['count']+=1
        dio.writejsondictFormatFile(dct,configFile)
        trainflag=True
        if(random.uniform(0,1)<randRate):
            trainvalDir=trainDir
            fp=open(txtfile,'a')
        else:
            trainvalDir=validDir 
            fp=open(valfile,'a')
            trainflag=False        
        tree = ET.parse(file)
        root = tree.getroot()
        name=root.find('filename')
        bKeep=CheckRandomXML(root,trainflag)
        if(not bKeep):continue
        orimg=os.path.join(imgDir,name.text)
        newimg=os.path.join(trainvalDir,str(dct['count'])+'.jpg')
        label=os.path.join(trainvalDir,str(dct['count'])+'.txt')
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        out_file=open(label,'w')
        box=[]
        namelst=[]
        count+=1
        basFunc.Process(count,len(filelst))
        rescls=[]
        resbox=[]
        for obj in root.iter('object'):
            #difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes:continue
            cls_id = classes.index(cls)
            cls_id=fClasses.index(nClasses[cls_id])   
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text),  float(xmlbox.find('ymin').text),float(xmlbox.find('xmax').text),
                    float(xmlbox.find('ymax').text))
            c=list(b)
            b=(min(c[0],c[2]),min(c[1],c[3]),max(c[0],c[2]),max(c[1],c[3]))
            box.append(b)
            namelst.append(fClasses[cls_id])
            if w > 0 and h > 0:
                if(fClasses[cls_id] not in dctData):
                    dctData[fClasses[cls_id]]=1
                else:
                    dctData[fClasses[cls_id]]+=1
                b=list(b)
                c=b
                b=basGeo.LtRbToxywh(b)                
                for i in range(2):
                    b[2*i]/=float(w)
                    b[2*i+1]/=float(h)                
                if(bMinBox and min(b[2],b[3])<boxMin):
                    continue
                rescls.append(fClasses[cls_id])
                resbox.append(c)        
                if(not bConvertOthData):
                    out_file.write(str(cls_id) + " " + " ".join([str(a) for a in b]) + '\n')        
        if(bConvertOthData):
            ConvertOtherData(orimg,resbox,rescls,w,h,trainvalDir)
        out_file.close()
        if(len(resbox)==0 or bConvertOthData):
            os.remove(label)
            continue
        shutil.copy(orimg,newimg) 
        fp.write(newimg+'\n')
        fp.close()        
        _,_,ftr=basFunc.GetfileDirNamefilter(newimg)
        newDebugImg=os.path.join(debugDir,str(dct['count'])+ftr)
        #if(not trainflag):continue
        basDraw.DrawImageCopyFileRectangles(newDebugImg,orimg,box,namelst=namelst)  
    strTrainval='train'
    if(not trainflag): strTrainval='valid'
    dct[mode+'_'+strTrainval+'_'+str(dct['count'])]=dctData
    dio.writejsondictFormatFile(dct,configFile)
    return
def CarryVOCFiles():
    tars=[]
    tars=basFunc.get_filelist(convertDir,tars,'.tar')
    global tempDir
    global trainDir
    global validDir
    global dct
    global configFile
    global trainTxt
    global validTxt
    global clsFilter
    count=0
    for tar in tars:
        if(not debugTemp):
            basFunc.MakeEmptyDir(tempDir)
        dir,name,ftr=basFunc.GetfileDirNamefilter(tar)
        dex=name.find('trainval_')
        if(dex<0):continue
        newtar=os.path.join(tempDir,name+ftr)
        shutil.copy(tar,newtar)
        if(not debugTemp):
            os.system('tar -xvf '+newtar+' -C '+tempDir)
        anndir=FindDirectoryName(tempDir,'Annotations')
        jpgdir=FindDirectoryName(tempDir,'JPEGImages')
        print('curTar:',tar)
        ConvertXML(anndir,jpgdir,trainDir,validDir,clsFilter,trainTxt,validTxt,0.99)
        count+=1        
    return
def FindPicture(data,id):
    for i in range(len(data['images'])):
        if(data['images'][i]['id']==id):
            return i
    return -1
def FindClassIDS(data,classes):
    clslst=[]
    clsdct={}
    for vals in data['categories']:
        if(vals['name'] in classes):
            clslst.append(vals['id'])
            clsdct[vals['id']]=vals['name']
    return clslst,clsdct
def ConvertPictureDict(data):
    dct={}
    for i in range(len(data['images'])):
        dct[data['images'][i]['id']]=i
    return dct
def ConvertJson(imgDir,jsfile,trainvalDir,txtfile,classes,debugflag=True):
    jsfp=open(jsfile,'r')
    data=json.load(jsfp)
    count=0
    debugDir=os.path.join(os.getcwd(),'debugDir')
    basFunc.MakeExistRetDir(debugDir)
    global dct,bConvertOthData
    global configFile
    global nClasses
    global fClasses
    global minMode
    global mode
    global boxMin
    bMinBox=False
    if(mode in minMode):bMinBox=True
    clslst,clsdct=FindClassIDS(data,classes)
    imgdexs=ConvertPictureDict(data)
    cnt=0
    minRes=-1
    maxRes=-1
    for i in range(len(data['annotations'])):
        ann=data['annotations'][i]
        basFunc.Process(i,len(data['annotations']),'maxminProcess:')
        if(ann['category_id'] not in clslst):continue
        res=imgdexs[ann['image_id']]
        if(minRes<0 or res<minRes):
            minRes=res
        if(maxRes<0 or maxRes<res):
            maxRes=res
        #if(i>26000):break
    if(minRes<0 or maxRes<0):
        print('error:',minRes,'/',maxRes)
        exit()
    nDict=[{} for i in range(maxRes-minRes+1)]
    dctData={}
    for i in range(len(data['annotations'])):
        ann=data['annotations'][i]
        basFunc.Process(i,len(data['annotations']),'annProcess')
        if(ann['category_id'] not in clslst):continue
        res=imgdexs[ann['image_id']]
        if(res<0): continue
        imgName =data['images'][res]['file_name']
        orimg=os.path.join(imgDir,imgName)
        classids=classes.index(clsdct[ann['category_id']])
        clss=nClasses[classids]
        if(clss not in dctData):
            dctData[clss]=1
        else:
            dctData[clss]+=1
        box=ann['bbox']
        mask=ann['segmentation']
        rbox=[box[0],box[1],box[0]+box[2],box[1]+box[3]]   
        h=data['images'][res]['height']
        w=data['images'][res]['width']
        b=basGeo.LtRbToxywh(rbox)
        for j in range(2):
            b[2*j]/=float(w)
            b[2*j+1]/=float(h)
        if(b[0]<=0 or b[1]<=0):continue
        if(b[0]>=1.0 or b[1]>=1.0):continue
        if(b[2]<=0 or b[3]<=0):continue
        if(bMinBox and min(b[2],b[3])<boxMin):continue
        flag=True
        for j in range(4):
            if(b[j]<0.0):
                flag=False
                break
        if(not flag):continue
        if('img' not in nDict[res-minRes]):
            nDict[res-minRes]={'img':orimg,'masks':[mask],'class':[clss],'box':[b],'rbox':[rbox],'height':h,'width':w,'imgName':imgName}
        else:
            nDict[res-minRes]['box'].append(b)
            nDict[res-minRes]['rbox'].append(rbox)
            nDict[res-minRes]['class'].append(clss)
            nDict[res-minRes]['masks'].append(mask)
        cnt+=1        
        #if(i>26000):break
    fp=open(txtfile,'a')
    count=0
    for value in nDict:
        count+=1
        basFunc.Process(count,len(nDict),'dictProcess:')
        if('img' not in value):continue
        orimg=value['img']
        if(not os.path.exists(orimg)):continue        
        dct['count']+=1        
        dio.writejsondictFormatFile(dct,configFile)
        newimg=os.path.join(trainvalDir,str(dct['count'])+'.jpg')
        newDebugImg=os.path.join(debugDir,str(dct['count'])+'.jpg')
        label=os.path.join(trainvalDir,str(dct['count'])+'.txt')        
        clas=value['class']
        trainflag=CheckTrainFlag(trainvalDir)
        bKeep=KeepFile(clas,trainflag,nClasses)
        if(not bKeep):continue
        box=value['box']
        fp.write(newimg+'\n')
        shutil.copy(orimg,newimg)
        cp=open(label,'w')
        namelst=[]
        resbox=[]
        rescls=[]
        for i in range(len(box)):
            b=box[i]
            clss=clas[i]
            namelst.append(clss)
            resbox.append(value['rbox'][i])
            rescls.append(clss)
            if(not bConvertOthData):
                cp.write(str(fClasses.index(clss)) + " " + " ".join([str(a) for a in b]) + '\n')
        if(bConvertOthData):
            ConvertOtherData(orimg,resbox,rescls,value['width'],value['height'],trainvalDir)
            SaveOtherData(orimg,value['masks'],rescls)
        cp.close()
        if(len(resbox)==0 or bConvertOthData):
            os.remove(label)
            os.remove(newimg)
            if(len(resbox)!=0):
                basDraw.DrawImageCopyFileRectangles(newDebugImg,orimg,value['rbox'],namelst=namelst)
            continue
        SaveOtherData(newimg,value['masks'],rescls)
        if(not debugflag): continue
        basDraw.DrawImageCopyFileRectangles(newDebugImg,orimg,value['rbox'],namelst=namelst)
    fp.close()
    strTrainval='train'
    trainflag=CheckTrainFlag(trainvalDir)
    if(not trainflag): strTrainval='valid'
    dct[mode+'_'+strTrainval+'_'+str(dct['count'])]=dctData
    dio.writejsondictFormatFile(dct,configFile)
    return
def CarryOBJ365Files():
    global tempDir
    global trainDir
    global validDir
    global dct
    global configFile
    global trainTxt
    global validTxt
    global clsFilter
    zips=[]
    zips=basFunc.get_filelist(convertDir,zips,'.zip')
    trainfile=None
    validfile=None
    cflag=False
    bflag=False
    for z in zips:
        _,name,ftr=basFunc.GetfileDirNamefilter(z)
        if(name.find('train')>=0):
            trainfile=z
            cflag=True
        if(name.find('val')>=0):
            validfile=z
            bflag=True
        if(cflag and bflag):
            break
    jsons=[]
    cflag=False
    bflag=False
    jsons=basFunc.get_filelist(convertDir,jsons,'.json')
    trainJson=None
    validJson=None
    for j in jsons:
        _,name,ftr=basFunc.GetfileDirNamefilter(j)
        if(name.find('train')>=0):
            trainJson=j
            cflag=True
        if(name.find('val')>=0):
            validJson=j
            bflag=True
        if(cflag and bflag):
            break
    print('valid Step')
    if(not debugTemp):
        basFunc.MakeEmptyDir(tempDir)
        os.system('unzip '+validfile+' -d '+tempDir)
    ConvertJson(os.path.join(tempDir,'val'),validJson,validDir,validTxt,clsFilter)
    print('train Step')
    if(not debugTemp):
        basFunc.MakeEmptyDir(tempDir)
        os.system('unzip '+trainfile+' -d '+tempDir)    
    ConvertJson(os.path.join(tempDir,'train'),trainJson,trainDir,trainTxt,clsFilter)
    return
def CheckRandomJson(data,trainflag):
    global clsFilter
    names=[]
    for obj in data['shapes']:
        cls = obj['label']
        if cls not in clsFilter:continue
        names.append(cls)
    return KeepFile(names,trainflag)
def ConvertOWNJson(dir,imgDir,labelDir,classes,txtfile,debugflag=True):
    filelst=basFunc.getdatas(imgDir)
    count=0
    debugDir=os.path.join(os.getcwd(),'debugDir')
    basFunc.MakeExistRetDir(debugDir)
    fp=open(txtfile,'a')
    global nClasses
    global fClasses
    global dct,bConvertOthData
    global boxMin
    global minMode
    global mode
    bMinBox=False
    if(mode in minMode):bMinBox=True
    dctData={}
    trainflag=CheckTrainFlag(dir)
    for file in filelst:    
        dct['count']+=1
        dio.writejsondictFormatFile(dct,configFile)    
        vdir,name,ftr=basFunc.GetfileDirNamefilter(file)
        jsonfile=os.path.join(labelDir,name+'.json')
        if(not os.path.exists(jsonfile)):
            continue
        newimg=os.path.join(dir,str(dct['count'])+ftr)       
        txtfile=os.path.join(dir,str(dct['count'])+'.txt')
        data=dio.getjsondata(jsonfile)
        bKeep=CheckRandomJson(data,trainflag)
        if(not bKeep):continue
        w = data['imageWidth']
        h = data['imageHeight']
        out_file=open(txtfile,'w')
        box=[]
        namelst=[]
        resbox=[]
        rescls=[]
        for obj in data['shapes']:
            cls = obj['label']
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            cls_id=fClasses.index(nClasses[cls_id])
            b = (float(obj['points'][0][0]), float(obj['points'][0][1]), float(obj['points'][1][0]),
                    float(obj['points'][1][1]))
            c=list(b)
            b=(min(c[0],c[2]),min(c[1],c[3]),max(c[0],c[2]),max(c[1],c[3]))
            box.append(list(b))
            namelst.append(fClasses[cls_id])
            if w > 0 and h > 0:
                if(fClasses[cls_id] not in dctData):
                    dctData[fClasses[cls_id]]=1
                else:
                    dctData[fClasses[cls_id]]+=1
                b=list(b)
                c=b
                b=basGeo.LtRbToxywh(b) 
                for i in range(2):
                    b[2*i]/=float(w)
                    b[2*i+1]/=float(h)
                if(bMinBox and min(b[2],b[3])<boxMin):continue
                if(b[0]<=0 or b[1]<=0):continue
                if(b[0]>=1.0 or b[1]>=1.0):continue
                if(b[2]<=0 or b[3]<=0):continue
                resbox.append(c)
                rescls.append(fClasses[cls_id])                  
                #for i in b:
                #    if(i<0 or i>=1 ):
                #        print('error box:',b)
                #        exit()
                if(not bConvertOthData):
                    out_file.write(str(cls_id) + " " + " ".join([str(a) for a in b]) + '\n')
        if(bConvertOthData):
            ConvertOtherData(file,resbox,rescls,w,h,dir)
        out_file.close()
        if(len(resbox)==0 or bConvertOthData):
            os.remove(txtfile)
            if(len(resbox)!=0):
                basDraw.DrawImageCopyFileRectangles(newDebugImg,file,box,namelst=namelst)
            continue
        shutil.copy(file,newimg)
        fp.write(newimg+'\n') 
        if(trainflag): SeperateJPGFile(rescls,resbox,file,dir,fp,debugflag)
        newDebugImg=os.path.join(debugDir,str(dct['count'])+ftr)        
        count+=1
        basFunc.Process(count,len(filelst))
        if(not debugflag):continue
        basDraw.DrawImageCopyFileRectangles(newDebugImg,file,box,namelst=namelst)
    fp.close()
    strTrainval='train'
    if(not trainflag): strTrainval='valid'
    dct[mode+'_'+strTrainval+'_'+str(dct['count'])]=dctData
    dio.writejsondictFormatFile(dct,configFile)
    return 
def ConvertOWNXML(dir,imgDir,labelDir,classes,txtfile,debugflag=True):
    filelst=basFunc.getdatas(imgDir)
    count=0
    debugDir=os.path.join(os.getcwd(),'debugDir')
    basFunc.MakeExistRetDir(debugDir)
    fp=open(txtfile,'a')
    global dct,bConvertOthData
    global configFile
    global boxMin
    global minMode
    global mode
    bMinBox=False
    if(mode in minMode):bMinBox=True
    dctData={}
    trainflag=CheckTrainFlag(dir)
    for file in filelst:     
        dct['count']+=1
        dio.writejsondictFormatFile(dct,configFile)   
        vdir,name,ftr=basFunc.GetfileDirNamefilter(file)
        xmlfile=os.path.join(labelDir,name+'.xml')
        if(not os.path.exists(xmlfile)):
            continue
        newimg=os.path.join(dir,str(dct['count'])+ftr)        
        txtfile=os.path.join(dir,str(dct['count'])+'.txt')
        tree = ET.parse(xmlfile)
        root = tree.getroot()
        bKeep=CheckRandomXML(root,trainflag)
        if(not bKeep):continue
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        out_file=open(txtfile,'w')
        box=[]
        namelst=[]
        resbox=[]
        rescls=[]
        global nClasses
        global fClasses
        for obj in root.iter('object'):
            #difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes :
                continue
            cls_id = classes.index(cls)
            cls_id = fClasses.index(nClasses[cls_id])
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),float(xmlbox.find('xmax').text), 
                    float(xmlbox.find('ymax').text))
            c=list(b)
            b=(min(c[0],c[2]),min(c[1],c[3]),max(c[0],c[2]),max(c[1],c[3]))
            box.append(b)
            namelst.append(fClasses[cls_id])
            if w > 0 and h > 0:
                if(fClasses[cls_id] not in dctData):
                    dctData[fClasses[cls_id]]=1
                else:
                    dctData[fClasses[cls_id]]+=1
                b=list(b)
                c=b
                b=basGeo.LtRbToxywh(b)   
                for i in range(2):
                    b[2*i]/=float(w)
                    b[2*i+1]/=float(h)
                if(bMinBox and min(b[2],b[3])<boxMin):continue
                resbox.append(c)
                rescls.append(fClasses[cls_id])                             
                if(not bConvertOthData):
                    out_file.write(str(cls_id) + " " + " ".join([str(a) for a in b]) + '\n')
        if(bConvertOthData):
            ConvertOtherData(file,resbox,rescls,w,h,dir)
        out_file.close()
        newDebugImg=os.path.join(debugDir,str(dct['count'])+ftr)
        count+=1
        basFunc.Process(count,len(filelst))
        if(len(resbox)==0 or bConvertOthData):
            os.remove(txtfile)
            if(len(resbox)!=0):
                basDraw.DrawImageCopyFileRectangles(newDebugImg,file,box,namelst=namelst)
                if(trainflag): SeperateJPGFile(rescls,resbox,file,dir,fp,debugflag)
            continue
        shutil.copy(file,newimg)
        fp.write(newimg+'\n')                
        if(trainflag): SeperateJPGFile(rescls,resbox,file,dir,fp,debugflag)
        if(not debugflag):continue
        basDraw.DrawImageCopyFileRectangles(newDebugImg,file,box,namelst=namelst)
    fp.close()
    strTrainval='train'
    if(not trainflag): strTrainval='valid'
    dct[mode+'_'+strTrainval+'_'+str(dct['count'])]=dctData
    dio.writejsondictFormatFile(dct,configFile)
    return    
def CarryOWNXMLFiles():
    global tempDir
    global trainDir
    global validDir
    global dct
    global configFile
    global trainTxt
    global validTxt
    global clsFilter
    global curDir
    imageDir=os.path.join(curDir,'trainimg')
    labelDir=os.path.join(curDir,'trainlabel')
    vimageDir=os.path.join(curDir,'validimg')
    vlabelDir=os.path.join(curDir,'validlabel')
    print('train step')
    ConvertOWNXML(trainDir,imageDir,labelDir,clsFilter,trainTxt)
    print('valid step')
    ConvertOWNXML(validDir,vimageDir,vlabelDir,clsFilter,validTxt)
    return
def CarryOWNJSONFiles():
    global tempDir
    global trainDir
    global validDir
    global dct
    global configFile
    global trainTxt
    global validTxt
    global clsFilter
    global curDir
    imageDir=os.path.join(curDir,'trainimg')
    labelDir=os.path.join(curDir,'trainlabel')
    vimageDir=os.path.join(curDir,'validimg')
    vlabelDir=os.path.join(curDir,'validlabel')
    print('train step')
    ConvertOWNJson(trainDir,imageDir,labelDir,clsFilter,trainTxt)
    print('valid step')
    ConvertOWNJson(validDir,vimageDir,vlabelDir,clsFilter,validTxt)
    return
def ResetAllProject():
    global trainDir
    global validDir
    global trainTxt
    global validTxt
    global dct
    global configFile
    debugDir=os.path.join(os.getcwd(),'debugDir')
    basFunc.MakeEmptyDir(trainDir)
    basFunc.MakeEmptyDir(validDir)
    basFunc.MakeEmptyDir(debugDir)
    fp=open(trainTxt,'w')
    fp.close()
    fp=open(validTxt,'w')
    fp.close()
    os.remove(configFile)
    dct={}
    dct['count']=0
    dio.writejsondictFormatFile(dct,configFile)
def ConvertCoCoJson(dir,imgDir,labelDir,classes,txtfile,debugflag=True):
    imgDir=basFunc.DeletePathLastSplit(imgDir)
    basName=os.path.basename(imgDir)
    [jsonfile]=basFunc.GetCurDirNames(['instances_'+basName+'.json'],labelDir)
    ConvertJson(imgDir,jsonfile,dir,txtfile,classes,debugflag)
def CarryCOCOFiles():
    global tempDir
    global trainDir
    global validDir
    global dct
    global configFile
    global trainTxt
    global validTxt
    global clsFilter
    global mode
    global convertDir
    zips=[]
    zips=basFunc.get_filelist(convertDir,zips,'.zip')
    trainfile=None
    validfile=None
    cflag=False
    bflag=False
    import re
    subdirs=GetSubDir(mode)
    for z in subdirs:
        if(not debugTemp):
            basFunc.MakeEmptyDir(tempDir)
        num=[int(s) for s in re.findall(r'\d+', z)]
        num=num[-1] 
        d=os.path.join(convertDir,z)
        [trainfile,validfile,annfile]=basFunc.GetCurDirNames(['train'+str(num)+'.zip','val'+str(num)+'.zip',
        'annotations_trainval'+str(num)+'.zip'],d)
        flagg=False
        if(os.path.exists(trainfile) and os.path.exists(validfile) and os.path.exists(annfile)):
            flagg=True
        if(not flagg):continue
        [ttrainDir,tvalidDir,tannDir]=basFunc.GetCurDirNames(['train'+str(num),'val'+str(num)
        ,'annotations'],tempDir)       
        print('Unzip train val '+str(num))
        if(not debugTemp):
            os.system('unzip '+trainfile+' -d '+tempDir)
            os.system('unzip '+validfile+' -d '+tempDir)
            os.system('unzip '+annfile+' -d '+tempDir)
        print('train Step')
        ConvertCoCoJson(trainDir,ttrainDir,tannDir,clsFilter,trainTxt)
        print('valid Step')
        ConvertCoCoJson(validDir,tvalidDir,tannDir,clsFilter,validTxt)
    return
def CarryFiles():
    global mode
    if(mode=='Initial'):
        ResetAllProject()
    if(mode=='VOC'):
        CarryVOCFiles()
    if(mode=='OBJECT365'):
        CarryOBJ365Files()
    if(mode=='OWNXMLDATA'):
        CarryOWNXMLFiles()
    if(mode=='OWNJSONDATA'):
        CarryOWNJSONFiles()
    if(mode=='COCO'):
        CarryCOCOFiles()

curDir=os.getcwd()
if(not debugTemp):
    [tempDir]\
    =GenerateEmtyDir\
    (['tempDir'])
else:
    [tempDir]=basFunc.GetCurDirNames(['tempDir'])
[trainDir,validDir,trainTxt,validTxt,weightDir,configFile]\
=GetCurDirNames\
(['train','valid','train.txt','test.txt','weight','config.json'])
basDir=basFunc.DeletePathLastSplit(curDir)
def GetDict():
    dct={'count':0}
    return dct
def FinalWord():
    curFile=os.path.join(curDir,'ConvertDataSet.py')
    fp=open(curFile,'r')
    txts=fp.readlines()
    linDex,curLineDex=basFunc.Findtxtlinelst(txts,'ResetMode=')
    txts[curLineDex]='ResetMode=False\n'
    fp=open(curFile,'w')
    fp.writelines(txts)
    fp.close()
    print('------------------------------------')
    print('------------------------------------')
    print('Disable This py File ResetMode:False')
    print('------------------------------------')
    print('------------------------------------')
    return
#FinalWord()
dct=dio.InitJsonConfig(GetDict,configFile)
if(ResetMode):
    ResetAllProject()
"""basDir=os.path.dirname(basDir)
oriWeightDir=os.path.join(basDir,'weight')
cocoNames=os.path.join(weightDir,'coco.names')
cocofiles=os.path.join(weightDir,'coco.data')
cfgfile=os.path.join(weightDir,yoloname+'.cfg')
oricfgfile=os.path.join(oriWeightDir,yoloname+'.cfg')
oriWeight=os.path.join(oriWeightDir,yoloname+'.weights')
weight=os.path.join(weightDir,yoloname+'.weights')
shutil.copy(oriWeight,weight)
AnalyzeGenCfg(oricfgfile,cfgfile,len(nClasses))
WriteCocoNames(cocoNames)
GenerateCoDataFromNames(cocoNames,len(nClasses),cocofiles,trainTxt,validTxt)
fp=open(os.path.join(curDir,'train.py'),'r')
lines=fp.readlines()
fp.close()
dex,sDex=basFunc.Findtxtlinelst(lines,'InstanTrain=')
lines[sDex]='InstanTrain=True\n'
fp=open(os.path.join(curDir,'train.py'),'w')
fp.writelines(lines)
fp.close()"""
CarryFiles()
FinalWord()
