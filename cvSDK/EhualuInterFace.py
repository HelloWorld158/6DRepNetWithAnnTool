import os,sys
import BasicUseFunc as basFunc
import DataIO as dio
import shutil
import BasicAlgorismGeo as basAlgo
import re
import xmlSample as xls
import cv2
import xml.etree.ElementTree as ET
import BasicDrawing as basDraw
import base64
def AnalizeXml(xmlfile,bflag):
    dxl=xls.get_xml_dom(xmlfile)
    objs=xls.get_elementsByTagName(dxl,'object')
    res=[]
    illegal=[]
    for obj in objs:
        name=xls.get_node_value(obj,'name')
        bndbx=xls.get_elementsByTagName(dxl,'bndbox')[0]
        left=xls.get_node_value(bndbx,'xmin')
        top=xls.get_node_value(bndbx,'ymin')
        right=xls.get_node_value(bndbx,'xmax')
        bottom=xls.get_node_value(bndbx,'ymax')
        if(name.lower()!='illegalcar'):
            illegal=['illegalcar',[left,top],[right,top],[right,bottom],[left,bottom]]
        else:
            res.append(basAlgo.LtRbToxywh([left,top,right,bottom]))
        #res.append(basAlgo.LtRbToxywh([left,top,right,bottom]))
    return res,illegal
def ConvertOWNJson(labelFile,classes=None,oriFile=None,debugFile=None):
    jsonfile=labelFile
    data=dio.getjsondata(jsonfile)
    w = data['imageWidth']
    h = data['imageHeight']
    box=[]
    namelst=[]
    clsidlst=[]
    for obj in data['shapes']:
        cls = obj['label']
        if(classes):
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            clsidlst.append(cls_id)
        b = (float(obj['points'][0][0]), float(obj['points'][0][1]), float(obj['points'][1][0]),
                    float(obj['points'][1][1]))
        c=list(b)
        b=(min(c[0],c[2]),min(c[1],c[3]),max(c[0],c[2]),max(c[1],c[3]))
        box.append(list(b))
        namelst.append(cls)
    newDebugImg=debugFile
    if(not newDebugImg or not oriFile):
        if(not classes):
            return namelst,box,w,h
        return namelst,clsidlst,box,w,h
    basDraw.DrawImageCopyFileRectangles(newDebugImg,oriFile,box,namelst=namelst)
    if(not classes):
        return namelst,box,w,h
    return namelst,clsidlst,box,w,h
def ConvertPolyOwnJson(labelFile):
    jsonfile=labelFile
    data=dio.getjsondata(jsonfile)
    w = data['imageWidth']
    h = data['imageHeight']
    boxes=[]
    namelst=[]
    for obj in data['shapes']:
        cls = obj['label']
        box=[]
        for i in range(len(obj['points'])):
            pt=obj['points'][i]
            box.append([pt[0],pt[1]])
        boxes.append(box)
        namelst.append(cls)
    return namelst,boxes
def WriteOwnJson(labelFile,imageHeight,imageWidth,boxes,clsNames):
    dct={}
    dct['imageWidth']=imageWidth
    dct['imageHeight']=imageHeight
    dct['shapes']=[]
    for i in range(len(boxes)):
        b=boxes[i]
        c=clsNames[i]
        cdct={}
        cdct['label']=c
        cdct['points']=[[int(b[0]),int(b[1])],[int(b[2]),int(b[3])]]
        dct['shapes'].append(cdct)
    dio.writejsondictFormatFile(dct,labelFile)
    return
def ConvertOWNXML(labelFile,classes=None,oriFile=None,debugFile=None):
    xmlfile=labelFile
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    box=[]
    namelst=[]
    clsidlst=[]
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if(classes):
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            clsidlst.append(cls_id)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),float(xmlbox.find('xmax').text), 
            float(xmlbox.find('ymax').text))
        c=b
        b=(min(c[0],c[2]),min(c[1],c[3]),max(c[0],c[2]),max(c[1],c[3]))
        box.append(list(b))
        namelst.append(cls)
    newDebugImg=debugFile
    if(not newDebugImg or not oriFile):
        if(not classes):
            return namelst,box,w,h
        return namelst,clsidlst,box,w,h
    basDraw.DrawImageCopyFileRectangles(newDebugImg,oriFile,box,namelst=namelst)
    if(not classes):
        return namelst,box,w,h
    return namelst,clsidlst,box,w,h
def ConvertXmlTojson(xmldir,picdir):
    imagelst=dio.getdatas(xmldir)
    xmlst=dio.getdatas(xmldir,'*.xml')
    tpname=basFunc.DeletePathLastSplit(xmldir)
    name=os.path.basename(tpname)
    dir=os.path.dirname(tpname)
    jsonfile=os.path.join(dir,name+'.json')
    for image in imagelst:
        dir,name,ext=basFunc.GetfileDirNamefilter(image)
        arrNum=[int(s) for s in re.findall(r'\d+',name)]
        newpath=os.path.join(picdir,str(arrNum[-1])+ext)
        shutil.copy(image,newpath)
    count=0
    bflag=True
    illigeal=None
    for xml in xmlst:
        _,name,_=basFunc.GetfileDirNamefilter(xml)
        arrNum=[int(s) for s in re.findall(r'\d+',name)]
        frameId=arrNum[-1]
        bArr,illigeal=AnalizeXml(xml,bflag)
        dct={'frameId':frameId,'structure':bArr}
        if(count==0):
            dio.writejsondictFormatFile(dct,jsonfile)
        else:
            bflag=False
            dio.writejsondictFormatFile(dct,jsonfile,'a')
        count+=1
def EhlExtractVideo(videoPath,iSep):
    cap = cv2.VideoCapture(videoPath)
    framenum = int(cap.get(7)) 
    numFrame = -1
    nCount=0
    videoDir,videoName,_=basFunc.GetfileDirNamefilter(videoPath)
    videoDir=os.path.join(videoDir,videoName)
    dio.MakeEmptyDir(videoDir)
    while True:
        if(cap.grab()):
            flag, frame = cap.retrieve()
            if (not flag):
                break
            else:
                numFrame += 1
                print("Process:",numFrame,'/',framenum,end='\r',flush=True)
                if(numFrame%iSep!=0): continue
                nCount+=1
                svPath=videoDir  
                newPath = os.path.join(svPath,videoName+'_'+str(nCount)+'.jpg')
                cv2.imencode('.jpg', frame)[1].tofile(newPath)
        else:
            break;
        if (cv2.waitKey(10) == 27):
            break
def ReadLabelClassFromJson(jsPath):
    d,name,ftr=basFunc.GetfileDirNamefilter(bmpPath)
    jsPath=os.path.join(d,name+'.json')
    if(not os.path.exists(jsPath)):return None
    data=dio.getjsondata(jsPath)
    return data['labelData']['type'][0]
def ConvertOwnSegJson(labelfile,classes=None):
    jsdata=dio.getjsondata(labelfile)
    ptsArr=jsdata['shapes']
    dxlst,ptlst,groupids=[],[],[]
    for i in range(len(ptsArr)):
        if classes is None or ptsArr[i]['label'] in classes:
            dxlst.append(ptsArr[i]['label'])
            ptlst.append(ptsArr[i]['points'])
            groupids.append(ptsArr[i]['group_id'])
    gtflag=[False for i in range(len(dxlst))]
    ndxlst,nptlst=[],[]
    for i in range(len(dxlst)):
        if groupids[i] is None:
            nptlst.append([ptlst[i]])
            ndxlst.append(dxlst[i])
            continue
        if gtflag[i]:continue
        segs=[ptlst[i]]
        for j in range(i+1,len(dxlst)):
            if groupids[j] is None:continue
            if gtflag[j]:continue
            if groupids[i]==groupids[j] and dxlst[i]==dxlst[j]:
                segs.append(ptlst[j])
                gtflag[j]=True
        nptlst.append(segs)
        ndxlst.append(dxlst[i])
    w=jsdata['imageWidth']
    h=jsdata['imageHeight']
    if classes is None:
        return nptlst,ndxlst,w,h
    nIndex=[classes.index(ndxlst[i]) for i in range(len(ndxlst))]
    return nptlst,ndxlst,nIndex,w,h
def WriteOwnSegJson(nptlst,ndxlst,jsfile,imgfile=None,w=-1,h=-1):
    if w<0 or h<0:
        img=dio.GetOriginImageData(imgfile)
        h=img.shape[0]
        w=img.shape[1]
    jsdata={'imageWidth':w,'imageHeight':h,"version": "5.0.1",
    "flags": {}}
    ptsArr=[]
    for i in range(len(ndxlst)):
        if len(nptlst[i])==1:
            groupid=None
        else:
            groupid=i+1        
        for j in range(len(nptlst[i])):
            data={'group_id':groupid,'label':ndxlst[i],'flags':{},"shape_type": "polygon"}
            data['points']=nptlst[i][j]
            ptsArr.append(data)
    jsdata['shapes']=ptsArr
    if imgfile is not None:
        jsdata=AddLabelmeTailImageData(imgfile,jsdata)
    dio.writejsondictFormatFile(jsdata,jsfile)
def AddLabelmeTailImageData(imgfile,dct):
    _,name,ftr=basFunc.GetfileDirNamefilter(imgfile)
    dct['imagePath']=name+ftr
    fp=open(imgfile,mode='rb')
    buffer=fp.read()
    fp.close()
    dct['imageData']=base64.b64encode(buffer).decode('utf-8')
    return dct
def WriteOwnLabelmeJson(imgfile,labelFile,boxes,clsNames):
    dct={}
    img=dio.GetOriginImageData(imgfile)
    dct['imageWidth']=img.shape[1]
    dct['imageHeight']=img.shape[0]
    dct['shapes']=[]
    dct['version']='5.0.1'
    for i in range(len(boxes)):
        b=boxes[i]
        c=clsNames[i]
        cdct={}
        cdct['label']=c
        cdct['shape_type']='rectangle'
        cdct['group_id']=None
        cdct['flags']={}
        cdct['points']=[[int(b[0]),int(b[1])],[int(b[2]),int(b[3])]]
        dct['shapes'].append(cdct)
    dct=AddLabelmeTailImageData(imgfile,dct)
    dio.writejsondictFormatFile(dct,labelFile)
    return