import BasicUseFunc as basFunc
import numpy as np
import os
import sys
import glob
import random
import matplotlib.pyplot as matplot
import matplotlib.image as matimage
import inspect
import PIL as pil
import json
import cv2
import getopt as gopt
import argparse
import struct
import BasicPicDeal as basPic
import pickle as pk

#######################这俩个函数是俩个神级别函数可以保存python的任何变量###########################
def LoadVariablefromPKL(file):
    fp=open(file,'rb')
    v=pk.load(fp)
    fp.close()
    return v
def SaveVariableToPKL(file,v):
    fp=open(file,'wb')
    pk.dump(v,fp)
    fp.close()
#####################################################################################################
#对数组进行归化处理
def BatchNormalizePic(img,nOriMin,nOriMax,nMin,nMax):
    timg=np.reshape(img,[-1])
    tflag=timg>nOriMax
    timg[:][tflag]=nOriMax
    tflag=timg<nOriMin
    timg[:][tflag]=nOriMin
    timg/=(nOriMax-nOriMin)
    timg*=(nMax-nMin)
    timg+=nMin    
    return timg
#读取并归化图片，注意向内截取满足宽高的图片
def GetImageMatData(imagePath,tHeight=480,tWidth=640,nMin=0.0,nMax=255.0):
    nwHeight=tWidth
    nwWidth=tHeight
    timg=pil.Image.open(imagePath)
    timg=timg.convert("RGB")#L是灰度图
    scale=min(timg.size[0],timg.size[1])
    asize=np.asarray(timg.size,dtype=np.float32)
    asize/=np.float(scale)
    if((asize[1]/asize[0])>=(np.float(tHeight)/np.float(tWidth))):
        asize*=tWidth
    else:
        asize*=tHeight
    nsize=(int(asize[0]),int(asize[1]))
    aimg=timg.resize(nsize,pil.Image.ANTIALIAS)
    bimg=aimg.crop((0,0,nwHeight,nwWidth))
    bArray=np.asarray(bimg,dtype=np.float32)
    imgt=np.reshape(bArray,[tHeight,tWidth,3])
    img=BatchNormalizePic(imgt,0,255,nMin,nMax)   
    img=np.reshape(img,[tHeight,tWidth,3])
    return img
def GetOriginImageData(imgfile,convertNumpy=True):
    timg=pil.Image.open(imgfile)
    timg=timg.convert("RGB")
    if(convertNumpy):
        timg=np.array(timg,np.uint8)
    return timg
#mask插值函数，但是往内部截取
def GetImageData(imgfile,tHeight,tWidth):
    nwHeight=tWidth
    nwWidth=tHeight
    timg=pil.Image.open(imgfile)
    timg=timg.convert("RGB")
    scale=min(timg.size[0],timg.size[1])
    asize=np.asarray(timg.size,dtype=np.float32)
    asize/=np.float(scale)
    if((asize[1]/asize[0])>=(np.float(tHeight)/np.float(tWidth))):
        asize*=tWidth
    else:
        asize*=tHeight
    nsize=(int(asize[0]),int(asize[1]))
    aimg=timg.resize(nsize,pil.Image.NEAREST)
    bimg=aimg.crop((0,0,nwHeight,nwWidth))
    bArray=np.asarray(bimg,dtype=np.uint8)
    imgt=np.reshape(bArray,[tHeight,tWidth,3])
    return imgt
'''
优先使用
'''
def GetExpandImageDataEx(imagePath,tHeight=480,tWidth=640,nMin=0.0,nMax=255.0,dealflag=False,outscale=False):
    nwHeight=tWidth
    nwWidth=tHeight
    if type(imagePath) is str:
        timg=pil.Image.open(imagePath)        
    else:
        timg=pil.Image.fromarray(imagePath)   
    timg=timg.convert("RGB")
    asize=np.asarray(timg.size,dtype=np.float32)
    ratio=np.ones([2],np.float32)    
    if((asize[1]/asize[0])>=(np.float(tHeight)/np.float(tWidth))):
        scale=timg.size[1]
        divd=tHeight        
    else:
        scale=timg.size[0]
        divd=tWidth
    ratio/=np.float(scale)
    asize/=np.float(scale)
    asize*=divd
    ratio*=divd
    nsize=(int(asize[0]),int(asize[1]))
    aimg=timg.resize(nsize,pil.Image.NEAREST)
    aimg=np.asarray(aimg,np.uint8)
    bimg=np.zeros([tHeight,tWidth,3],np.uint8)
    bimg[:aimg.shape[0],:aimg.shape[1],:]=aimg
    if(dealflag):
        bimg=np.array(bimg,np.float32)
        img=BatchNormalizePic(bimg,0,255,nMin,nMax)   
        img=np.reshape(img,[tHeight,tWidth,3])
        if outscale:
            return img,ratio
        return img
    if outscale:
        return bimg,ratio
    return bimg

#真正使用扩充的image填充黑色最好使用BasicPicDeal中的GenerateExpandImageData函数
#原始图片导入
def GetExpandImageMatData(imagePath,tHeight=480,tWidth=640,nMin=0.0,nMax=255.0,dealflag=False):
    nwHeight=tWidth
    nwWidth=tHeight
    timg=pil.Image.open(imagePath)
    timg=timg.convert("RGB")
    scale=max(timg.size[0],timg.size[1])
    asize=np.asarray(timg.size,dtype=np.float32)
    asize/=np.float(scale)
    if((asize[1]/asize[0])>=(np.float(tHeight)/np.float(tWidth))):
        asize*=tHeight
    else:
        asize*=tWidth
    nsize=(int(asize[0]),int(asize[1]))
    aimg=timg.resize(nsize,pil.Image.ANTIALIAS)
    aimg=np.asarray(aimg,np.float32)
    bimg=np.zeros([tHeight,tWidth,3],np.uint8)
    bimg[:aimg.shape[0],:aimg.shape[1],:]=aimg
    bimg=np.zeros([tHeight,tWidth,3],np.uint8)
    bimg[:aimg.shape[0],:aimg.shape[1],:]=aimg
    if(dealflag):
        bimg=np.array(bimg,np.float32)
        img=BatchNormalizePic(bimg,0,255,nMin,nMax)   
        img=np.reshape(img,[tHeight,tWidth,3])
        return img
    return bimg
def GetExpandImageMatDataFromImg(img,tHeight=480,tWidth=640,nMin=0.0,nMax=255.0,dealflag=False):
    nwHeight=tWidth
    nwWidth=tHeight
    timg=pil.Image.fromarray(img)
    timg=timg.convert("RGB")
    scale=max(timg.size[0],timg.size[1])
    asize=np.asarray(timg.size,dtype=np.float32)
    asize/=np.float(scale)
    if((asize[1]/asize[0])>=(np.float(tHeight)/np.float(tWidth))):
        asize*=tHeight
    else:
        asize*=tWidth
    nsize=(int(asize[0]),int(asize[1]))
    aimg=timg.resize(nsize,pil.Image.ANTIALIAS)
    aimg=np.asarray(aimg,np.float32)
    bimg=np.zeros([tHeight,tWidth,3],np.uint8)
    bimg[:aimg.shape[0],:aimg.shape[1],:]=aimg
    bimg=np.zeros([tHeight,tWidth,3],np.uint8)
    bimg[:aimg.shape[0],:aimg.shape[1],:]=aimg
    if(dealflag):
        bimg=np.array(bimg,np.float32)
        img=BatchNormalizePic(bimg,0,255,nMin,nMax)   
        img=np.reshape(img,[tHeight,tWidth,3])
        return img
    return bimg
#真正使用扩充的image填充黑色最好使用BasicPicDeal中的GenerateExpandImageData函数或者GenerateExpandImageFile函数
#mask插值函数
def GetExpandImageData(imagePath,tHeight=480,tWidth=640,nMin=0.0,nMax=255.0,dealflag=False):
    nwHeight=tWidth
    nwWidth=tHeight
    timg=pil.Image.open(imagePath)
    timg=timg.convert("RGB")
    scale=max(timg.size[0],timg.size[1])
    asize=np.asarray(timg.size,dtype=np.float32)
    asize/=np.float(scale)
    if((asize[1]/asize[0])>=(np.float(tHeight)/np.float(tWidth))):
        asize*=tHeight
    else:
        asize*=tWidth
    nsize=(int(asize[0]),int(asize[1]))
    aimg=timg.resize(nsize,pil.Image.NEAREST)
    aimg=np.asarray(aimg,np.uint8)
    bimg=np.zeros([tHeight,tWidth,3],np.uint8)
    bimg[:aimg.shape[0],:aimg.shape[1],:]=aimg
    if(dealflag):
        bimg=np.array(bimg,np.float32)
        img=BatchNormalizePic(bimg,0,255,nMin,nMax)   
        img=np.reshape(img,[tHeight,tWidth,3])
        return img
    return bimg
def GenerateExpandImageFile(imgfile,tHeight=224,tWidth=224):
    img=GetOriginImageData(imgfile)
    img=basPic.GenerateExpandImageData(img,tHeight,tWidth)
    return img
#输出图片
def OutImage(imagePath,img,nwHeight,nwWidth):
    simg=img[0]
    m=np.asarray(simg,dtype=np.float32)
    #mimg=np.reshape(timg,[3,-1])
    #m=np.transpose(mimg)
    timg=m.reshape([nwHeight,nwWidth,3])
    qimg=BatchNormalizePic(timg,-1.0,1.0,0.0,1.0)
    aimg=np.reshape(qimg,[nwHeight,nwWidth,3])
    matimage.imsave(imagePath,aimg);
def SaveNormalizeImage(img,imgPath):
    img=np.array(img,np.float32)
    mx=np.max(img)
    mn=np.min(img)
    imgOut=(img-mn)/(mx-mn)
    matimage.imsave(imgPath,imgOut)
#随机打乱图片并输出batchsize个
def GetRandomChoice(imagelists,batchsize):    
    return np.random.choice(imagelists,batchsize,replace=True)
#读取jsondata，这个函数可以用来读入writejsondictFormat下生出来的json文件
def getjsondata(data_file):
    with open(data_file) as f:
        data = f.read().strip()
    try:
        data = json.loads(data)
    except Exception as e:
        print('error',str(e))
        return None
    return data
#按行读取json文件保存为list[dict[]]
#data=json.load(jsfp)
def getjsdatlstlindct(data_file):
    with open(data_file) as f:
        data = []
        for line in f.readlines():
            dic = json.loads(line)
            data.append(dic)
    return data
def writejsondictlines(dct,fp):
    json_str = json.dumps(dct)
    json_str+='\n'
    fp.write(json_str)
    fp.flush()
def writejsondictFilelines(dct,strfile,mode='w'):
    with open(strfile,mode) as fp:
        writejsondictlines(dct,fp)
        fp.flush()
def writejsondiclstFilelines(dctlst,strfile,mode='w'):
    fp=open(strfile,mode)
    for i in range(len(dctlst)):
        dct=dctlst[i]
        writejsondictlines(dct,fp)
    fp.close()
    return
def ReplaceJsonString(json_str,left='[',right=']',linenum=80):
    nwstr=''
    lcount=0
    scount=0
    bFlag=False
    for m in json_str:        
        if m==left :
            lcount+=1
            scount=0
        if m==right :
            lcount-=1
            scount=0
        if lcount==0: 
            nwstr+=m
            continue
        if bFlag and m==' ': continue
        if m=='\n' and lcount>0 and scount<linenum :
            bFlag=True
            continue
        bFlag=False
        nwstr+=m
        if(m=='\n'): scount=0
        scount+=1
    return nwstr
        
def writejsondictFormat(dct,fp):
    json_str = json.dumps(dct,skipkeys=True
    ,ensure_ascii=False,indent=4)
    json_str=ReplaceJsonString(json_str)
    json_str+='\n'
    fp.write(json_str)
    fp.flush()
def writejsondictFormatFile(dct,strfile,mode='w+'):
    with open(strfile,mode) as fp:
        writejsondictFormat(dct,fp)
        fp.flush()
def GetDefaultDict(strCurDir):
    strBasDir=basFunc.DeletePathLastSplit(strCurDir)
    strBasDir=os.path.dirname(strBasDir)
    dct={
            'mode':'train',
            'paramDir':os.path.join(strCurDir,'modelParam'),
            'trainDir':os.path.join(strBasDir,'train'),
            'validDir':os.path.join(strBasDir,'valid'),#cross valid 请使用None
            'testDir':os.path.join(strBasDir,'test'),
            'bestParamDir':os.path.join(strCurDir,'bestModelParam'),
            'gpuid':[-1],
            'epchos':10000,
            'learningrate':0.001,
            'batchsize':16,
            'validepoch':20,
            'validMinEpoch':1000,
            'paramEpoch':5,
            'epcho':0,
            'crossValidTrainNum':10,
            'crossValidShuffleDex':4,
            'crossMinValidDex':30,
            'crossValidTrainLoop':40,
            'curParamDir':None,
            'ResetMode':True,
            'debugTrainMode':True,
            'gpuflag':False,
            'OptLoop':1,
            'imgaug':True,
            'backBoneLr':0.00025,
            'weightdecay':0.000025,
            'stepsize':20,
            'gamma':0.99,
            'EnumStopStep':-1,
            'TimeLeft':-1,
            'ShowCurveLoop':1
        }
    return dct
class Config(object):
    """
    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """

        ret = Config(vars(self))
        
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)
    
    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)        

def getjsdatlstlinlst(data_file):
    with open(data_file) as f:
        data = []
        for line in f.readlines():
            dic = json.loads(line)
            dic=list(dic.values())
            data.append(dic)
    return data
def ConvertdctTolst(dct):
    lst=[]
    for a in dct:
        dic=list(a.values())
        lst.append(dic)
    return lst
def GetDictMethod():
    dct={'test':1}
    return dct
def InitJsonConfig(DictMethod,filename,rewrite=True):
    dct=DictMethod()
    dcts=None
    if(os.path.exists(filename)):
        dcts=getjsondata(filename)
    if(not dcts):
        dct=DictMethod()
    else:
        dct.update(dcts)
    if rewrite or not os.path.exists(filename):
        with open(filename,'w') as fp:
            writejsondictFormat(dct,fp)
    return dct
def GetFrameData(videoPath,iFrameStart=-1,iFrameEnd=-1,iSample=-1):
    cap = cv2.VideoCapture(videoPath)
    wid = int(cap.get(3))
    hei = int(cap.get(4))
    framerate = int(cap.get(5))
    framenum = int(cap.get(7)) 
    if(iFrameEnd<0): iFrameEnd=framenum+1
    framenum=min(framenum,iFrameEnd-iFrameStart+1)
    if(iSample<0):sep=1
    else:sep=iSample
    video = np.zeros((int(framenum/sep),hei,wid,3),dtype=np.int8)    
    numFrame=0
    count=0
    if iFrameStart>1:
        cap.set(cv2.CAP_PROP_POS_FRAMES,iFrameStart)
        if iFrameEnd<0:
            iFrameEnd=framenum-1
        framenum-=iFrameStart
        iFrameEnd-=iFrameStart
        iFrameStart=0
    while True:
        if(cap.grab()):
            flag, frame = cap.retrieve()
            if (not flag):
                break
            else:
                numFrame += 1
                if(iSample>0 and numFrame%iSample!=0):continue
                if(numFrame<iFrameStart): continue
                count+=1
                if(iFrameEnd !=-1 and numFrame>iFrameEnd): break  
                b=np.asarray(frame,dtype=np.int8)
                video[count-1]=b
                print("Process:",numFrame,'/',framenum,end='\r',flush=True)
                
        else:
            break;
    return video
#
def SaveFrameData(videoPath,svPath,iFrameStart=-1,iFrameEnd=-1,iSample=-1):
    cap = cv2.VideoCapture(videoPath)
    wid = int(cap.get(3))
    hei = int(cap.get(4))
    framerate = int(cap.get(5))
    framenum = int(cap.get(7)) 
    if(iFrameEnd<0): iFrameEnd=framenum+1
    framenum=min(framenum,iFrameEnd-iFrameStart+1)
    if(iSample<0):sep=1
    else:sep=iSample
    video = np.zeros((framenum/sep,hei,wid,3),dtype=np.int8)    
    numFrame=0
    count=0
    if iFrameStart>1:
        cap.set(cv2.CAP_PROP_POS_FRAMES,iFrameStart)
        if iFrameEnd<0:
            iFrameEnd=framenum-1
        framenum-=iFrameStart
        iFrameEnd-=iFrameStart
        iFrameStart=0
    while True:
        if(cap.grab()):
            flag, frame = cap.retrieve()
            if (not flag):
                break
            else:
                numFrame += 1                
                if(iSample>0 and numFrame%iSample!=0):continue
                if(numFrame<iFrameStart): continue
                count+=1
                if(iFrameEnd !=-1 and numFrame>iFrameEnd): break  
                b=np.asarray(frame,dtype=np.int8)
                video[count]=b
                print("Process:",numFrame,'/',framenum,end='\r',flush=True)                
        else:
            break;
    print('\nendProcess,saving...')
    video.tofile(svPath)
def SaveFrame(videoPath, svDir,iFrameStart=-1,iFrameEnd=-1,iSample=-1):
    cap = cv2.VideoCapture(videoPath)
    framenum = int(cap.get(7)) 
    numFrame = 0
    count=0
    if iFrameStart>1:
        cap.set(cv2.CAP_PROP_POS_FRAMES,iFrameStart)
        if iFrameEnd<0:
            iFrameEnd=framenum-1
        framenum-=iFrameStart
        iFrameEnd-=iFrameStart
        iFrameStart=0
    while True:
        if(cap.grab()):
            flag, frame = cap.retrieve()
            if (not flag):
                break
            else:
                numFrame += 1
                if(numFrame<iFrameStart): continue
                if(iSample>0 and numFrame!=1 and numFrame%iSample!=0):continue
                count+=1
                if(iFrameEnd !=-1 and numFrame>iFrameEnd): break
                print("Process:",numFrame,'/',framenum,end='\r',flush=True)                
                newPath = os.path.join(svDir , str(count) + ".jpg")
                cv2.imencode('.jpg', frame)[1].tofile(newPath)
        else:
            break;
def CutSaveMp4File(file,svfile,iStart=-1,iEnd=-1,iSample=-1,loopfunc=None,fps=60,size=None):
    cap = cv2.VideoCapture(file)
    #注意size这部分要求输入的图像尺寸要和size一致否则生不出MP4
    #特别注意
    defaulsize=(int(cap.get(3)),int(cap.get(4)))
    if size==None:
        size=defaulsize
    wrt=cv2.VideoWriter(svfile,cv2.VideoWriter_fourcc('X','V','I','D'),fps,size)
    framenum = int(cap.get(7)) 
    numFrame = 0
    count=0    
    while True:
        if(cap.grab()):
            flag, frame = cap.retrieve()
            if (not flag):
                break
            else:
                numFrame += 1
                if(numFrame<iStart): continue
                if(iSample>0 and numFrame%iSample!=0):continue
                count+=1
                if(iEnd !=-1 and numFrame>iEnd): break
                print("Process:",numFrame,'/',framenum,end='\r',flush=True)    
                if(loopfunc):frame=loopfunc(count,frame,size)
                wrt.write(frame)
        else:
            break;
    cap.release()
    wrt.release()
def GetArgs(argv,largeArglst):
    num=min(len(largeArglst),26)
    ch=ord('a')
    smallArglst=''
    retsmlArgs=[]
    retlargArgs=[]
    for i in range(num):
        c=chr(ch+i)
        smallArglst+=c
        if(largeArglst[i][-1]=='='):
            smallArglst+=':'
        retsmlArgs.append('-'+c)
        retlargArgs.append('--'+largeArglst[i][:-1])
    opts,args=gopt.getopt(argv[1:],smallArglst,largeArglst)
    return opts,args,{'smallArgs':retsmlArgs,'largeArgs':retlargArgs}
def WriteLine(fp,strline,end='\n'):
    fp.write(strline+end)
def ReadFileMultiLinesConvOneArr(strfile):
    fp=open(strfile,'r')
    txtlines=[]
    while(True):
        txtline=fp.readline()
        idex=txtline.find('\n')
        txtline=txtline[:idex]+txtline[idex+1:]
        if(not txtline):
            break
        txtlines.append(txtline)
    fp.close()
    return txtlines
def ReadMultiLines(filename):
    fp=open(filename,'r')
    lines=fp.readlines()
    fp.close()
    return lines
def WriteMultiLines(filename,lines,end=None):
    fp=open(filename,'w')
    if(not end):
        fp.writelines(lines)
    else:
        for i in range(len(lines)):
            WriteLine(fp,lines[i],end)
    fp.close()
    return
if(__name__=="__main__"):
    opts,args,_=GetArgs(sys.argv,['mode=','usefull'])
    for names,values in opts:
        if(names in ('--mode')):
            print(values)
        if(names in ('--usefull')):
            print(True)
    parser = argparse.ArgumentParser()      
    strDir=b'/data/BITMAN/crowdtest/'
    parser.add_argument('--verbose', required=True, type=int)
    #required标签就是说--verbose参数是必需的，并且类型为int，输入别的类型会报错。
    parser.add_argument('--gpu_id', type=int, default="2", help='The gpu id')
    parser.add_argument('--meta') #带--都是可选参数   
    parser.add_argument('data') #必须存在的参数
    args = parser.parse_args()  
    m=args.data
    os.environ['CUDA_VISIBLE_DEVICE'] = str(args.gpu_id)
    if(args.meta): print('1')
    a=np.zeros([1,1])
    a.tofile('a.bin')
    b=np.fromfile('a.bin',dtype=np.float)
    ###readline相关请搜索该文件即可

    
    
    