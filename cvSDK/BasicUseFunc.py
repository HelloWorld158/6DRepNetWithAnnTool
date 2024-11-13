import os
import sys
sys.path.append(os.path.dirname(__file__))



#是否自动安装模块，如果自动安装设置为True，否则设置为False
bAutoSetupModule=True
bBasFuncFilePos=False
#moduleInstDir=os.path.dirname(__file__)
moduleInstDir=os.getcwd()
givMem=-1



printfunc=None
if 'BASPRINTFUNC' in os.environ:
    '''
    具体使用方法:
    otherfile
    import os
    print_ori = print
    class Logger(object):
        def __init__(self):
            pass
        def info(self, content):   # 传递普通参数和关键字指定的参数
            print_ori('info:', content)
    logger = Logger()
    def my_print(*args):
        args_list = [str(v) for v in args]
        res_str = ' '.join(args_list)
        logger.info(res_str)
    os.environ['BASPRINTFUNC']='from otherfile import my_print'
    import BasicUseFunc as basFunc
    可使用os.system下面
    '''
    import builtins
    exec(os.environ['BASPRINTFUNC']+' as printfunc')
    builtins.__dict__['print'] = printfunc
def ImportModules(modulelst,installModlst,sources=['https://pypi.tuna.tsinghua.edu.cn/simple','https://mirrors.aliyun.com/pypi/simple']):
    for i in range(len(modulelst)):
        res=-1
        for s in sources:
            print('try source:',s)
            res=os.system('pip3 install -i '+s+' '+installModlst[i])
            if(res==0):break
        print('install result:',res==0)
        if(res!=0):
            print('Error Install Failed:',modulelst[i],',install:',installModlst[i])
            continue
    return

    
moduleDct={
        #'tensorflow':'tensorflow-gpu==1.14.0',
        'matplotlib':'matplotlib',
        'PIL':'pillow',
        'pynvml':'nvidia-ml-py3',
        'numpy':'numpy',
        #'cv2':'opencv-python',
        #'cv2contrib':'opencv-contrib-python',
        #'imgaug':'imgaug',
        #'webcolors':'webcolors',
        #'tensorboardX':'tensorboardX',
        'requests':'requests'        
    }
specModuleDct={
        #'torch':'pip3 --default-timeout=100 install torch==1.4.0 torchvision==0.5.0 -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com'
    }
def ImportSpModules():
    global specModuleDct
    for kk,vv in specModuleDct.items():
        res=-1
        res=os.system(vv)
        print('install result:',res==0)
        if(res!=0):
            print('Error Install Failed:',kk,',install:',vv)
            continue
def GetDefaultInstallModule(module):
    global moduleDct
    dct=moduleDct
    if(module in dct.keys()):
        return dct[module]
    return None
    
def ImportDefaultModule(sources=['https://pypi.tuna.tsinghua.edu.cn/simple','https://mirrors.aliyun.com/pypi/simple','http://pypi.mirrors.ustc.edu.cn/simple','http://pypi.douban.com/simple']):
    global moduleDct
    installModules=[]
    modulelst=[key for key,value in moduleDct.items()]
    for i in range(len(modulelst)):
        res=GetDefaultInstallModule(modulelst[i])
        if(not res):
            print('No Default Module Name:',modulelst[i])
            exit()
        installModules.append(res)
    return ImportModules(modulelst,installModules,sources)
def DeletePathLastSplit(path,spt=['\\','/']):
    newpath=path
    if(len(path)!=0):
        for s in spt:
            if(path[-1]==s):
                newpath=path[:-1]
    return newpath
def FeedModuleDict(path,dct):
    fp=open(path,'r')
    lines=fp.readlines()
    fp.close()
    for i in range(len(lines)):
        ndctstr=DeletePathLastSplit(lines[i],'\n')
        if(len(ndctstr)):
            dct[i]=ndctstr
    return dct
def ReadInstallModule():
    global moduleDct,specModuleDct
    ndir=os.getcwd()
    if(bBasFuncFilePos):
        ndir=os.path.abspath(os.path.dirname(__file__))
    path=os.path.join(ndir,'moduleDct.cfg')
    if(os.path.exists(path)):
        moduleDct=FeedModuleDict(path,moduleDct)
    else:
        fp=open(path,'w')
        fp.close()
    path=os.path.join(os.getcwd(),'specModuleDct.cfg')
    if(os.path.exists(path)):
        specModuleDct=FeedModuleDict(path,specModuleDct) 
    else:
        fp=open(path,'w')
        fp.close()
curflagFile=os.path.join(os.getcwd(),'ModuleInstalled.cvg')
if(bAutoSetupModule and os.name!='nt' and not os.path.exists(curflagFile)):
    ReadInstallModule()
    ImportDefaultModule() 
    ImportSpModules()          
    fp=open(curflagFile,'w')
    for kk,vv in moduleDct.items():
        fp.write('ModuleInstalled:'+vv+'\n')
    for kk,vv in specModuleDct.items():
        fp.write('ModuleInstalled:'+vv+'\n')
    fp.close()        
else:
    print('----------------------------------------warning----------------------------------------------')
    print('Don\'t execute Installed,if you need.please Delete current Directory File:ModuleInstalled.cvg')
    print('if ModuleInstalled.cvg is not exists.Please Check this Py file bAutoSetupModule is not False.')
    print('----------------------------------------warning----------------------------------------------')
import numpy as np
import glob
import shutil
import re
import pynvml as nv
import uuid   
from typing import List
def GPUExportSymbol():
    os.system('export PATH=/usr/local/cuda/bin:$PATH')
    os.system('export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH ')
    os.system('export CUDA_HOME=/usr/local/cuda')
    return    
print('operator system:',os.name)
if(os.name!='nt'):
    nv.nvmlInit()
gpuLst=None
gpuCnt=0
def GetModuleFileNumber(matchstr,defaultNum):
    if(givMem>0):return givMem
    file=os.path.join(moduleInstDir,'ModuleInstalled.cvg')
    if(not os.path.exists(file)):
        return defaultNum
    fp=open(file,'r')
    txts=fp.readlines()
    dex=-1
    for i in range(len(txts)):
        txt =txts[i]
        if(matchstr==txt.split(':')[0]):
            defaultNum=float(txt.split(':')[1])
            dex=i
    fp.close()
    fp=open(file,'w')
    if(dex!=-1):
        txts[dex]=matchstr+':'+str(defaultNum)+'\n'
    else:
        txts.append(matchstr+':'+str(defaultNum)+'\n')
    fp.writelines(txts)
    fp.close()
    return defaultNum
#os.environ['CUDA_VISIBLE_DEVICES']=str(gpulst[0])
def GetAvailableGPUsList(freeMem=7000,printfunc=print):#单位MB
    freeMem=GetModuleFileNumber('GPUFreeMem',freeMem)
    global gpuCnt,gpuLst
    if(gpuCnt!=0):
        return gpuLst,gpuCnt
    count=0
    gpulst=[]
    gpuMem=[]
    deviceCount = nv.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nv.nvmlDeviceGetHandleByIndex(i)
        info = nv.nvmlDeviceGetMemoryInfo(handle)
        free=float(info.free)/(1024.0*1024.0)
        gpuMem.append(round(free,3))
        if(free>freeMem):
            gpulst.append(i)
    printfunc('=================================GPU Setting===========================================')
    printfunc('Avaliable GPU list:',gpulst)
    printfunc('All GPUFreeMem:',gpuMem)
    printfunc('Require GPUFreeMem Setting:',freeMem,' Please Setting your Number in ModuleInstalled.cvg GPUFreeMem.')
    printfunc('=======================================================================================')
    if(len(gpulst)==0): gpulst=None
    gpuLst=gpulst
    gpuCnt=deviceCount
    return gpulst,deviceCount
def GetAvailableGPUsListRate(freerate=0.99):
    freerate=GetModuleFileNumber('GPUFreeRate',freerate)
    global gpuCnt,gpuLst
    if(gpuCnt!=0):
        return gpuLst,gpuCnt
    count=0
    gpulst=[]
    gpuRate=[]
    deviceCount = nv.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nv.nvmlDeviceGetHandleByIndex(i)
        info = nv.nvmlDeviceGetMemoryInfo(handle)
        rate=float(info.free)/float(info.total)
        gpuRate.append(round(rate,3))
        if(rate>freerate):
            gpulst.append(i)
    print('Avaliable GPU list:',gpulst)
    print('All GPUFreeRate:',gpuRate)
    if(len(gpulst)==0): gpulst=None
    gpuLst=gpulst
    gpuCnt=deviceCount
    return gpulst,deviceCount
#递归获取文件夹及子文件夹全部图像filter不要加*
def get_filelist(dir, Filelist,filter='.jpg'):
    newDir = dir
    if os.path.isfile(dir): 
        _,_,ex=GetfileDirNamefilter(dir)
        if(ex.lower()==filter or filter=='.*'):
            Filelist.append(dir) 
    elif os.path.isdir(dir): 
        for s in os.listdir(dir): 
            # 如果需要忽略某些文件夹，使用以下代码 
            #if s == "xxx": 
                #continue 
            newDir=os.path.join(dir,s) 
            get_filelist(newDir, Filelist,filter) 
    return Filelist
def get_dirlist(dir,dirlist):
    newdir=dir
    if(os.path.isdir(dir)):
        dirlist.append(dir)
        for s in os.listdir(dir): 
            newDir=os.path.join(dir,s)             
            get_dirlist(newDir, dirlist)
    return dirlist
def SearchDirectorysFromName(dir,name):
    dirlst=[]
    dirlst=get_dirlist(dir,dirlst)
    retDirs=[]
    for d in dirlst:
        bn=os.path.basename(d)
        if(bn==name):
            retDirs.append(d)
    return retDirs
##函数尚未测试
def SearchFilesFromName(dir,name):
    filelst=[]
    shotname, extension = os.path.splitext(name)
    filelst=get_filelist(dir,filelst,extension)
    retfiles=[]
    for d in filelst:
        bn=os.path.basename(d)
        if(bn==name):
            retfiles.append(d)
    return retfiles
#返回路径 文件名 文件后缀
def GetfileDirNamefilter(filename):
    filepath, tmpfilename = os.path.split(filename)
    shotname, extension = os.path.splitext(tmpfilename)
    return filepath, shotname, extension
def GetCurrentFileDir(filename):
    return os.path.dirname(os.path.abspath(filename))
#获取当前文件夹下全部后缀名相关的文件
def getdatas(imagedir,filter='*.jpg'):
    imagelist=[file for file in glob.glob(os.path.join(imagedir,filter))]
    finalimglst=[]
    for imgfile in imagelist:
        fileid=os.path.splitext(os.path.basename(imgfile))[0]
        finalimglst.append(imgfile) 
    return finalimglst
def Process(pros,alllen,name='Process:',flush=True):
    if(flush):
        print(name,pros,'/',alllen,'\t\t\t',end='\r',flush=True)
    else:
        print(name,pros,'/',alllen)
def removeDir(dirPath):
    if not os.path.isdir(dirPath):
        return
    files = os.listdir(dirPath)
    for file in files:
        filePath = os.path.join(dirPath, file)
        if os.path.isfile(filePath):
            os.remove(filePath)
        elif os.path.isdir(filePath):
            removeDir(filePath)
    os.rmdir(dirPath)
def MakeExistRetDir(dirname):
    if(not os.path.exists(dirname)):
        os.mkdir(dirname)
def MakeEmptyDir(dirname):
    if(os.path.exists(dirname)):
        removeDir(dirname)
    os.mkdir(dirname)
    return dirname
def CopyFiles(dstdir,srcdir,filter):
    lstfile=getdatas(srcdir,filter)
    for file in lstfile:
        _,name,ext=GetfileDirNamefilter(file)
        dstfile=os.path.join(dstdir,name+ext)
        shutil.copy(file,dstfile)
def RenameGroupFiles(dirname,strfilename,groupfilter=['.jpg']):
    if(len(groupfilter)==0): return
    lstfile=getdatas(dirname,'*'+groupfilter[0])
    count=0
    for file in lstfile:
        dir,name,ext=GetfileDirNamefilter(file)
        for filter in groupfilter:
            nfilename=os.path.join(dir,name+filter)
            if(not os.path.exists(nfilename)): continue
            os.rename(nfilename,os.path.join(dir,strfilename+str(count)+filter))
            count+=1
#按顺序生成新的文件夹，basDir基础的目录，newDir，如果存在newDir会在后面加一判断
def GenerateOrderNumberDir(basDir,newDir):
    iNum=0
    testDir=os.path.join(basDir,newDir+str(iNum))
    lastDir=None
    while(os.path.exists(testDir)):
        lastDir=testDir
        iNum+=1
        testDir=os.path.join(basDir,newDir+str(iNum))
    os.mkdir(testDir)
    return testDir,lastDir
def search(path,name):
    for root, dirs, files in os.walk(path):  # path 为根目录
        if name in dirs or name in files:
            flag = 1      #判断是否找到文件
            root = str(root)
            dirs = str(dirs)
            return os.path.join(root, dirs)
    return -1    
#lst=np.zeros([len(files)],dtype=np.int)
def GetMinFrame(lst,files):
    minDex=-1
    curDex=-1
    for i in range(len(files)):
        if(lst[i]!=0):continue
        file=files[i]
        _,name,_=GetfileDirNamefilter(file)
        dex=[int(s) for s in re.findall(r'\b\d+\b', name)]
        dex=dex[-1]
        if(minDex<0 or dex<minDex):
            minDex=dex
            curDex=i
    if(curDex>=0):
        lst[curDex]=1
        return curDex
    return None

#def LoopFunc(basAlert,obj,param):
#    import BasicAlgorismGeo as basGeo
#    maxiou=-1
#    maxkey=None
#    for key in basAlert.curObjlst.keys():
#        bobj=basAlert.curObjlst[key]['object']
#        curiou=basGeo.ioubox(bobj,obj)
#        if(maxiou<0 or maxiou<curiou):
#            maxiou=curiou
#            maxkey=key
#    if(maxiou>param):
#        basAlert.UpdateKeyObjs(obj,maxkey)
#    else:
#        newkey=basAlert.GetNewKey()
#        basAlert.UpdateKeyObjs(obj,newkey)
class BasicAlertObject(object):
    def __init__(sf,LoopFunc,destroyTimes=5000,fitTimes=90,debugDir=None,DebugLoopFunc=None):
        sf.LoopFunc=LoopFunc
        sf.count=0
        sf.curObjlst=None
        sf.destroyTimes=destroyTimes
        sf.fitTimes=fitTimes
        sf.ClearLoopCount()
        sf.debugDir=debugDir
        sf.DebugLoopFunc=DebugLoopFunc
        if(sf.debugDir):
            MakeEmptyDir(debugDir)
        return
    def RefineAllKeys(sf):
        newkeys=set()
        if(not sf.curObjlst):
            return
        for key in sf.curObjlst.keys():
            value=sf.curObjlst[key]
            if(value['destroyCount']>sf.destroyTimes or value['destroyCount']-value['fitTimes']>sf.fitTimes):
                newkeys.add(key)
            sf.curObjlst[key]['destroyCount']+=1
        sf.curObjlst=sf.GatherKeyObjectNot(newkeys)
        return        
    def GenNewKey(sf):
        return str(uuid.uuid1())
    def GatherKeyObject(sf,keys):
        keyobjs={k:v for k,v in sf.curObjlst.items() if k in keys}
        return keyobjs
    def GatherKeyObjectNot(sf,keys):
        keyobjs={k:v for k,v in sf.curObjlst.items() if k not in keys}
        return keyobjs
    def UpdateKeyObjs(sf,obj,key,updateflag=True):
        newkey=None
        if key not in sf.curObjlst:
            sf.curObjlst[key]=dict()
            sf.curObjlst[key]['object']=obj
            sf.curObjlst[key]['destroyCount']=0
            sf.curObjlst[key]['fitTimes']=0
            newkey=key
        else:
            if(updateflag):
                sf.curObjlst[key]['object']=obj
            sf.curObjlst[key]['fitTimes']+=1
        return newkey
    def ClearLoopCount(sf):
        sf.count=0
        sf.curObjlst=dict()
    def __call__(sf,objlst,param=None,refineFlag=True,debugfile=None):        
        newkeys=set()
        for obj in objlst:
            key=sf.LoopFunc(sf,obj,param)
            if(key):
                newkeys.add(key)
        if(refineFlag):
            sf.RefineAllKeys()
        if(sf.DebugLoopFunc):
            sf.DebugLoopFunc(sf,debugfile)
        sf.count+=1
        if(len(newkeys)==0): return None
        return sf.GatherKeyObject(newkeys)    
#[extDir]=GenerateEmtyDir(['extDir'])
def GenerateEmtyDir(dirs,curDir=os.getcwd()):
    retdirs=[]
    for dir in dirs:
        ndir=os.path.join(curDir,dir)
        retdirs.append(ndir)
        MakeEmptyDir(ndir)
    return retdirs
def GetCurDirNames(dirfiles,curDir=os.getcwd()):
    retNames=[]
    for df in dirfiles:
        ndf=os.path.join(curDir,df)
        retNames.append(ndf)
    return retNames
#注意所有函数下面：start<end       
def Findtxtlinelst(lines,matchstr,startPos=-1,endPos=-1,strStartPos=-1,strEndPos=-1
,reverse=False,reverseStr=False):
    if(endPos<0):
        endPos=len(lines)
    if(startPos<0):
        startPos=0
    if(reverse):
        pos=startPos
        startPos=endPos
        endPos=pos
    linDex=-1
    curLineDex=-1
    step=1
    if(reverse):step=-1
    for i in range(startPos,endPos,step):
        line=lines[i]
        if(i==startPos):
            if(strStartPos<0):strStartPos=0
            if(strEndPos<0):strEndPos=len(line)
        else:
            strStartPos=0
            strEndPos=len(line)
        if(reverseStr):
            linDex=line.rfind(matchstr,strStartPos,strEndPos)
        else:
            linDex=line.find(matchstr,strStartPos,strEndPos)
        if(linDex>=0):
            curLineDex=i
            break
    return linDex,curLineDex
def AddTailStrlists(lines,end='\n'):
    for i in range(len(lines)):
        lines[i]+=end
    return lines    
import time,urllib
lastcnt=0
lastime=0
def report(count, blockSize, totalSize):
    percent = int(count*blockSize*100/totalSize)
    global lastcnt,lastime
    delta=count*blockSize-lastcnt
    deltatime=time.time()-lastime
    lastime=time.time()
    lastcnt=count*blockSize
    print('percent:',percent,'speed:',lastcnt/lastime,'',end='\r',flush=True)

def DownloadFile(url,filename):
    lastcnt=0
    lastime=time.time()
    urllib.request.urlretrieve(url,filename,reporthook=report)
#如果duplist是个只有数字的列表建议使用duplist=list(set(duplist))来去除重复
def RemoveDuplicateElemList(duplist):
    newlst=[]
    while(len(duplist)):
        elem=duplist.pop(0)
        newlst.append(elem)
        while(elem in duplist):
            duplist.pop(duplist.index(elem))
    return newlst
def ChangePythonLine(txts,lineDex,changeTxt,labelDex):
    txts[lineDex]=changeTxt+'  ###label'+str(labelDex)+',不要修改或删除这个注释，否则会影响代码中某些逻辑###\n'
    return txts
#def ChangeTxtContent(lines):
#    return lines
def ChangeTxtFileContent(file,ChangeTxtContent,end='\n'):
    fp =open(file,'r',encoding='utf-8', errors='ignore')
    lines=fp.readlines()
    fp.close()
    lines=ChangeTxtContent(lines)
    for i in range(len(lines)):
        lines[i]=DeletePathLastSplit(lines[i],['\n'])
        lines[i]+=end
    fp=open(file,'w',encoding='utf-8', errors='ignore')
    fp.writelines(lines)
    fp.close()
def ChangePythonFileFromLabel(file,changeTxtlst,labelDexes=[],end='\n'):
    fp =open(file,'r',encoding='utf-8', errors='ignore')
    lines=fp.readlines()
    fp.close()
    if(len(labelDexes)==0):
        labelDexes=[i for i in range(len(changeTxtlst))]
    for i in range(len(labelDexes)):
        label='###label'+str(labelDexes[i])
        _,dex=Findtxtlinelst(lines,label)
        lines=ChangePythonLine(lines,dex,changeTxtlst[i],labelDexes[i])
    for i in range(len(lines)):
        lines[i]=DeletePathLastSplit(lines[i],['\n'])
        lines[i]+=end
    fp=open(file,'w',encoding='utf-8', errors='ignore')
    fp.writelines(lines)
    fp.close()
def ReplaceAllMatchStr(txt,findtxt,replacetxt):
    start=0
    start=txt.find(findtxt)
    end=txt.find(replacetxt)
    while(start!=-1):
        if(start!=end or (start==end and len(replacetxt)<len(findtxt))):
            m=txt[start:].replace(findtxt,replacetxt)
            n=txt[:start]
            txt=n+m
        start+=len(replacetxt)
        txtstart=txt.find(findtxt,start)
        end=txt.find(replacetxt,start)
        start=txtstart
    return txt
def GetPythonDir():
    path=os.__file__
    return os.path.abspath(os.path.dirname(path))
def GetRelativeRootFile(fullPath,root):
    iRet=fullPath.find(root)
    if(iRet<0):return []
    dirlst=[]
    while(True):
        dirPath=os.path.dirname(fullPath)
        basPath=os.path.basename(fullPath)
        if(dirPath==root):
            dirlst.insert(0,basPath)
            dirlst.insert(0,root)
            break
        else:
            dirlst.insert(0,basPath)
        fullPath=dirPath
    return dirlst
#dct={'test_gatherkeys':['a','b','c'],'test_func':'a=5\nb=6\nc=a+b','test_file':'test.py'}
def RunExeDctFuncFile(exedct,regname,gathername=None,defaultadd=False,defaultarg=[]):
    if gathername is None:gathername=regname
    reskeys=exedct.get(gathername+'_gatherkeys',[])
    result=[]
    runflag=False
    for kk,vv in exedct.items():
        if kk.find(regname)<0:continue
        if kk!=regname+'_file' and kk!=regname+'_func':
            continue
        if kk.find('_file')>=0:
            fp=open(exedct[kk],'r')
            txts=fp.read()
            fp.close()
            loc=locals()
            exec(txts)            
            runflag=True
            break
        elif kk.find('_func')>=0:
            loc=locals()
            exec(exedct[kk])
            runflag=True
            break
    if not runflag:
        return defaultarg
    for key in reskeys:
        result.append(loc[key])
    if defaultadd:
        result.extend(defaultarg)
    return result 
def SetAvailableGPUsList(freeMem:int=7000,usegpuCount:int=1,printfunc=print):
    gpulst,gpucnt=GetAvailableGPUsList(freeMem,printfunc)
    usegpuCount=min(usegpuCount,len(gpulst))
    uselst=','.join([str(gpulst[i]) for i in range(usegpuCount)])
    printfunc('Use GPUList:',uselst)
    if usegpuCount<1:return gpulst,gpucnt
    os.environ['CUDA_VISIBLE_DEVICES']=uselst
    return gpulst,gpucnt
class FileDirLst(object):
    def __init__(self,filesDirs,name,flush=True
                 ,parentDir='',loopDir=False,
                 iterdexlst=None) -> None:
        self.fileDirs=filesDirs
        self.parentDir=parentDir
        self.loopDir=loopDir
        self.dexlst=iterdexlst
        self.name=name
        self.flush=flush
    def GetAll(self):
        for i,fileDir in enumerate(self.fileDirs):
            Process(i,len(self.fileDirs))
            yield(fileDir)
    def GetItem(self):
        for i,fileDir in enumerate(self.fileDirs):
            Process(i,len(self.fileDirs))
            yield(fileDir[0])
    def GetLastItems(self):
        for i,fileDir in enumerate(self.fileDirs):
            Process(i,len(self.fileDirs))
            yield(fileDir[1:])
    def __len__(self):
        return len(self.fileDirs)
    def __iter__(self):
        self.iter=0
        return self
    def __next__(self):
        if self.iter<len(self.fileDirs):
            data=self.fileDirs[self.iter]
            Process(self.iter,len(self.fileDirs),name=self.name,flush=self.flush)
            self.iter+=1
            if self.dexlst is not None:
                res=[data[dex] for dex in self.dexlst]
                if len(res)==1:
                    res=res[0]
            else:
                res=data
            return res
        else:
            raise StopIteration
def GetFileDirLst(fileDir:str,loopDir:bool=False,recursion:bool=False
                  ,filter:str='*.jpg',iterdexlst:List[int]=None
                  ,procname:str='Process',flush:bool=True,useabspath:bool=True)->FileDirLst:
    if loopDir:
        dirs=[]
        dirs=get_dirlist(fileDir,dirs)
        results=[]
        for dir in dirs:
            if dir==fileDir:continue
            if useabspath:
                dir=os.path.abspath(dir)
            basdir,basname=os.path.dirname(dir),os.path.basename(dir)
            if basdir.find(fileDir)<0:continue
            if recursion:
                results.append([dir,basdir,basname])
            else:
                if basdir!=fileDir:continue
                results.append([dir,basdir,basname])
    else:
        if recursion:
            files=[]
            files=get_filelist(fileDir,files,filter[1:])
        else:
            files=getdatas(fileDir,filter)
        results=[]
        for file in files:
            if useabspath:
                file=os.path.abspath(file)
            d,name,ftr=GetfileDirNamefilter(file)
            results.append([file,d,name,ftr])
    result=FileDirLst(results,procname,flush,fileDir,loopDir,iterdexlst)
    return result
if(__name__=="__main__"):
    curDir=os.path.dirname(os.path.realpath(__file__))#获取当前文件的文件夹路径
    os.getcwd()#取当前目录
    os.path.basename('/op/op')
    op=DeletePathLastSplit('/op/op')
    os.path.basename(op)
    os.path.dirname(os.getcwd())#取当前目录上一级目录
    srcfile='a.txt'
    dstfile='b.txt'
    shutil.copy(srcfile,dstfile)
    os.remove(dstfile)
    os.rename(srcfile,dstfile)
    string="A1.45，b5，6.45，8.82"
    print(re.findall('(?:-|)[0-9]+(?:\.?[0-9]+|)(?:e-?[0-9]+|)',string))    
    [int(s) for s in re.findall('(?:-|)[0-9]+(?:\.?[0-9]+|)(?:e-?[0-9]+|)', 'he33llo 42 I\'m a 32 string 30')]
    [int(s) for s in re.findall('(?:-|)[0-9]+(?:\.?[0-9]+|)(?:e-?[0-9]+|)', 'he33llo 42 I\'m a 32 string 30')]
    basAlert=BasicAlertObject(LoopFunc)
    a=[[1,2,3,4]]
    altobjs=basAlert(a,7.5)
    fp=open('txtfile.txt','r')
    ls=fp.readlines()
    data=fp.read()
    dic = {1: 'D', 2: 'B', 3: 'B', 4: 'E', 5: 'A'}
    sorted(dic.items(), key=lambda x : (x[1]))
    lis = [{"age":20,"name":"a"},{"age":25,"name":"b"},{"age":10,"name":"c"}]
    lis=sorted(lis, key=lambda x :(x['age']))