import BasicUseFunc as basFunc
import numpy as np
import os
import sys
import glob
import shutil
import threading as thd
import multiprocessing as mprs
import signal
import threading
import multiprocessing
from shutil import copyfile
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from functools import partial
import time,re
import psutil 
from typing import List,Dict
import mmap
import pickle
import DataIO as dio
class BasicMMapMem:
    def __init__(self,synfile=os.path.join(os.getcwd(),'buffer.bin')
                 ,size=10*1024*1024
                 ,autoAddPkl=True) -> None:
        self.binfile=synfile
        d,name,ftr=basFunc.GetfileDirNamefilter(synfile)
        self.pklfile=os.path.join(d,name+'.pkl')
        if os.path.exists(self.pklfile):os.remove(self.pklfile)
        if os.path.exists(self.binfile):os.remove(self.binfile)
        self.pfile=os.open(self.binfile,os.O_RDWR|os.O_CREAT)
        os.write(self.pfile,bytes(size))
        self.fp=mmap.mmap(self.pfile,size,access = mmap.ACCESS_WRITE, flags=mmap.MAP_SHARED)
        self.allmems={}
        self.allsize=size
        self.nextpos=0
        self.autoAddPkl=autoAddPkl
    def DumpMem(self):
        dct={'nextpos':self.nextpos,'allmems':self.allmems,'allsize':self.allsize}
        dio.SaveVariableToPKL(self.pklfile,dct)
    def AddMem(self,size,funcname):
        if self.nextpos+size>=self.allsize:
            raise(f"nextpos {self.nextpos},size: {size}|allsize: {self.allsize}")
        self.allmems[funcname]=[self.nextpos,size]
        memname=funcname
        baspos=self.nextpos
        self.nextpos+=size
        if self.autoAddPkl:self.DumpMem()
        return baspos,memname
    def close(self):
        os.close(self.pfile)
    def GetPickleBuffer(self,memname):
        self.fp.seek(self.allmems[memname][0])
        buf=self.fp.read(8)
        size=int.from_bytes(buf,byteorder='little')
        if(size>self.allmems[memname][1]-8):
            raise(f'GetSize:{size},reserveSize:{self.allmems[memname]-8-1}')
        buf=self.fp.read(size)
        res=pickle.loads(buf)
        return res
    def WritePickleBuffer(self,memname,data):
        buffer=pickle.dumps(data)
        size=len(buffer)
        if(size>self.allmems[memname][1]-8):
            raise(f'GetSize:{size},reserveSize:{self.allmems[memname]-8-1}')
        self.fp.seek(self.allmems[memname][0])
        self.fp.write(size.to_bytes(8,'little'))
        self.fp.write(buffer)
        return
    def GetBufferByte(self,memname):
        self.fp.seek(self.allmems[memname][0])
        assert(self.allmems[memname][1]==1)
        buf=self.fp.read_byte()
        return buf
    def WriteBufferByte(self,memname,data):
        self.fp.seek(self.allmems[memname][0])
        assert(self.allmems[memname][1]==1)
        self.fp.write_byte(data)
        return
class BasicMMapMemClient(BasicMMapMem):
    def __init__(self,synfile=os.path.join(os.getcwd(),'buffer.bin')
                 ,autoLoadPkl=True,
                 size:int=None):
        self.binfile=synfile
        d,name,ftr=basFunc.GetfileDirNamefilter(synfile)
        self.pklfile=os.path.join(d,name+'.pkl')
        if autoLoadPkl:
            dct=dio.LoadVariablefromPKL(self.pklfile)
            self.allsize=dct['allsize']
            self.nextpos=dct['nextpos']
            self.allmems=dct['allmems']
        else:
            assert size is not None
            self.allmems={}
            self.allsize=size
            self.nextpos=0
        self.autoAddPkl=autoLoadPkl
        self.pfile=os.open(self.binfile,os.O_RDWR)
        self.fp=mmap.mmap(self.pfile,self.allsize,access = mmap.ACCESS_WRITE, flags=mmap.MAP_SHARED)
class MMapEvent:
    def __init__(self,mmapMem:BasicMMapMem,eventName:str,mempos:int=None) -> None:
        self.mem=mmapMem
        if mempos is None:
            self.eventpos,self.key=self.mem.AddMem(1,eventName)
            self.ResetEvent()
        else:
            self.eventpos,self.key=mempos,eventName        
    def SetEvent(self,state=1):
        self.mem.WriteBufferByte(self.key,state)
    def GetEventState(self):
        return self.mem.GetBufferByte(self.key)
    def ResetEvent(self):
        self.mem.WriteBufferByte(self.key,0)
    def Wait(self,timesleep=0.005):
        while(True):
            flg=self.mem.GetBufferByte(self.key)
            if flg==0:
                time.sleep(timesleep)
                continue
            else:
                break
        return    
class MMapBuffer:
    def __init__(self,mmapMem:BasicMMapMem,estSize:int,bufName:str,mempos:int=None) -> None:
        self.mem=mmapMem
        if mempos is None:
            self.eventpos,self.key=self.mem.AddMem(estSize+8,bufName)
        else:
            self.eventpos,self.key=mempos,bufName
    def GetBuffer(self):
        return self.mem.GetPickleBuffer(self.key)
    def WriteBuffer(self,data):
        return self.mem.WritePickleBuffer(self.key,data)
'''
主要使用这个模块进行进程间通信(MMapProcess)
通信方式通过内存映射进行,建议使用master 开启slaver 进程
代码思路如下:
A->open_bproc--->init{b}.wait->run{b}.set------------------------retev{b}.wait->retmem{b}.get
   B->initialize------------->run{b}.wait->运行->retmem{b}.write->retev{b}.set->run{b}.reset
                                  |                                                   |
                                  <----------------------------------------------------
其中A可以通过init{b}控制B是否运行结束
master 代码片段:
import cvSDK.MultiProcess as multproc
def InitProcess(memdct:Dict[str,multproc.MMapBuffer]
                ,eventdct:Dict[str,multproc.MMapEvent]
                ,procobj:multproc.MMapProcess):
    testout=procDir+'/testout'
    basFunc.MakeEmptyDir(testout)
    procobj.procs=[]
    times=2
    for i in range(times):
        procobj.InsertMem('retmem'+str(i),len(pickle.dumps(1.0)))
        procobj.InsertEvent('init'+str(i))
        procobj.InsertEvent('run'+str(i))
        procobj.InsertEvent('retev'+str(i))
    for i in range(times):
        env={'CUDA_VISIBLE_DEVICES':'1'}
        env.update(os.environ)
        procobj.procs.append(spro.Popen(['/usr/local/bin/python',procfile]+curargs+["--procIndex",str(i)],cwd=curDir,env=env))
        eventdct['init'+str(i)].Wait(0.01)
    return
def RunProcess(memdct:Dict[str,multproc.MMapBuffer]
                ,eventdct:Dict[str,multproc.MMapEvent]
                ,procobj:multproc.MMapProcess):
    for i in range(len(procobj.procs)):
        eventdct['run'+str(i)].SetEvent()
    eventlst=[]
    for i in range(len(procobj.procs)):
        eventlst.append(eventdct['retev'+str(i)])
    procobj.WaitEventLst(eventlst)
    #procobj.WaitEventLstFromFilter('init',len(procobj.procs),0.01)
    costimes=[]
    for i in range(len(procobj.procs)):
        costimes.append(memdct['retmem'+str(i)].GetBuffer())
    for i in range(len(procobj.procs)):
        eventdct['init'+str(i)].SetEvent(2)
        eventdct['run'+str(i)].SetEvent()
    return sum(costimes)/len(costimes)
if __name__=='__main__':
    proc=multproc.MMapProcess(RunProcess,InitProcess,useClient=False)
    costime=proc.run()
    proc.close()
    print('meancost:',costime)
slaver代码片段
import cvSDK.MultiProcess as multproc
from typing import Dict,List
import time
def InitProcessClient(memdct:Dict[str,multproc.MMapBuffer]
                ,eventdct:Dict[str,multproc.MMapEvent]
                ,procobj:multproc.MMapProcess):
    index=procobj.procdata['index']
    eventdct['init'+str(index)].SetEvent()
    return
def RunProcessClient(memdct:Dict[str,multproc.MMapBuffer]
                ,eventdct:Dict[str,multproc.MMapEvent]
                ,procobj:multproc.MMapProcess):
    costime=None
    index=procobj.procdata['index']
    while(True):
        eventdct['run'+str(index)].Wait(0.001)
        if(eventdct['init'+str(index)].GetEventState()==2):
            break
        costime=1.0
        memdct['retmem'+str(index)].WriteBuffer(costime)
        eventdct['retev'+str(index)].SetEvent()
        eventdct['run'+str(index)].ResetEvent()
    return costime
if __name__=='__main__':
    args=parse_args()
    lst=[]
    procdata={'index':args.procIndex}
    proc=multproc.MMapProcess(RunProcessClient,InitProcessClient,
                            procdata=procdata)
    proc.lst=lst
    proc.runclose()
'''
class MMapProcess:
    def __init__(self,
                 runcallback,initcallback=None,
                 memdict:dict={},eventlst:list=[],procdata:dict={},
                 synfile=os.path.join(os.getcwd(),'buffer.bin'),
                 synsize=10*1024*1024
                 ,useClient=True,
                 autoAddPkl=True) -> None:
        if not useClient:
            self.mem=BasicMMapMem(synfile,synsize,autoAddPkl)
        else:
            self.mem=BasicMMapMemClient(synfile,autoAddPkl,synsize)
        self.procdata=procdata
        self.memdict:Dict[str,MMapBuffer]={kk:MMapBuffer(self.mem,vv,kk) for kk,vv in memdict.items()}
        self.eventdict:Dict[str,MMapEvent]={l:MMapEvent(self.mem,l) for l in eventlst}
        self.UpdateMem()
        self.runcallback=runcallback
        self.initcallback=initcallback
        if initcallback is not None:
            self.initcallback(self.memdict,self.eventdict,self)
    def UpdateMem(self):
        if len(self.mem.allmems)==0:
            return
        allst=[]
        for kk,vv in self.mem.allmems.items():
            allst.append([kk,vv[0],vv[1]])
        finalst=list(sorted(allst,key=lambda x:x[1]))
        for f in finalst:
            if f[2]==1:
                self.eventdict[f[0]]=MMapEvent(self.mem,f[0],f[1])
                continue
            self.memdict[f[0]]=MMapBuffer(self.mem,f[2],f[0],f[1])
        return
    def InsertMem(self,memkey:str,memsize:int):
        self.memdict[memkey]=MMapBuffer(self.mem,memsize,memkey)
        return
    def InsertEvent(self,eventkey):
        self.eventdict[eventkey]=MMapEvent(self.mem,eventkey)
        return
    def InsertMemDct(self,memdct:dict):
        for kk,vv in memdct.items():
            self.memdict[kk]=MMapBuffer(self.mem,vv,kk)
    def InsertEventlst(self,eventlst):
        for event in eventlst:
            self.eventdict[event]=MMapEvent(self.mem,event)
    def ResetEventLst(self,eventlst:List[MMapEvent]):
        for event in eventlst:
            event.ResetEvent()
    def SetEventLst(self,eventlst:List[MMapEvent]):
        for event in eventlst:
            event.SetEvent()
    def WaitEventLst(self,eventlst:List[MMapEvent],timesleep=0.005):
        for event in eventlst:
            event.Wait(timesleep)
    #处理不包含maxDex的索引
    def ResetEventLstFromFilter(self,filter:str,maxDex:int):
        eventlst:List[MMapEvent]=[]
        for kk,vv in self.eventdict.items():
            if kk.find(filter)<0:continue
            dex=int(re.findall('(?:-|)[0-9]+(?:\.?[0-9]+|)(?:e-?[0-9]+|)',kk)[0])
            if dex>=maxDex:
                break
            eventlst.append(vv)
        for event in eventlst:
            event.ResetEvent()
    #处理不包含maxDex的索引
    def SetEventLstFromFilter(self,filter:str,maxDex:int):
        eventlst:List[MMapEvent]=[]
        for kk,vv in self.eventdict.items():
            if kk.find(filter)<0:continue
            dex=int(re.findall('(?:-|)[0-9]+(?:\.?[0-9]+|)(?:e-?[0-9]+|)',kk)[0])
            if dex>=maxDex:
                break
            eventlst.append(vv)
        for event in eventlst:
            event.SetEvent()
    #处理不包含maxDex的索引
    def WaitEventLstFromFilter(self,filter:str,maxDex:int,timesleep=0.005):
        eventlst:List[MMapEvent]=[]
        for kk,vv in self.eventdict.items():
            if kk.find(filter)<0:continue
            dex=int(re.findall('(?:-|)[0-9]+(?:\.?[0-9]+|)(?:e-?[0-9]+|)',kk)[0])
            if dex>=maxDex:
                break
            eventlst.append(vv)
        for event in eventlst:
            event.Wait(timesleep)
    def WaitEventLst(self,eventlst:List[MMapEvent],timesleep=0.005):
        for event in eventlst:
            event.Wait(timesleep)
    def run(self):
        obj=self.runcallback(self.memdict,self.eventdict,self)
        return obj
    def close(self):
        self.mem.close()
    def runclose(self):
        ret=self.run()
        self.close()
        return ret
def FuncStart(a,b,c):
    r=a+b-c
    return r
def FuncRun(r,d,e):
    m=r-d+e
    return m
def WaitEvent(ev):
    ev.wait()
    ev.clear()
def LoopFunc(dct,mainEv,funcEv):
    dct['r']=FuncStart(dct['a'],dct['b'],dct['c'])
    funcEv.set()
    while(dct['run']):
        WaitEvent(mainEv)
        dct['m']=FuncRun(dct['r'],dct['d'],dct['e'])
        funcEv.set()
class ParallelProcess(object):
    def __init__(self, num_workers=None, parallel_type='process'):
        """并行处理初始化函数

        Args:
            num_workders (int, optional): 使用多少进程或线程，None的时候默认使用服务器cpu核数，Defaults to None.
            parallel_type (str, optional): 使用多线程还是多进程 'thread'是线程，'process'是多进程. Defaults to 'process'.
        """    
        assert parallel_type in {'process', 'thread'}, 'Wrong parallel type!'
        self.num_workers = multiprocessing.cpu_count() if num_workers == None else num_workers
        
        if parallel_type == 'process':                              # 定义进程池和线程池
            # 注意多进程只能在主函数中执行，包含在if __name__ == '__main__'中
            self.parallel_method = Pool(processes=self.num_workers)
        else:
            self.parallel_method = ThreadPoolExecutor(max_workers=self.num_workers)
    
    def run(self, func, data):
        """多线程或多进程执行函数

        Args:
            func (functional object): 待并行执行的函数
            data (iterable object, 如list, tuple): 待处理的数据
        """        
        assert len(data) > 0, 'The number of data is wrong!'
        start = time.time()
        with self.parallel_method as pool:                         # 并行处理数据
            results = pool.map(func, data)        
        print(time.time() - start)
def GetAllProcess(attrlst=['pid', 'name', 'username', 'ppid',
                           'cpu_percent', 'memory_info','cmdline','cwd']):
    allprocess=[]
    for process in psutil.process_iter():
        try:
            process_info = process.as_dict(attrs=attrlst)
            allprocess.append(process_info)
        except:
            print(process,'Error not get info.')
            continue
    return allprocess
def FindProcesses(dct:Dict,processes:List[Dict]):
    def CheckValueIn(data,vv):
        if type(data) is int:
            return data==vv
        if type(data) is list:
            for d in data:
                if type(d) is str and d.find(vv)>=0:
                    return True
            return False
        if type(data) is str and data.find(vv)>=0:
            return True
        return False 
    svprocess=[]
    for process in processes:
        flag=True
        for kk,vv in dct.items():
            if not CheckValueIn(process[kk],vv):
                flag=False
                break
        if not flag:continue
        svprocess.append(process)
    return svprocess
def TraceAllSubProcess(processes,pid):
    flags=[False for _ in processes]
    for i,proc in enumerate(processes):
        if proc['pid']==pid:
            flags[i]=True
    trackprocs=[]
    pids=[pid]
    lastsize=-1
    while(lastsize<0 or len(trackprocs)!=lastsize):
        lastsize=len(trackprocs)
        for i,proc in enumerate(processes):
            if flags[i]:continue
            if proc['ppid'] in pids:
                pids.append(proc['pid'])
                trackprocs.append(proc)
                flags[i]=True
    return trackprocs
def TraceAllSubProcessFromProc(processes,frmproc):
    flags=[False for _ in processes]
    for i,proc in enumerate(processes):
        if proc['pid']==frmproc['pid']:
            flags[i]=True
    trackprocs=[]
    pids=[frmproc['pid']]
    lastsize=-1
    while(lastsize<0 or len(trackprocs)!=lastsize):
        lastsize=len(trackprocs)
        for i,proc in enumerate(processes):
            if flags[i]:continue
            if proc['ppid'] in pids:
                pids.append(proc['pid'])
                trackprocs.append(proc)
                flags[i]=True
    return trackprocs
def KillProcessFromPids(pids):
    for pid in pids:
        os.system(f'kill {pid}')
    return
def KillProcessFromProcs(procs):
    for proc in procs:
        os.system(f'kill {proc["pid"]}')
    return
if(__name__=="__main__"):
    process=[]
    def OutFuncStart(a,b,c):
        dct=mprs.Manager().dict()
        dct['run']=True
        dct['a']=a
        dct['b']=b
        dct['c']=c
        mainEv=mprs.Event()
        funcEv=mprs.Event()
        p=mprs.Process(target=LoopFunc,args=(dct,mainEv,funcEv))
        process.append(p)
        signal.signal(signal.SIGTERM, 0)
        funcEv.clear()
        mainEv.clear()
        p.daemon=True
        p.start()
        WaitEvent(funcEv)
        return (dct,p,mainEv,funcEv)
        
    def OutFuncRun(rs,d,e):
        (dct,p,mainEv,funcEv)=rs
        dct['d']=d
        dct['e']=e
        mainEv.set()
        WaitEvent(funcEv)
        return dct['m']
    data_type = 'distraction'  # 'distraction' | 'seatbelt'
    data_root = '/notebooks/zw/data/ai_view/haerbing/seatbelt_driver/src_imgs'
    dest_root = '/notebooks/zw/data/ai_view/haerbing/seatbelt_driver/src_imgs_en'

    rule_Chinese = re.compile(r'[\u4e00-\u9fa5]') 
    def change_name(src_path, dest_root):
        try:
            img_name = os.path.basename(src_path)
            print(img_name)
            img_name_en = rule_Chinese.sub('', img_name)          # 删除字符串中的汉字
            if not os.path.exists(dest_root):
                os.makedirs(dest_root)
            dest_path = os.path.join(dest_root, img_name_en)
            copyfile(src_path, dest_path)
        except Exception as err:
            print(err)
    a=OutFuncStart(1,2,3)
    b=OutFuncStart(4,5,6)
    r=OutFuncRun(b,7,8)
    k=OutFuncRun(a,9,10)
    s=OutFuncRun(a,2,10)
    r=OutFuncRun(a,1,10)
    # 在执行休眠类任务时多线程快一点，但执行复制图片时多进程快一点 
    # parallel_process = ParallelProcess(num_workers=32, parallel_type='thread')  # 8.93007516860962
    parallel_process = ParallelProcess(num_workers=32, parallel_type='process')   # 2.5

    img_paths = glob.glob(os.path.join(data_root, '*'))
    task_func = partial(change_name, dest_root=dest_root)
    parallel_process.run(task_func, img_paths)


