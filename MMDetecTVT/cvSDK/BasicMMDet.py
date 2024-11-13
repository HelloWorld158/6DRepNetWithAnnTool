import BasicUseFunc as basFunc
import mmdet as det
import os,sys
import re
import DataIO as dio
import numpy as np
import os.path as osp
from mmengine import Config
from mmengine.logging import print_log
import BasicConfig as basCfg
import logging
curDir=os.path.dirname(__file__)
def GetCudaInferDevice(gpulst):
    if(len(gpulst)==0):
        print('no gpu avalible,error')
        exit()
        return
    gpustr='cuda:'+str(gpulst[0])
    print('FinalGPUse:',gpulst[0])
    return gpustr
def ChangeURL(url):
    s=url
    dex=s.find('.com')
    lstwords=url[dex+4:]
    return 'https://open-mmlab.oss-cn-beijing.aliyuncs.com'+lstwords
def GetPthFileName(url):
    s=url
    lstStr=s.split('/')
    wkDir=os.path.dirname(curDir)
    wkDir=os.path.join(wkDir,'FineTune')
    pthfile=os.path.join(wkDir,lstStr[-1])
    pyfile=lstStr[-2]+'.py'
    detDir=lstStr[-3]
    return pthfile,pyfile,detDir
def GetMMDetDir():
    mmdetDir=os.path.dirname(det.__file__)
    mmdetDir=os.path.dirname(mmdetDir)
    return mmdetDir
def ChangeOldTxt(txtlines,oldTxt,repTxt):
    linDex,curLineDex=basFunc.Findtxtlinelst(txtlines,oldTxt)
    bExitFlag=False
    while(curLineDex!=-1):
        if(txtlines[curLineDex].find('import')!=-1 or txtlines[curLineDex].find('from')!=-1):
            bExitFlag=True
            txtlines[curLineDex]=txtlines[curLineDex].replace(oldTxt,repTxt)
        curLineDex+=1
        curLineDex=max(0,curLineDex)
        linDex,curLineDex=basFunc.Findtxtlinelst(txtlines,oldTxt,curLineDex)
    return txtlines,bExitFlag
def NeedChangePyFileDet(files,bUseSetupFlag):
    detDir=GetMMDetDir()
    basname=os.path.basename(detDir)
    cdetDir=os.path.join(os.getcwd(),basname)
    repTxt=' '+basname+'.mmdet'
    if(not os.path.exists(cdetDir)):
        bUseSetupFlag=True
    else:
        if(not bUseSetupFlag):
            print('Use MMDetectDir in Current WorkDir')
        else:
            print('Warning!!! Detect MMDetectDir in Current WorkDir but Not Use')
    bExitFlag=False        
    for f in files:
        file=os.path.join(os.getcwd(),f)
        fp=open(file,'r')
        txtlines=fp.readlines()
        fp.close()
        if(bUseSetupFlag):
            txtlines,bTempFlag=ChangeOldTxt(txtlines,repTxt+' ',' mmdet ')
            if(bTempFlag):bExitFlag=True
            txtlines,bTempFlag=ChangeOldTxt(txtlines,repTxt+'.',' mmdet.')
            if(bTempFlag):bExitFlag=True
        else:
            txtlines,bTempFlag=ChangeOldTxt(txtlines,' mmdet ',repTxt+' ')
            if(bTempFlag):bExitFlag=True
            txtlines,bTempFlag=ChangeOldTxt(txtlines,' mmdet.',repTxt+'.')
            if(bTempFlag):bExitFlag=True
        fp=open(file,'w')
        fp.writelines(txtlines)
        fp.close()
    print('mmdetPath:',GetMMDetDir())
    if(bExitFlag):
        print("First Setup Changed Path may not Right,but the Path changed,everything is Running Normal,Don\'t Worry!")
def GetMMConfigDir(netsDir):
    mmdetDir=GetMMDetDir()
    cfgDir=os.path.join(mmdetDir,'configs')
    return os.path.join(cfgDir,netsDir)
def ChangeDetectFile(pyfile,ndetfile):
    _,name,ftr=basFunc.GetfileDirNamefilter(pyfile)
    name+=ftr
    fp=open(ndetfile,'r')
    txts=fp.readlines()
    fp.close()
    if(len(txts)==0):
        txts=['']
    txts[0]='_base_ = \''+pyfile+'\'\n'
    fp=open(ndetfile,'w')
    fp.writelines(txts)
    fp.close()
def ConvertModelPyfile(pyfile):
    if os.path.isabs(pyfile):
        return pyfile
    dex=pyfile.find('configs')
    if dex<0:
        raise('write modelpyfile is error')
    cfgfile=pyfile[dex:]
    wkDir=os.path.dirname(curDir)
    wkDir=os.path.join(wkDir,'mmdetection')
    pyfile=os.path.join(wkDir,cfgfile)
    if not os.path.exists(pyfile):
        wkDir=os.path.dirname(curDir)
        wkDir=os.path.join(wkDir,'mmyolodet')
        pyfile=os.path.join(wkDir,cfgfile)
    return pyfile
def InitPthandConfig(url,detDir,file=None):
    if(url[:4]=='http'):
        #url=ChangeURL(url)
        pthfile,pyfile,d=GetPthFileName(url)
        if(not os.path.exists(pthfile)):
            print('Not Found PthFile:',pthfile)
            basFunc.DownloadFile(url,pthfile)
        detDir=os.path.join(detDir,d)
        pyfile=os.path.join(detDir,pyfile)   
        if(file is not None):
            pyfile=ConvertModelPyfile(file)
            detDir=os.path.dirname(pyfile)
    else:
        pthfile=url
        pyfile=ConvertModelPyfile(file)
        detDir=os.path.dirname(pyfile)
    print('pthfile:',pthfile)
    print('pyfile:',pyfile)
    print('detDir:',detDir)
    return pthfile,pyfile,detDir
def GetLastLogJsonFile(workDir):
    files=basFunc.getdatas(workDir,'*.json')
    numlst=[]
    nfiles=files
    files=[]
    for i in range(len(nfiles)):
        file=nfiles[i]
        _,name,ftr=basFunc.GetfileDirNamefilter(file)
        name=name.split('.')[0]
        anum=name.split('_')[0]
        if(not anum.isnumeric()):continue
        bnum=name.split('_')[1]
        if not bnum.isnumeric():continue
        nstr=anum+bnum        
        numlst.append(int(nstr))
        files.append(nfiles[i])
    zlist=zip(numlst,files)
    zlist=list(zlist)
    nlist=sorted(zlist,key=lambda x: x[0],reverse=True)
    return nlist[0][1]
def FindBestKeyFromJsFile(jsfile,config,pthDir):
    dctlst=dio.getjsdatlstlindct(jsfile)
    ndctlst=[dct for dct in dctlst if config.SortKey in dct]
    bReverse=False
    if(config.SortKeyMaxMin=='max'):
        bReverse=True
    nSortlst=sorted(ndctlst,key=lambda x: x[config.SortKey],reverse=bReverse)
    if(len(nSortlst)):
        pthfile=os.path.join(pthDir,'epoch_'+str(nSortlst[0]['epoch'])+'.pth')
    else:
        pthfile=os.path.join(pthDir,'latest.pth')
    return pthfile
def WriteCurrentProcess(workDir):
    if(not os.path.exists(os.path.join(workDir,'process.txt'))):
        fp=open(os.path.join(workDir,'process.txt'),'w')
    else:
        fp=open(os.path.join(workDir,'process.txt'),'a')
    fp.write(str(os.getpid())+'\n')
    fp.close()
def GenerateProcess(workDir):
    fp=open(os.path.join(workDir,'process.txt'),'w')
    fp.close()
def GetProcess(workDir):
    if(not os.path.exists(os.path.join(workDir,'process.txt'))):
        return []
    fp=open(os.path.join(workDir,'process.txt'),'r')
    procs=fp.readlines()
    fp.close()
    plist=[]
    for i in range(len(procs)):
        if(procs[i]==''):continue
        plist.append(int(procs[i]))
    return plist
def DumpWorkDirConfig(args):
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume    
    return cfg
def ChangeCfgTrain(cfg,gpus,args,dctcfg):
    if len(args.weightFile)!=0:
        print('use WeightFile:',args.weightFile)
        if args.resumeWeight:
            cfg.resume = True
        else:
            cfg.resume = False
        cfg.load_from = args.weightFile
    else:
        cfg.load_from=None
    if len(gpus)==1:
        cfg.launcher='none'
        dctcfg['launcher']='none'
    else:
        gpucnt=basCfg.GetExtConfigStr(args.train_ext_cfg,'--gpucnt=')
        if(len(gpucnt)>0 and int(gpucnt[0])>1):
            print('------------use cvlauch------------')
            cfg.launcher='cvlaunch'
            dctcfg['launcher']='cvlaunch'
    dataloaders=['train_dataloader','val_dataloader','test_dataloader']
    for loader in dataloaders:
        curloader=eval(f'cfg.{loader}')
        if curloader.num_workers==0:
            exec(f'cfg.{loader}.persistent_workers=False')
    return cfg,dctcfg
if __name__ == "__main__":
    pass



    
    