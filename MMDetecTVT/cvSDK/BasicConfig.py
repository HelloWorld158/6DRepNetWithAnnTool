import os,sys
import BasicUseFunc as basFunc
import DataIO as dio
def GetDefaultDict(mode):    
    dct={}
    dct['gpuid']=[-1]
    dct['WorkModelDir']=[]#当前目录里面的MMDetection，为空时程序会自己找
    dct['modelurl']=''#请到MMDetection的目录下面config的文件夹下找对应模型的url
    dct['modelpyfile']=None
    dct['seed']=1024#随机种子设置，注意目前反向传播精度没搞定
    dct['PythonExe']='python3'
    dct['configpyfile']=''#请输入configpyfile的文件一般在FineTune文件夹下面
    dct['work_dir']='workDir'
    dct['amp']=False
    dct['auto-scale-lr']=False
    dct['weightFile']=''
    dct['resumeWeight']=False   
    dct['TestWeightFile']=''
    dct['launcher']='pytorch'#'none', 'pytorch', 'slurm', 'mpi' 
    dct['tta']=False   
    dct['PyConfigOption']=None #'override some settings in the used config, the key-value pair '
                                #'in xxx=yyy format will be merged into config file.'
                                #用来覆盖configpyfile内的运行文件，如果运行文件内的命令与这个重复，运行文件内的
                                #设置会被覆盖
    dct['LocalRank']=0
    dct['MASTER_ADDR']='127.0.0.1'
    dct['MASTER_PORT']=29500
    dct['LOCAL_RANK']=0
    dct['NPROC_PER_NODE']=1
    dct['NNODES']=1
    dct['NODE_RANK']=0
    dct['train_ext_cfg']=''#torch.dist.launch的额外参数
    '''
    #2.0之后
    dct['train_startup_ddp']='torchrun'
    #2.0之前
    dct['train_startup_ddp']='python3 -m torch.distributed.launch'
    #2.0之后另一种写法
    dct['train_startup_ddp']='python3 -m torch.distributed.run'
    '''
    dct['train_startup_ddp']='torchrun'
    dct['Infer_UseFile']=False
    dct['Infer_WeightFile']=''
    return dct
def GetArgsFromTrainConfig(dct):
    matchdct={
        'configpyfile':'config',
        'work_dir':'work_dir',
        'amp':'amp',
        'auto-scale-lr':'auto_scale_lr',
        'PyConfigOption':'cfg_options',
        'launcher':'launcher',
        'LocalRank':'local_rank'
    }
    argdct={}
    for kk,vv in matchdct.items():
        argdct[vv]=dct[kk]
    argdct['resume']=None
    args=dio.Config(argdct)
    return args
def GetArgsFromTestConfig(dct,pthfile,gpus):
    matchdct={
        'configpyfile':'config',
        'work_dir':'work_dir',
        'PyConfigOption':'cfg_options',
        'launcher':'launcher',
        'LocalRank':'local_rank',
        'tta':'tta'
    }
    argdct={}
    for kk,vv in matchdct.items():
        argdct[vv]=dct[kk]
    argdct['checkpoint']=pthfile
    if len(gpus)==1:
        argdct['launcher']='none'
    if argdct['launcher']=='cvlaunch':
        argdct['launcher']='pytorch'
    argdct['show']=None
    argdct['show_dir']=None
    argdct['out']=None
    args=dio.Config(argdct)
    return args
def CheckDetDir(config,cfgdct,trainCfg):
    if(config.OriMMdetDir in sys.path):return
    import BasicMMDet as basDet
    cfgdct['OriMMdetDir']=basDet.GetMMDetDir()
    dio.writejsondictFormatFile(cfgdct,trainCfg)
    print('===========================================')
    print('Change MMDetection Setup Dir,Please Restart')
    print('===========================================')
    exit()
def GetDefaultBestPth(curDir,config,weightFile):
    if(weightFile==''):
        workDir=os.path.join(curDir,config.work_dir)
        weightFile=os.path.join(workDir,'best.pth')
    return weightFile
def WriteConfig(cfgfile):
    fp=open(cfgfile,'w')
    for i in range(3):
        fp.write(' \n')
    fp.close()
def GenerateConfigFile(config,cfgfile,dctcfg):
    if(dctcfg['configpyfile']!='' and os.path.exists(dctcfg['configpyfile'])):
        return config,dctcfg
    dctcfg['configpyfile']='FineTune/MMDetectionConfig.py'
    config.configpyfile=dctcfg['configpyfile']
    WriteConfig(dctcfg['configpyfile'])
    dio.writejsondictFormatFile(dctcfg,cfgfile)
    return config,dctcfg
def ChangeMMDetectionDir(config,dctcfg,cfgfile):
    dirnames=[os.path.basename(config.WorkModelDir[i]) for i in range(len(config.WorkModelDir))]
    procDir=os.path.dirname(basFunc.GetCurrentFileDir(__file__))
    mmNDirs=[os.path.join(procDir,dirnames[i]) for i in range(len(dirnames))]
    for mmNDir in mmNDirs:
        if(not os.path.exists(mmNDir)):
            print('Please reconfig trainConfig.json:WorkModelDir,last dir must be mmdetection')
            exit()
    config.WorkModelDir=mmNDirs
    dctcfg['WorkModelDir']=mmNDirs
    dio.writejsondictFormatFile(dctcfg,cfgfile)
    svpath=[]
    empty=False
    for path in sys.path:
        detlst=['mmdet','mmyolo']
        flag=True
        if path=='':
            empty=True
            continue
        for det in detlst:
            curMMpth=os.path.join(path,det)
            if os.path.exists(curMMpth):
                flag=False
        if not flag:continue
        if path not in svpath:
            svpath.append(path)
    if empty:
        svpath.insert(0,'')
    for i in range(len(config.WorkModelDir)):
        svpath.append(config.WorkModelDir[i])
    sys.path=list(svpath)
    print('sys path:\n',sys.path)
    return config
def ChangeFineTune(config,curDir):
    _,name,ftr=basFunc.GetfileDirNamefilter(config.configpyfile)
    wkNDir=os.path.join(curDir,config.work_dir)
    dataDir=os.path.dirname(curDir)
    wkfile=os.path.join(wkNDir,name+ftr)
    fp=open(wkfile,'r')
    txts=fp.readlines()
    fp.close()
    ntxts=[]    
    ntxts.append('\n')
    ntxts.append('workDir=\''+dataDir+'/\'\n')
    ntxts.append('#############################################################\n')
    ntxts.append('######################特别注意################################\n')
    ntxts.append('#前两行预留，可以什么都不写，但是必须留出来，否则程序会覆盖前两行\n')
    ntxts.append('####可以修改分数阈值，可以修改nms###############################\n')
    ntxts.append('##############################################################\n')
    ntxts+=txts
    newfile=os.path.join(os.path.join(curDir,'FineTune'),name+ftr)
    fp=open(newfile,'w')
    fp.writelines(ntxts)
    fp.close()
    return
def CheckFineTuneDir(config,curDir):
    fp=open(config.configpyfile,'r')
    txts=fp.readlines()
    fp.close()
    flag=False
    if(len(txts)>10):
        flag=True
    if(not flag):
        ChangeFineTune(config,curDir)
        print('---------------------------------confirm---------------------------------------------')
        print('Please Change MMDetectionConfig.py in FineTune,Steady in train,Nextime will Train')
        print('-------------------------------------------------------------------------------------')
        exit()
    return
def CheckFineTuneFile(config,length):
    fp=open(config.configpyfile,'r')
    txts=fp.readlines()
    fp.close()
    flag=False
    if(len(txts)>length):
        flag=True
    return flag
def ChangePyfileContent(lines):
    if(len(lines)<2):return lines
    wkDir=os.path.dirname(basFunc.GetCurrentFileDir(__file__))
    wkDir=os.path.dirname(wkDir)
    lines[1]='workDir=\''+wkDir+'/\''
    return lines
def GetExtConfigStr(traincfg,strcmd):
    import re
    reslst=re.findall(strcmd+'(\S)',traincfg)
    return reslst