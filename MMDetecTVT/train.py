import os,sys
curDir=os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curDir,'cvSDK'))
sdkDir=os.path.join(curDir,'cvSDK')
import cvSDK.BasicUseFunc as basFunc
import cvSDK.DataIO as dio
import cvSDK.BasicTorchOperator as basTchOp
import shutil
import cvSDK.BasicConfig as basCfg
#所有操作都要在train.py所在根目录下面执行，不许离开这个目录执行python文件
#训练模式只支持拷贝MMDetection到内部文件夹进行训练，不允许使用外部的MMdetection安装包方式
# 进行训练，但是推理可以使用外部安装的MMDetection进行推理。目前仅支持目标检测XML格式拷贝，
#其他方式暂缓支持
#目前要想使用train.py训练必须将MMDetection安装到pip目录中去才可以
#-------------------------------warning--------------------------------------------
#目前未能搞定重复训练数值精度问题，请后世小子搞之
#风险重复训练有可能会产生不同的MAP,MAP经过设置torch.backends.cudnn俩个选项依旧无法控制
#反向传播的精度，正向传播不会出现精度不足的问题，随机种子已经被固定，也可以从程序打开
#----------------------------------------------------------------------------------
#目前只支持单进程多GPU运算
#1.填写trainConfig.json的modelurl,modelurl从MMDetection/configs文件夹中选择最接近自己心中模型的url放入
#2.选择GPU的数量，并行的参数等如无问题执行python train.py
#3.生成FineTune/MMDetectionConfig.py的配置文件，可以在FineTune/文件夹下把心中的模型下载好放入，可以照示例的
#Cascadercnndet.py修改MMDetectionConfig.py文件,多GPU修改laucher为非none,可以修改WeightFile
#############################这个重点注意##############################################
#4.注意FineTune/MMDetectionConfig.py(注意前两行要空出来，不要添加新的代码到前两行)删除
#这个文件，重新运行python train.py可以新生成一个全新参数的文件
######################################################################################
#5.确认无误后，开始python train.py
#6.调试程序必须使用一个GPU，使用trainMMDetection.py作为调试文件
#---------------------------------------下面这个经常用-------------------------------------------
#7.调试主体循环/usr/local/lib/python3.7/site-packages/mmcv/runner/epoch_based_runner.py第72行左右
#8.调试初始位置是在mmdetection文件夹下mmdet/apis/train.py
#-----------------------------------------------------------------------------------------------
def GetDefaultDict():    
    dct=basCfg.GetDefaultDict(None)
    return dct
cfgfile=os.path.join(os.getcwd(),'trainConfig.json')
config=dio.InitJsonConfig(GetDefaultDict,cfgfile)
dctcfg=config
config=dio.Config(config)
def SetGPUConfig(config):
    gpuflag=False
    gpulst,gpuCount=basFunc.GetAvailableGPUsList()
    if(gpulst==None or gpuCount==0):
        print('No GPU Device Avaliabel,find Device:',gpuCount)
        exit()
    if(len(config.gpuid)!=0):
        gpuflag=True
        clst=[]
        if(len(config.gpuid)==1):
            gpucnt=basCfg.GetExtConfigStr(config.train_ext_cfg,'--gpucnt=')
            if len(gpucnt)==0:
                gpucnt=1
            else:
                gpucnt=int(gpucnt[0])
            if(config.gpuid[0]<0):
                for i in range(min(len(gpulst),abs(config.gpuid[0]*gpucnt))):
                    clst.append(gpulst[i])
            if(config.gpuid[0]>=0):
                clst=config.gpuid
        else:
            clst=config.gpuid
            gpucnt=len(clst)
        lst=basTchOp.ClientGvGPUs(clst)
    return lst,gpuflag,gpucnt
gpus,_,gpucnt=SetGPUConfig(config)
basCfg.ChangeMMDetectionDir(config,dctcfg,cfgfile)
import mmdet as det
import cvSDK.BasicMMDet as basDet
def RefreshWorkDir():    
    workDir=os.path.join(curDir,config.work_dir)
    if(os.path.exists(workDir)):
        shutil.rmtree(workDir)
    basFunc.MakeEmptyDir(workDir)
    debugDir=os.path.join(curDir,'debugDir')
    if os.path.exists(debugDir):
        shutil.rmtree(debugDir)
    return
modelurl=config.modelurl
if(modelurl==''):
    print('-----------warning--------------')
    print('Please Set trainConfig modelurl ')
    print('-----------warning--------------')
    exit()
config,dctcfg=basCfg.GenerateConfigFile(config,cfgfile,dctcfg)
basefile=config.configpyfile
detDir=basFunc.GetCurrentFileDir(det.__file__)
detCfgDir=os.path.join(os.path.dirname(detDir),'configs')
pthfile,pyfile,_=basDet.InitPthandConfig(modelurl,detCfgDir,config.modelpyfile)
workDir=os.path.join(curDir,config.work_dir)
RefreshWorkDir()
basDet.GenerateProcess(workDir)
def TrainProcess():
    if(not basCfg.CheckFineTuneFile(config,10)):
        basDet.ChangeDetectFile(pyfile,config.configpyfile)
    basFunc.ChangeTxtFileContent(basefile,basCfg.ChangePyfileContent)
    gpucount=basCfg.GetExtConfigStr(config.train_ext_cfg,'--gpucnt=')
    if len(config.train_ext_cfg)!=0 and len(gpucount)>0 and int(gpucount[0])>1:
        cmd=config.PythonExe+' -m cvSDK.launch_dist --nproc_per_node='+str(len(gpus)//gpucnt)\
            +' --master_port='+str(config.MASTER_PORT)+' --nnodes='+str(config.NNODES)+\
            ' --node_rank='+str(config.NODE_RANK)+' --master_addr='+str(config.MASTER_ADDR)\
            +' '+config.train_ext_cfg+' trainMMDetection.py'
    else:
        if(len(gpus)==1):
            cmd=config.PythonExe+' trainMMDetection.py'
        else:
            cmd=config.train_startup_ddp+' --nproc_per_node='+str(len(gpus))\
                +' --master_port='+str(config.MASTER_PORT)+' --nnodes='+str(config.NNODES)+\
                ' --node_rank='+str(config.NODE_RANK)+' --master_addr='+str(config.MASTER_ADDR)\
                +' trainMMDetection.py'        
    print('cmd:',cmd)
    os.system(cmd)
    return
TrainProcess()
